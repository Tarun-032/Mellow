//! System cursor sampling and the cursor-following bone guide.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use tauri::{
    AppHandle, Emitter, LogicalSize, Manager, PhysicalPosition, PhysicalSize, State,
};

use crate::PetMonitor;

const CURSOR: &str = "cursor";
const GUIDE_ARRIVED: &str = "guide-arrived";
const INTERVAL: Duration = Duration::from_millis(16);
const MONITOR_REFRESH: Duration = Duration::from_secs(2);

const FOLLOW_OFFSET_X: f64 = 42.0;
const FOLLOW_OFFSET_Y: f64 = 34.0;
const EDGE_MARGIN: f64 = 8.0;
const EDGE_HYSTERESIS: f64 = 24.0;
const RETURN_CANCEL_DISTANCE: f64 = 24.0;
const REDUCED_RETURN_RETARGET_DISTANCE: f64 = 4.0;
const DIALOGUE_WIDTH: f64 = 420.0;
const DIALOGUE_HEIGHT: f64 = 220.0;
// Gap so the explanation clears the bone sprite.
const DIALOGUE_GAP_Y: f64 = 22.0;
// Soft follow spring: trail briefly, then settle.
const FOLLOW_FREQUENCY: f64 = 3.3;
const FOLLOW_DAMPING: f64 = 0.82;

#[derive(Clone, serde::Serialize)]
struct Cursor {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn distance(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

#[derive(Clone, Copy, Debug)]
struct Screen {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
    scale: f64,
}

impl Screen {
    fn contains(self, point: Point) -> bool {
        point.x >= self.left && point.x < self.right && point.y >= self.top && point.y < self.bottom
    }

    fn clamp(self, point: Point, width: f64, height: f64) -> Point {
        Point {
            x: point.x.clamp(
                self.left + EDGE_MARGIN * self.scale,
                self.right - width - EDGE_MARGIN * self.scale,
            ),
            y: point.y.clamp(
                self.top + EDGE_MARGIN * self.scale,
                self.bottom - height - EDGE_MARGIN * self.scale,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlightKind {
    Target,
    Return,
}

#[derive(Debug)]
struct Flight {
    kind: FlightKind,
    revision: u64,
    start: Point,
    end: Point,
    control: Point,
    cursor_at_start: Point,
    started: Instant,
    duration: f64,
}

#[derive(Debug)]
enum Phase {
    Following,
    Flying(Flight),
    Dwelling { at: Point },
}

#[derive(Debug)]
struct GuideMotion {
    position: Point,
    velocity: Point,
    initialized: bool,
    ready: bool,
    quiet: bool,
    reduced_motion: bool,
    revision: u64,
    width: f64,
    height: f64,
    tip_x: f64,
    tip_y: f64,
    side_x: f64,
    side_y: f64,
    phase: Phase,
}

impl Default for GuideMotion {
    fn default() -> Self {
        Self {
            position: Point::default(),
            velocity: Point::default(),
            initialized: false,
            ready: false,
            quiet: false,
            reduced_motion: false,
            revision: 0,
            width: 36.0,
            height: 36.0,
            tip_x: 10.0,
            tip_y: 10.0,
            side_x: 1.0,
            side_y: 1.0,
            phase: Phase::Following,
        }
    }
}

#[derive(Clone, Default)]
pub struct GuideState {
    motion: Arc<Mutex<GuideMotion>>,
    runtime: Arc<Mutex<GuideRuntime>>,
    tick_pending: Arc<AtomicBool>,
}

#[derive(Debug)]
struct GuideRuntime {
    screens: Vec<Screen>,
    refreshed: Instant,
    last_tick: Instant,
    last_cursor: (f64, f64),
    last_window_position: (i32, i32),
    last_scale: f64,
}

impl Default for GuideRuntime {
    fn default() -> Self {
        Self {
            screens: Vec::new(),
            refreshed: Instant::now(),
            last_tick: Instant::now(),
            last_cursor: (f64::NAN, f64::NAN),
            last_window_position: (i32::MIN, i32::MIN),
            last_scale: f64::NAN,
        }
    }
}

#[derive(serde::Serialize)]
pub struct GuideAck {
    accepted: bool,
    arrived: bool,
}

#[derive(Clone, serde::Serialize)]
struct GuideArrived {
    revision: u64,
}

fn screens(app: &AppHandle) -> Vec<Screen> {
    app.available_monitors()
        .unwrap_or_default()
        .into_iter()
        .map(|monitor| {
            let position = monitor.position();
            let size = monitor.size();
            Screen {
                left: f64::from(position.x),
                top: f64::from(position.y),
                right: f64::from(position.x) + f64::from(size.width),
                bottom: f64::from(position.y) + f64::from(size.height),
                scale: monitor.scale_factor(),
            }
        })
        .collect()
}

fn screen_for(point: Point, screens: &[Screen]) -> Option<Screen> {
    screens
        .iter()
        .copied()
        .find(|screen| screen.contains(point))
        .or_else(|| {
            screens.iter().copied().min_by(|a, b| {
                let distance = |screen: Screen| {
                    let x = point.x.clamp(screen.left, screen.right);
                    let y = point.y.clamp(screen.top, screen.bottom);
                    (point.x - x).hypot(point.y - y)
                };
                distance(*a).total_cmp(&distance(*b))
            })
        })
}

fn follow_destination(cursor: Point, screen: Screen, guide: &mut GuideMotion) -> Point {
    let width = guide.width * screen.scale;
    let height = guide.height * screen.scale;
    let tip_x = guide.tip_x * screen.scale;
    let tip_y = guide.tip_y * screen.scale;
    let margin = EDGE_MARGIN * screen.scale;
    let hysteresis = EDGE_HYSTERESIS * screen.scale;

    let candidate = |side_x: f64, side_y: f64| Point {
        x: cursor.x + side_x * FOLLOW_OFFSET_X * screen.scale - tip_x,
        y: cursor.y + side_y * FOLLOW_OFFSET_Y * screen.scale - tip_y,
    };

    let right = candidate(1.0, guide.side_y);
    if guide.side_x > 0.0 && right.x + width > screen.right - margin {
        guide.side_x = -1.0;
    } else if guide.side_x < 0.0
        && right.x >= screen.left + margin + hysteresis
        && right.x + width <= screen.right - margin - hysteresis
    {
        guide.side_x = 1.0;
    }

    let below = candidate(guide.side_x, 1.0);
    if guide.side_y > 0.0 && below.y + height > screen.bottom - margin {
        guide.side_y = -1.0;
    } else if guide.side_y < 0.0
        && below.y >= screen.top + margin + hysteresis
        && below.y + height <= screen.bottom - margin - hysteresis
    {
        guide.side_y = 1.0;
    }

    screen.clamp(candidate(guide.side_x, guide.side_y), width, height)
}

fn target_destination(nx: f64, ny: f64, screen: Screen, guide: &GuideMotion) -> Point {
    Point {
        x: screen.left + nx.clamp(0.0, 1.0) * (screen.right - screen.left)
            - guide.tip_x * screen.scale,
        y: screen.top + ny.clamp(0.0, 1.0) * (screen.bottom - screen.top)
            - guide.tip_y * screen.scale,
    }
}

fn visibility_score(point: Point, screens: &[Screen]) -> f64 {
    screens
        .iter()
        .map(|screen| {
            if screen.contains(point) {
                (point.x - screen.left)
                    .min(screen.right - point.x)
                    .min(point.y - screen.top)
                    .min(screen.bottom - point.y)
            } else {
                let x = point.x.clamp(screen.left, screen.right);
                let y = point.y.clamp(screen.top, screen.bottom);
                -Point { x, y }.distance(point)
            }
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

fn flight_control(start: Point, end: Point, screens: &[Screen], scale: f64) -> Point {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let distance = dx.hypot(dy);
    let midpoint = Point {
        x: (start.x + end.x) * 0.5,
        y: (start.y + end.y) * 0.5,
    };
    if distance < 1.0 {
        return midpoint;
    }
    let arc = (distance * 0.16).min(90.0 * scale);
    let normal = Point {
        x: -dy / distance,
        y: dx / distance,
    };
    let a = Point {
        x: midpoint.x + normal.x * arc,
        y: midpoint.y + normal.y * arc,
    };
    let b = Point {
        x: midpoint.x - normal.x * arc,
        y: midpoint.y - normal.y * arc,
    };
    if visibility_score(a, screens) >= visibility_score(b, screens) {
        a
    } else {
        b
    }
}

fn flight_duration(
    start: Point,
    end: Point,
    scale: f64,
    kind: FlightKind,
    reduced_motion: bool,
) -> f64 {
    let logical_distance = start.distance(end) / scale.max(0.5);
    if reduced_motion {
        return match kind {
            FlightKind::Target => (0.32 + logical_distance / 7800.0).clamp(0.32, 0.55),
            FlightKind::Return => (0.28 + logical_distance / 9000.0).clamp(0.28, 0.48),
        };
    }
    match kind {
        FlightKind::Target => (0.50 + logical_distance / 3400.0).clamp(0.50, 1.10),
        FlightKind::Return => (0.40 + logical_distance / 4200.0).clamp(0.40, 0.75),
    }
}

fn begin_flight(
    guide: &mut GuideMotion,
    end: Point,
    kind: FlightKind,
    revision: u64,
    cursor: Point,
    screens: &[Screen],
    scale: f64,
) {
    let start = guide.position;
    // Reduced motion: short straight path, no decorative arc.
    let control = if guide.reduced_motion {
        Point {
            x: (start.x + end.x) * 0.5,
            y: (start.y + end.y) * 0.5,
        }
    } else {
        flight_control(start, end, screens, scale)
    };
    guide.phase = Phase::Flying(Flight {
        kind,
        revision,
        start,
        end,
        control,
        cursor_at_start: cursor,
        started: Instant::now(),
        duration: flight_duration(start, end, scale, kind, guide.reduced_motion),
    });
    guide.velocity = Point::default();
}

fn smootherstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn quadratic(start: Point, control: Point, end: Point, t: f64) -> Point {
    let one_minus = 1.0 - t;
    Point {
        x: one_minus * one_minus * start.x + 2.0 * one_minus * t * control.x + t * t * end.x,
        y: one_minus * one_minus * start.y + 2.0 * one_minus * t * control.y + t * t * end.y,
    }
}

fn step_spring(value: &mut f64, velocity: &mut f64, target: f64, dt: f64) {
    let omega = std::f64::consts::PI * 2.0 * FOLLOW_FREQUENCY;
    let f = 1.0 + 2.0 * dt * FOLLOW_DAMPING * omega;
    let oo = omega * omega;
    let hoo = dt * oo;
    let hhoo = dt * hoo;
    let inverse = 1.0 / (f + hhoo);
    let previous = *value;
    let previous_velocity = *velocity;
    *value = (f * previous + dt * previous_velocity + hhoo * target) * inverse;
    *velocity = (previous_velocity + hoo * (target - previous)) * inverse;
}

fn pet_screen(app: &AppHandle) -> Result<Screen, String> {
    let pet = app
        .get_webview_window("pet")
        .ok_or_else(|| "pet window is missing".to_string())?;
    let monitor = pet
        .current_monitor()
        .map_err(|error| error.to_string())?
        .or_else(|| app.primary_monitor().ok().flatten())
        .ok_or_else(|| "no monitor is available".to_string())?;
    let position = monitor.position();
    let size = monitor.size();
    Ok(Screen {
        left: f64::from(position.x),
        top: f64::from(position.y),
        right: f64::from(position.x) + f64::from(size.width),
        bottom: f64::from(position.y) + f64::from(size.height),
        scale: monitor.scale_factor(),
    })
}

fn reported_screen(monitor: PetMonitor, screens: &[Screen]) -> Option<Screen> {
    let right = f64::from(monitor.left) + f64::from(monitor.width);
    let bottom = f64::from(monitor.top) + f64::from(monitor.height);
    screens.iter().copied().find(|screen| {
        screen.left == f64::from(monitor.left)
            && screen.top == f64::from(monitor.top)
            && screen.right == right
            && screen.bottom == bottom
    })
}

fn hide_dialogue(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("guide-bubble") {
        let _ = window.hide();
    }
}

fn current_cursor(app: &AppHandle) -> Result<Point, String> {
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    Ok(Point {
        x: cursor.x,
        y: cursor.y,
    })
}

fn initialize_at_cursor(cursor: Point, guide: &mut GuideMotion, screens: &[Screen]) {
    if guide.initialized {
        return;
    }
    if let Some(screen) = screen_for(cursor, screens) {
        guide.position = follow_destination(cursor, screen, guide);
        guide.initialized = true;
    }
}

#[tauri::command]
pub fn guide_ready(
    app: AppHandle,
    state: State<'_, GuideState>,
    width: f64,
    height: f64,
    tip_x: f64,
    tip_y: f64,
    reduced_motion: bool,
) -> Result<(), String> {
    let all_screens = screens(&app);
    // Cursor may fail during WebView startup; first successful tick places it.
    let cursor = current_cursor(&app).ok();
    let mut guide = state.motion.lock().map_err(|_| "guide state poisoned")?;
    let first_ready = !guide.ready;
    guide.width = width.max(1.0);
    guide.height = height.max(1.0);
    guide.tip_x = tip_x.clamp(0.0, guide.width);
    guide.tip_y = tip_y.clamp(0.0, guide.height);
    guide.reduced_motion = reduced_motion;
    guide.ready = true;
    // First ready only: reset to follow (StrictMode must not clear a live point).
    if first_ready {
        guide.phase = Phase::Following;
        guide.velocity = Point::default();
        guide.initialized = false;
    }
    if let Some(cursor) = cursor {
        initialize_at_cursor(cursor, &mut guide, &all_screens);
    }
    drop(guide);
    if let Some(window) = app.get_webview_window("guide") {
        window
            .set_always_on_top(true)
            .map_err(|error| error.to_string())?;
    }
    if first_ready {
        hide_dialogue(&app);
    }
    Ok(())
}

#[tauri::command]
pub fn guide_set_target(
    app: AppHandle,
    state: State<'_, GuideState>,
    revision: u64,
    nx: f64,
    ny: f64,
    monitor: Option<PetMonitor>,
) -> Result<GuideAck, String> {
    let all_screens = screens(&app);
    let cursor = current_cursor(&app)?;
    // Resolve capture monitor against current OS list (may differ from pet's).
    let target_screen = monitor
        .and_then(|reported| reported_screen(reported, &all_screens))
        .map(Ok)
        .unwrap_or_else(|| pet_screen(&app))?;
    let mut guide = state.motion.lock().map_err(|_| "guide state poisoned")?;
    if revision <= guide.revision {
        return Ok(GuideAck {
            accepted: false,
            arrived: false,
        });
    }
    guide.revision = revision;
    initialize_at_cursor(cursor, &mut guide, &all_screens);
    let destination = target_destination(nx, ny, target_screen, &guide);
    if guide.position.distance(destination) <= 2.0 * target_screen.scale {
        guide.position = destination;
        guide.velocity = Point::default();
        guide.phase = Phase::Dwelling { at: destination };
        drop(guide);
        hide_dialogue(&app);
        let _ = app.emit(GUIDE_ARRIVED, GuideArrived { revision });
        return Ok(GuideAck {
            accepted: true,
            arrived: true,
        });
    }
    begin_flight(
        &mut guide,
        destination,
        FlightKind::Target,
        revision,
        cursor,
        &all_screens,
        target_screen.scale,
    );
    drop(guide);
    hide_dialogue(&app);
    Ok(GuideAck {
        accepted: true,
        arrived: false,
    })
}

#[tauri::command]
pub fn guide_clear(
    app: AppHandle,
    state: State<'_, GuideState>,
    revision: u64,
) -> Result<GuideAck, String> {
    let all_screens = screens(&app);
    let cursor = current_cursor(&app)?;
    let cursor_screen =
        screen_for(cursor, &all_screens).ok_or_else(|| "no monitor is available".to_string())?;
    let mut guide = state.motion.lock().map_err(|_| "guide state poisoned")?;
    if revision <= guide.revision {
        return Ok(GuideAck {
            accepted: false,
            arrived: false,
        });
    }
    guide.revision = revision;
    initialize_at_cursor(cursor, &mut guide, &all_screens);
    let destination = follow_destination(cursor, cursor_screen, &mut guide);
    if matches!(guide.phase, Phase::Following)
        || guide.position.distance(destination) <= 2.0 * cursor_screen.scale
    {
        guide.position = destination;
        guide.velocity = Point::default();
        guide.phase = Phase::Following;
        drop(guide);
        hide_dialogue(&app);
        return Ok(GuideAck {
            accepted: true,
            arrived: true,
        });
    }
    begin_flight(
        &mut guide,
        destination,
        FlightKind::Return,
        revision,
        cursor,
        &all_screens,
        cursor_screen.scale,
    );
    drop(guide);
    hide_dialogue(&app);
    Ok(GuideAck {
        accepted: true,
        arrived: false,
    })
}

/// Show/hide the pointing explanation in its own native window.
#[tauri::command]
pub fn guide_set_dialogue(
    app: AppHandle,
    visible: bool,
    monitor: Option<PetMonitor>,
    nx: Option<f64>,
    ny: Option<f64>,
    side: Option<String>,
    lift: Option<String>,
) -> Result<(), String> {
    let window = app
        .get_webview_window("guide-bubble")
        .ok_or_else(|| "guide bubble window is missing".to_string())?;
    if !visible {
        let _ = window.hide();
        return Ok(());
    }

    let all_screens = screens(&app);
    let screen = monitor
        .and_then(|reported| reported_screen(reported, &all_screens))
        .map(Ok)
        .unwrap_or_else(|| pet_screen(&app))?;
    let target = Point {
        x: screen.left + nx.unwrap_or(0.5).clamp(0.0, 1.0) * (screen.right - screen.left),
        y: screen.top + ny.unwrap_or(0.5).clamp(0.0, 1.0) * (screen.bottom - screen.top),
    };
    let width = DIALOGUE_WIDTH * screen.scale;
    let height = DIALOGUE_HEIGHT * screen.scale;
    let gap_y = DIALOGUE_GAP_Y * screen.scale;
    let right = side.as_deref() == Some("right");
    let below = lift.as_deref() == Some("below");
    let x = if right { target.x } else { target.x - width }
        .clamp(screen.left, screen.right - width);
    let y = if below {
        target.y + gap_y
    } else {
        target.y - height - gap_y
    }
        .clamp(screen.top, screen.bottom - height);

    window
        .set_size(PhysicalSize::new(width.round() as u32, height.round() as u32))
        .map_err(|error| error.to_string())?;
    window
        .set_position(PhysicalPosition::new(x.round() as i32, y.round() as i32))
        .map_err(|error| error.to_string())?;
    window
        .set_always_on_top(true)
        .map_err(|error| error.to_string())?;
    window.show().map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn guide_set_quiet(
    app: AppHandle,
    state: State<'_, GuideState>,
    quiet: bool,
) -> Result<(), String> {
    state
        .motion
        .lock()
        .map_err(|_| "guide state poisoned")?
        .quiet = quiet;
    if quiet {
        hide_dialogue(&app);
    }
    Ok(())
}

#[tauri::command]
pub fn guide_set_reduced_motion(
    state: State<'_, GuideState>,
    reduced_motion: bool,
) -> Result<(), String> {
    state
        .motion
        .lock()
        .map_err(|_| "guide state poisoned")?
        .reduced_motion = reduced_motion;
    Ok(())
}

/// One cursor/guide frame on the main thread (window APIs must not run off-thread).
fn tick(app: &AppHandle, state: &GuideState) {
    let Some(pet) = app.get_webview_window("pet") else {
        return;
    };
    let Some(guide_window) = app.get_webview_window("guide") else {
        return;
    };
    let Ok(cursor_position) = app.cursor_position() else {
        return;
    };
    let cursor = Point {
        x: cursor_position.x,
        y: cursor_position.y,
    };

    let all_screens = {
        let mut runtime = state
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if runtime.screens.is_empty()
            || runtime.refreshed.elapsed() >= MONITOR_REFRESH
            || screen_for(cursor, &runtime.screens).is_none()
        {
            runtime.screens = screens(app);
            runtime.refreshed = Instant::now();
        }
        runtime.screens.clone()
    };

    if let Ok(Some(monitor)) = pet.current_monitor() {
        let origin = monitor.position();
        let scale = monitor.scale_factor();
        let current = (
            (cursor.x - f64::from(origin.x)) / scale,
            (cursor.y - f64::from(origin.y)) / scale,
        );
        let changed = {
            let mut runtime = state
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if current == runtime.last_cursor {
                false
            } else {
                runtime.last_cursor = current;
                true
            }
        };
        if changed {
            let _ = app.emit(
                CURSOR,
                Cursor {
                    x: current.0,
                    y: current.1,
                },
            );
        }
    }

    let Some(cursor_screen) = screen_for(cursor, &all_screens) else {
        return;
    };
    let now = Instant::now();
    let dt = {
        let mut runtime = state
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dt = now.duration_since(runtime.last_tick).as_secs_f64().min(0.05);
        runtime.last_tick = now;
        dt
    };
    let mut arrival = None;
    let (position, ready, quiet, width, height) = {
        let mut motion = state
            .motion
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        initialize_at_cursor(cursor, &mut motion, &all_screens);
        let old_position = motion.position;

        // Own the phase this tick to transition without borrow conflicts.
        let phase = std::mem::replace(&mut motion.phase, Phase::Following);
        match phase {
            Phase::Following => {
                let target = follow_destination(cursor, cursor_screen, &mut motion);
                if motion.reduced_motion {
                    motion.position = target;
                    motion.velocity = Point::default();
                } else {
                    let mut x = motion.position.x;
                    let mut vx = motion.velocity.x;
                    step_spring(&mut x, &mut vx, target.x, dt);
                    motion.position.x = x;
                    motion.velocity.x = vx;
                    let mut y = motion.position.y;
                    let mut vy = motion.velocity.y;
                    step_spring(&mut y, &mut vy, target.y, dt);
                    motion.position.y = y;
                    motion.velocity.y = vy;
                }
            }
            Phase::Dwelling { at } => {
                // The frontend clears this at actual speech completion.
                motion.position = at;
                motion.velocity = Point::default();
                motion.phase = Phase::Dwelling { at };
            }
            Phase::Flying(flight) => {
                let return_retarget_distance = if motion.reduced_motion {
                    REDUCED_RETURN_RETARGET_DISTANCE
                } else {
                    RETURN_CANCEL_DISTANCE
                };
                if flight.kind == FlightKind::Return
                    && cursor.distance(flight.cursor_at_start)
                        > return_retarget_distance * cursor_screen.scale
                {
                    let target = follow_destination(cursor, cursor_screen, &mut motion);
                    if motion.reduced_motion {
                        // Retarget with another short line (Following would snap).
                        begin_flight(
                            &mut motion,
                            target,
                            FlightKind::Return,
                            flight.revision,
                            cursor,
                            &all_screens,
                            cursor_screen.scale,
                        );
                    } else {
                        motion.phase = Phase::Following;
                        motion.velocity = Point {
                            x: (motion.position.x - old_position.x) / dt.max(1.0 / 240.0),
                            y: (motion.position.y - old_position.y) / dt.max(1.0 / 240.0),
                        };
                        let mut x = motion.position.x;
                        let mut vx = motion.velocity.x;
                        step_spring(&mut x, &mut vx, target.x, dt);
                        motion.position.x = x;
                        motion.velocity.x = vx;
                        let mut y = motion.position.y;
                        let mut vy = motion.velocity.y;
                        step_spring(&mut y, &mut vy, target.y, dt);
                        motion.position.y = y;
                        motion.velocity.y = vy;
                    }
                } else {
                    let progress = now.duration_since(flight.started).as_secs_f64()
                        / flight.duration.max(0.001);
                    motion.position = quadratic(
                        flight.start,
                        flight.control,
                        flight.end,
                        smootherstep(progress),
                    );
                    motion.velocity = Point {
                        x: (motion.position.x - old_position.x) / dt.max(1.0 / 240.0),
                        y: (motion.position.y - old_position.y) / dt.max(1.0 / 240.0),
                    };
                    if progress >= 1.0 {
                        let kind = flight.kind;
                        let revision = flight.revision;
                        let end = flight.end;
                        motion.position = end;
                        motion.velocity = Point::default();
                        if kind == FlightKind::Target {
                            motion.phase = Phase::Dwelling { at: end };
                            arrival = Some(revision);
                        } else {
                            motion.phase = Phase::Following;
                        }
                    } else {
                        motion.phase = Phase::Flying(flight);
                    }
                }
            }
        }
        (
            motion.position,
            motion.ready,
            motion.quiet,
            motion.width,
            motion.height,
        )
    };

    if let Some(revision) = arrival {
        let _ = app.emit(GUIDE_ARRIVED, GuideArrived { revision });
    }

    let should_show = ready && !quiet && pet.is_visible().unwrap_or(false);
    let currently_visible = guide_window.is_visible().unwrap_or(false);
    if !should_show {
        if currently_visible {
            let _ = guide_window.hide();
            hide_dialogue(app);
        }
        return;
    }

    let x = position.x.round() as i32;
    let y = position.y.round() as i32;
    let move_window = {
        let mut runtime = state
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if (x, y) == runtime.last_window_position {
            false
        } else {
            runtime.last_window_position = (x, y);
            true
        }
    };
    if move_window {
        let _ = guide_window.set_position(PhysicalPosition::new(x, y));
    }

    let guide_center = Point {
        x: position.x + width * cursor_screen.scale * 0.5,
        y: position.y + height * cursor_screen.scale * 0.5,
    };
    if let Some(screen) = screen_for(guide_center, &all_screens) {
        let resize = {
            let mut runtime = state
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if (screen.scale - runtime.last_scale).abs() <= 0.001 {
                false
            } else {
                runtime.last_scale = screen.scale;
                true
            }
        };
        if resize {
            let _ = guide_window.set_size(LogicalSize::new(width, height));
        }
    }

    // Position before show; reassert topmost (Windows may drop z-order while hidden).
    if !currently_visible {
        let _ = guide_window.set_always_on_top(true);
        let _ = guide_window.show();
    }
}

/// Timer thread schedules; main thread runs. `tick_pending` coalesces backlog.
pub fn spawn(app: AppHandle, state: GuideState) {
    thread::Builder::new()
        .name("mellow-cursor-follow".into())
        .spawn(move || loop {
            thread::sleep(INTERVAL);
            if state.tick_pending.swap(true, Ordering::AcqRel) {
                continue;
            }
            let tick_app = app.clone();
            let tick_state = state.clone();
            let scheduled = app.run_on_main_thread(move || {
                let result = catch_unwind(AssertUnwindSafe(|| tick(&tick_app, &tick_state)));
                if result.is_err() {
                    eprintln!("cursor follower tick panicked; recovering on the next frame");
                    tick_state.motion.clear_poison();
                    tick_state.runtime.clear_poison();
                }
                tick_state.tick_pending.store(false, Ordering::Release);
            });
            if scheduled.is_err() {
                state.tick_pending.store(false, Ordering::Release);
                return;
            }
        })
        .expect("could not start cursor follower");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen() -> Screen {
        Screen {
            left: 0.0,
            top: 0.0,
            right: 1920.0,
            bottom: 1080.0,
            scale: 1.0,
        }
    }

    #[test]
    fn follow_flips_at_edges_and_keeps_clear_of_them() {
        let mut guide = GuideMotion::default();
        let right = follow_destination(
            Point {
                x: 1915.0,
                y: 500.0,
            },
            screen(),
            &mut guide,
        );
        assert_eq!(guide.side_x, -1.0);
        assert!(right.x + guide.width <= screen().right - EDGE_MARGIN);
        let bottom = follow_destination(
            Point {
                x: 800.0,
                y: 1075.0,
            },
            screen(),
            &mut guide,
        );
        assert_eq!(guide.side_y, -1.0);
        assert!(bottom.y + guide.height <= screen().bottom - EDGE_MARGIN);
    }

    #[test]
    fn target_hotspot_is_exact_and_clamped() {
        let guide = GuideMotion::default();
        let target = target_destination(2.0, -1.0, screen(), &guide);
        assert_eq!(target.x + guide.tip_x, 1920.0);
        assert_eq!(target.y + guide.tip_y, 0.0);
    }

    #[test]
    fn reported_geometry_selects_an_offset_monitor() {
        let first = screen();
        let second = Screen {
            left: -2560.0,
            top: -240.0,
            right: 0.0,
            bottom: 1200.0,
            scale: 1.5,
        };
        let reported = PetMonitor {
            left: -2560,
            top: -240,
            width: 2560,
            height: 1440,
        };
        let selected = reported_screen(reported, &[first, second]).unwrap();
        let guide = GuideMotion::default();
        let target = target_destination(0.5, 0.5, selected, &guide);
        assert_eq!(selected.left, -2560.0);
        assert_eq!(target.x + guide.tip_x * selected.scale, -1280.0);
        assert_eq!(target.y + guide.tip_y * selected.scale, 480.0);
    }

    #[test]
    fn flights_keep_their_endpoints_and_timing_bounds() {
        let start = Point { x: 100.0, y: 900.0 };
        let end = Point {
            x: 1700.0,
            y: 100.0,
        };
        let control = flight_control(start, end, &[screen()], 1.0);
        assert_eq!(quadratic(start, control, end, smootherstep(0.0)), start);
        assert_eq!(quadratic(start, control, end, smootherstep(1.0)), end);
        let outward = flight_duration(start, end, 1.0, FlightKind::Target, false);
        let returning = flight_duration(start, end, 1.0, FlightKind::Return, false);
        assert!((0.50..=1.10).contains(&outward));
        assert!((0.40..=0.75).contains(&returning));
        assert!(returning < outward);

        let reduced_outward = flight_duration(start, end, 1.0, FlightKind::Target, true);
        let reduced_returning = flight_duration(start, end, 1.0, FlightKind::Return, true);
        assert!((0.32..=0.55).contains(&reduced_outward));
        assert!((0.28..=0.48).contains(&reduced_returning));
        assert!(reduced_outward < outward);
        assert!(reduced_returning < returning);
    }
}
