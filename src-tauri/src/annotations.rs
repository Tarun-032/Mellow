//! One lazy, click-through webview per used monitor. Geometry is physical pixels.
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{Emitter, Manager, PhysicalPosition, PhysicalSize, WebviewUrl, WebviewWindowBuilder};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Monitor {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mark {
    kind: String,
    points: Vec<[f64; 2]>,
    text: Option<String>,
    color: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    presentation_id: String,
    frame_id: String,
    hwnd: isize,
    window: Option<[i32; 4]>,
    monitor: Monitor,
    lifetime_ms: u32,
    marks: Vec<Mark>,
    #[serde(default)]
    reveal_from: usize,
    #[serde(default)]
    caption: String,
}

impl Scene {
    fn validate(&self) -> Result<(), String> {
        if self.presentation_id.len() != 32
            || self.frame_id.len() != 32
            || self.hwnd == 0
            || self.window.is_none()
            || self.monitor.width == 0
            || self.monitor.height == 0
            || !(1..=60000).contains(&self.lifetime_ms)
            || !(1..=32).contains(&self.marks.len())
            || self.reveal_from > self.marks.len()
            || self.caption.chars().count() > 60
            || self.caption.chars().any(char::is_control)
        {
            return Err("invalid drawing scene".into());
        }
        for mark in &self.marks {
            let count = match mark.kind.as_str() {
                "rectangle" | "highlight" | "ellipse" | "line" | "arrow" => 2,
                "label" => 1,
                "quadratic" => 3,
                "cubic" => 4,
                "polygon" if (3..=16).contains(&mark.points.len()) => mark.points.len(),
                _ => return Err("unsupported drawing kind".into()),
            };
            if mark.points.len() != count
                || !["mint", "amber", "violet"].contains(&mark.color.as_str())
            {
                return Err("invalid drawing mark".into());
            }
            for [x, y] in &mark.points {
                if !x.is_finite()
                    || !y.is_finite()
                    || *x < self.monitor.left as f64
                    || *y < self.monitor.top as f64
                    || *x > self.monitor.left as f64 + self.monitor.width as f64
                    || *y > self.monitor.top as f64 + self.monitor.height as f64
                {
                    return Err("drawing leaves its monitor".into());
                }
            }
            if mark.kind == "label" {
                let text = mark.text.as_ref().ok_or("label has no text")?;
                if text.is_empty()
                    || text.chars().count() > 60
                    || text.chars().any(char::is_control)
                {
                    return Err("invalid label".into());
                }
            } else if mark.text.is_some() {
                return Err("unexpected drawing text".into());
            }
            if ["rectangle", "highlight", "ellipse"].contains(&mark.kind.as_str())
                && (mark.points[0][0] >= mark.points[1][0]
                    || mark.points[0][1] >= mark.points[1][1])
            {
                return Err("empty drawing box".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, serde::Serialize)]
pub struct Display {
    revision: u64,
    scene: Scene,
    scale: f64,
    pen_origin: [f64; 2],
}

struct Active {
    display: Display,
    label: String,
    expires: Instant,
    ready: bool,
    remaining_ms: Option<f64>,
    complete: bool,
    finished: bool,
}

impl Active {
    fn can_renew(&self, revision: u64, now: Instant) -> bool {
        self.display.revision == revision && self.ready && !self.finished && self.expires > now
    }
}

#[derive(Default)]
struct Inner {
    revision: u64,
    active: Option<Active>,
    capturing: bool,
}

#[derive(Clone, Default)]
pub struct State(Arc<Mutex<Inner>>, Arc<std::sync::atomic::AtomicBool>);

pub async fn on_main<T: Send + 'static>(
    app: tauri::AppHandle,
    work: impl FnOnce(tauri::AppHandle) -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let handle = app.clone();
    app.run_on_main_thread(move || {
        let _ = sender.send(work(handle));
    })
    .map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || receiver.recv().map_err(|e| e.to_string()))
        .await
        .map_err(|e| e.to_string())??
}

fn pet(window: &tauri::WebviewWindow) -> Result<(), String> {
    if window.label() != "pet" {
        return Err("only the pet owns drawings".into());
    }
    Ok(())
}

fn monitor(app: &tauri::AppHandle, wanted: &Monitor) -> Option<tauri::Monitor> {
    app.available_monitors().ok()?.into_iter().find(|m| {
        let p = m.position();
        let s = m.size();
        (p.x, p.y, s.width, s.height) == (wanted.left, wanted.top, wanted.width, wanted.height)
    })
}

#[cfg(windows)]
fn source_valid(scene: &Scene) -> bool {
    #[repr(C)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }
    #[link(name = "user32")]
    extern "system" {
        fn GetForegroundWindow() -> isize;
        fn GetWindowRect(hwnd: isize, rect: *mut Rect) -> i32;
        fn IsWindowVisible(hwnd: isize) -> i32;
        fn IsIconic(hwnd: isize) -> i32;
        fn GetAncestor(hwnd: isize, flags: u32) -> isize;
        fn GetWindowThreadProcessId(hwnd: isize, pid: *mut u32) -> u32;
    }
    let mut rect = Rect {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };
    // Read only a source HWND and its physical rectangle. No input is sent.
    unsafe {
        // Mellow's own windows are not a switch away from the source.
        let foreground = GetForegroundWindow();
        let mut foreground_pid = 0;
        GetWindowThreadProcessId(foreground, &mut foreground_pid);
        if (foreground != scene.hwnd && foreground_pid != std::process::id())
            || IsWindowVisible(scene.hwnd) == 0
            || IsIconic(scene.hwnd) != 0
            || GetWindowRect(scene.hwnd, &mut rect) == 0
        {
            return false;
        }
        // Match capture.window_rect: a same-process popup includes its owner.
        let root = GetAncestor(scene.hwnd, 3);
        let (mut pid, mut owner_pid) = (0, 0);
        GetWindowThreadProcessId(scene.hwnd, &mut pid);
        GetWindowThreadProcessId(root, &mut owner_pid);
        if root != 0 && root != scene.hwnd && pid == owner_pid {
            let mut owner = Rect {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            };
            if GetWindowRect(root, &mut owner) != 0 {
                rect.left = rect.left.min(owner.left);
                rect.top = rect.top.min(owner.top);
                rect.right = rect.right.max(owner.right);
                rect.bottom = rect.bottom.max(owner.bottom);
            }
        }
        scene.window
            == Some([
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
            ])
    }
}

#[cfg(not(windows))]
fn source_valid(_scene: &Scene) -> bool {
    false
}

fn invalid_reason(app: &tauri::AppHandle, active: &Active) -> Option<&'static str> {
    if active.expires <= Instant::now() {
        return Some("expired");
    }
    if !source_valid(&active.display.scene) {
        return Some("source_changed");
    }
    if !app
        .get_webview_window("pet")
        .is_some_and(|w| w.is_visible().unwrap_or(false))
    {
        return Some("pet_hidden");
    }
    if !monitor(app, &active.display.scene.monitor)
        .is_some_and(|m| (m.scale_factor() - active.display.scale).abs() < 0.001)
    {
        return Some("display_changed");
    }
    None
}

fn valid(app: &tauri::AppHandle, active: &Active) -> bool {
    invalid_reason(app, active).is_none()
}

fn receipt(app: &tauri::AppHandle, active: &Active, outcome: &str, reason: Option<&'static str>) {
    let mut payload = serde_json::json!({
        "revision": active.display.revision, "presentation_id": active.display.scene.presentation_id,
        "outcome": outcome, "scale": active.display.scale
    });
    if let Some(reason) = reason {
        payload["reason"] = serde_json::json!(reason);
    }
    if outcome == "ready" {
        if let Some(remaining_ms) = active.remaining_ms {
            payload["remaining_ms"] = serde_json::json!(remaining_ms);
        }
    }
    let _ = app.emit_to("pet", "drawing-receipt", payload);
}

fn discard(app: &tauri::AppHandle, inner: &mut Inner, keep_pen: bool, reason: &'static str) {
    if let Some(active) = inner.active.take() {
        if let Some(window) = app.get_webview_window(&active.label) {
            let _ = window.hide();
            let _ = window.emit("annotation-scene", Option::<Display>::None);
        }
        receipt(app, &active, "failed", Some(reason));
    }
    if !keep_pen { crate::cursor::return_pen(app); }
}

/// The monitor's hidden click-through overlay, built once and reused.
fn overlay_window(handle: &tauri::AppHandle, name: String) -> Result<tauri::WebviewWindow, String> {
    if let Some(existing) = handle.get_webview_window(&name) {
        return Ok(existing);
    }
    WebviewWindowBuilder::new(handle, name, WebviewUrl::App("index.html".into()))
        .title("Mellow annotations")
        .inner_size(1.0, 1.0)
        .resizable(false)
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .skip_taskbar(true)
        .shadow(false)
        .focusable(false)
        .focused(false)
        .visible(false)
        .build()
        .map_err(|e| e.to_string())
}

/// Build a monitor's overlay while a screen question is still being answered,
/// so the first drawing does not wait for WebView2 to start and load the page.
#[tauri::command]
pub async fn annotation_prepare(
    app: tauri::AppHandle,
    window: tauri::WebviewWindow,
    left: i32,
    top: i32,
) -> Result<(), String> {
    pet(&window)?;
    let label = format!("annotation-{left}-{top}");
    tauri::async_runtime::spawn_blocking(move || overlay_window(&app, label).map(|_| ()))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn annotation_present(
    app: tauri::AppHandle,
    window: tauri::WebviewWindow,
    state: tauri::State<'_, State>,
    revision: u64,
    scene: Scene,
) -> Result<bool, String> {
    pet(&window)?;
    scene.validate()?;
    let mon = monitor(&app, &scene.monitor).ok_or("drawing monitor is missing")?;
    let scale = mon.scale_factor();
    let expires = Instant::now() + Duration::from_millis(scene.lifetime_ms as u64);
    let label = format!("annotation-{}-{}", scene.monitor.left, scene.monitor.top);
    let owned = state.inner().clone();
    let reserve = owned.clone();
    if !on_main(app.clone(), move |app| {
        let mut inner = reserve.0.lock().map_err(|_| "drawing state poisoned")?;
        if revision <= inner.revision {
            return Ok(false);
        }
        inner.revision = revision;
        discard(&app, &mut inner, true, "replaced");
        Ok(true)
    })
    .await?
    {
        return Ok(false);
    }
    // Building a WebView2 window in a synchronous event handler can deadlock.
    let handle = app.clone();
    let name = label.clone();
    let native = tauri::async_runtime::spawn_blocking(move || overlay_window(&handle, name))
        .await
        .map_err(|e| e.to_string())??;
    on_main(app, move |app| {
        let mut inner = owned.0.lock().map_err(|_| "drawing state poisoned")?;
        if revision != inner.revision {
            return Ok(false);
        }
        native
            .set_ignore_cursor_events(true)
            .map_err(|e| e.to_string())?;
        native.set_focusable(false).map_err(|e| e.to_string())?;
        native
            .set_position(PhysicalPosition::new(scene.monitor.left, scene.monitor.top))
            .map_err(|e| e.to_string())?;
        native
            .set_size(PhysicalSize::new(scene.monitor.width, scene.monitor.height))
            .map_err(|e| e.to_string())?;
        let display = Display {
            revision,
            scene,
            scale,
            pen_origin: crate::cursor::lend_pen(&app)?,
        };
        inner.active = Some(Active {
            display: display.clone(),
            label,
            expires,
            ready: false,
            remaining_ms: None,
            complete: false,
            finished: false,
        });
        native
            .emit("annotation-scene", Some(display))
            .map_err(|e| e.to_string())?;
        Ok(true)
    })
    .await
}

#[tauri::command]
pub async fn annotation_clear(
    app: tauri::AppHandle,
    window: tauri::WebviewWindow,
    state: tauri::State<'_, State>,
    revision: u64,
    keep_pen: Option<bool>,
) -> Result<(), String> {
    pet(&window)?;
    let owned = state.inner().clone();
    on_main(app, move |app| {
        let mut inner = owned.0.lock().map_err(|_| "drawing state poisoned")?;
        if revision > inner.revision {
            inner.revision = revision;
            discard(&app, &mut inner, keep_pen.unwrap_or(false), "cleared");
        }
        Ok(())
    })
    .await
}

#[tauri::command]
pub fn annotation_snapshot(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, State>,
) -> Option<Display> {
    state
        .0
        .lock()
        .ok()?
        .active
        .as_ref()
        .filter(|a| a.label == window.label())
        .map(|a| a.display.clone())
}

#[tauri::command]
pub async fn annotation_occlusion(
    window: tauri::WebviewWindow,
    rects: Vec<[f64; 4]>,
) -> Result<Vec<[f64; 4]>, String> {
    pet(&window)?;
    if rects.len() > 8
        || rects
            .iter()
            .any(|r| r.iter().any(|v| !v.is_finite()) || r[2] <= 0.0 || r[3] <= 0.0)
    {
        return Err("invalid pet occlusion bounds".into());
    }
    let origin = window.inner_position().map_err(|e| e.to_string())?;
    let scale = window.scale_factor().map_err(|e| e.to_string())?;
    Ok(rects
        .into_iter()
        .map(|[x, y, w, h]| {
            [
                origin.x as f64 + x * scale,
                origin.y as f64 + y * scale,
                w * scale,
                h * scale,
            ]
        })
        .collect())
}

#[tauri::command]
pub async fn annotation_painted(
    app: tauri::AppHandle,
    window: tauri::WebviewWindow,
    state: tauri::State<'_, State>,
    revision: u64,
    painted: bool,
    remaining_ms: Option<f64>,
) -> Result<bool, String> {
    let owned = state.inner().clone();
    // A clean capture hides drawings for a moment. Wait it out (off the main
    // thread) instead of declining: the renderer sends each receipt once, so a
    // decline here used to strand the scene until its arrival timeout.
    let watched = owned.clone();
    let _ = tauri::async_runtime::spawn_blocking(move || {
        for _ in 0..60 {
            if !watched.0.lock().map(|inner| inner.capturing).unwrap_or(false) {
                break;
            }
            std::thread::sleep(Duration::from_millis(16));
        }
    })
    .await;
    on_main(app, move |app| {
        let mut inner = owned.0.lock().map_err(|_| "drawing state poisoned")?;
        if inner.capturing {
            return Ok(false);
        }
        let Some(active) = inner
            .active
            .as_mut()
            .filter(|a| a.label == window.label() && a.display.revision == revision)
        else {
            return Ok(false);
        };
        if let Some(reason) = invalid_reason(&app, active) {
            discard(&app, &mut inner, false, reason);
            return Ok(false);
        }
        crate::show_annotation_below_pet(&app, &window)?;
        if painted && !active.ready {
            active.ready = true;
            // This timing is supplied by our renderer, never by the model.
            active.remaining_ms = remaining_ms.filter(|ms| ms.is_finite() && (0.0..=55000.0).contains(ms));
            receipt(&app, active, "ready", None);
        }
        Ok(true)
    })
    .await
}

#[tauri::command]
pub async fn annotation_complete(
    app: tauri::AppHandle,
    window: tauri::WebviewWindow,
    state: tauri::State<'_, State>,
    revision: u64,
) -> Result<(), String> {
    let owned = state.inner().clone();
    on_main(app, move |app| {
        let mut inner = owned.0.lock().map_err(|_| "drawing state poisoned")?;
        if let Some(active) = inner.active.as_mut().filter(|a| {
            a.label == window.label() && a.display.revision == revision && a.ready && !a.complete
        }) {
            active.complete = true;
            receipt(&app, active, "complete", None);
        }
        Ok(())
    })
    .await
}

/// Pose samples are for handback only. Ink and bone are painted together in JS;
/// no native-window movement or mouse input is performed for each stroke frame.
#[tauri::command]
pub async fn annotation_pen_position(
    app: tauri::AppHandle, window: tauri::WebviewWindow,
    state: tauri::State<'_, State>, revision: u64, tip: [f64; 2], rects: Vec<[f64; 4]>,
) -> Result<(), String> {
    let owned = state.inner().clone();
    on_main(app, move |app| {
        let inner = owned.0.lock().map_err(|_| "drawing state poisoned")?;
        if let Some(active) = inner.active.as_ref().filter(|a| a.label == window.label()
            && a.display.revision == revision && valid(&app, a)) {
            let m = &active.display.scene.monitor;
            let scale = active.display.scale;
            if rects.len() > 3 || rects.iter().any(|r| r.iter().any(|v| !v.is_finite())
                || r[2] <= 0.0 || r[3] <= 0.0 || r[2] > 256.0 * scale || r[3] > 64.0 * scale) {
                return Err("invalid pen occlusion".into());
            }
            if tip.iter().all(|n| n.is_finite()) && tip[0] >= m.left as f64
                && tip[1] >= m.top as f64 && tip[0] <= m.left as f64 + m.width as f64
                && tip[1] <= m.top as f64 + m.height as f64 {
                crate::cursor::pen_at(&app, tip, active.display.scale);
                let _ = app.emit_to("pet", "drawing-receipt", serde_json::json!({
                    "revision": revision, "presentation_id": active.display.scene.presentation_id,
                    "outcome": "position", "occlusions": rects
                }));
            }
        }
        Ok(())
    }).await
}

#[tauri::command]
pub async fn annotation_finish(
    app: tauri::AppHandle, window: tauri::WebviewWindow,
    state: tauri::State<'_, State>, revision: u64,
) -> Result<(), String> {
    pet(&window)?;
    let owned = state.inner().clone();
    on_main(app, move |app| {
        let mut inner = owned.0.lock().map_err(|_| "drawing state poisoned")?;
        // A late finish from an older socket/turn cannot release a newer pen.
        if let Some(active) = inner.active.as_mut().filter(|a| a.display.revision == revision) {
            // Active narration has a larger bound; completed ink gets a short
            // reading window measured from finish, rather than capture time.
            active.expires = Instant::now() + Duration::from_secs(12);
            active.finished = true;
            if let Some(view) = app.get_webview_window(&active.label) {
                let _ = view.emit("annotation-pen-finished", revision);
            }
            crate::cursor::return_pen(&app);
        } else if inner.active.is_none() && revision == inner.revision {
            crate::cursor::return_pen(&app);
        }
        Ok(())
    }).await
}

/// Narration may continue after the strokes finish. Extend only the still
/// owned active scene; no renderer update or input movement happens here.
#[tauri::command]
pub async fn annotation_renew(
    app: tauri::AppHandle, window: tauri::WebviewWindow,
    state: tauri::State<'_, State>, revision: u64,
) -> Result<bool, String> {
    pet(&window)?;
    let owned = state.inner().clone();
    on_main(app, move |app| {
        let mut inner = owned.0.lock().map_err(|_| "drawing state poisoned")?;
        let capturing = inner.capturing;
        let Some(active) = inner.active.as_mut().filter(|a| a.display.revision == revision) else {
            return Ok(false);
        };
        let now = Instant::now();
        if !active.can_renew(revision, now) {
            return Ok(false);
        }
        if let Some(reason) = (!capturing).then(|| invalid_reason(&app, active)).flatten() {
            discard(&app, &mut inner, false, reason);
            return Ok(false);
        }
        if !crate::cursor::renew_pen(&app, now)? {
            discard(&app, &mut inner, false, "expired");
            return Ok(false);
        }
        active.expires = now + Duration::from_secs(60);
        Ok(true)
    }).await
}

impl State {
    pub fn capture(&self, app: &tauri::AppHandle, capturing: bool) {
        let mut inner = self.0.lock().unwrap_or_else(|p| p.into_inner());
        inner.capturing = capturing;
        if let Some(active) = inner.active.as_ref() {
            // capture_prepare excludes supported windows from capture while
            // keeping ink visible. Only its unsupported-affinity fallback hides
            // a window; its current validated scene owns restoration below.
            if capturing {
                return;
            }
            if let Some(reason) = invalid_reason(app, active) {
                discard(app, &mut inner, false, reason);
            } else if active.ready {
                if let Some(window) = app.get_webview_window(&active.label) {
                    if let Err(error) = crate::show_annotation_below_pet(app, &window) {
                        eprintln!("[mellow] could not restore annotation layer: {error}");
                        discard(app, &mut inner, false, "native_failure");
                    }
                }
            }
        }
    }
}

pub fn spawn(app: tauri::AppHandle, state: State) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_millis(150));
        if state
            .0
            .lock()
            .map_or(true, |inner| inner.active.is_none() || inner.capturing)
        {
            continue;
        }
        if state.1.swap(true, std::sync::atomic::Ordering::Relaxed) {
            continue;
        }
        let state = state.clone();
        let handle = app.clone();
        let pending = state.1.clone();
        if app
            .run_on_main_thread(move || {
                let mut inner = state.0.lock().unwrap_or_else(|p| p.into_inner());
                if !inner.capturing {
                    if let Some(reason) = inner
                        .active
                        .as_ref()
                        .and_then(|a| invalid_reason(&handle, a))
                    {
                        eprintln!("[mellow] drawing retired: {reason}");
                        discard(&handle, &mut inner, false, reason);
                    }
                }
                state.1.store(false, std::sync::atomic::Ordering::Relaxed);
            })
            .is_err()
        {
            pending.store(false, std::sync::atomic::Ordering::Relaxed);
            break;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene() -> Scene {
        serde_json::from_value(serde_json::json!({
            "presentation_id": "a".repeat(32), "frame_id": "b".repeat(32),
            "hwnd": 1, "window": [-1800, -100, 1000, 800],
            "monitor": {"left": -1920, "top": -200, "width": 1920, "height": 1080},
            "lifetime_ms": 12000,
            "marks": [{"kind":"rectangle", "points":[[-1728,-92],[-1536,16]], "text":null, "color":"mint"}]
        })).unwrap()
    }
    #[test]
    fn renewal_keeps_only_the_owned_unfinished_live_scene() {
        let now = Instant::now();
        let mut active = Active {
            display: Display { revision: 7, scene: scene(), scale: 1.0, pen_origin: [0.0, 0.0] },
            label: "annotations".into(), expires: now + Duration::from_secs(60),
            ready: true, remaining_ms: None, complete: true, finished: false,
        };
        assert!(active.can_renew(7, now));
        assert!(!active.can_renew(6, now));
        assert!(!active.can_renew(7, now + Duration::from_secs(60)));
        active.finished = true;
        assert!(!active.can_renew(7, now));
        active.finished = false;
        active.ready = false;
        assert!(!active.can_renew(7, now));
    }
    #[test]
    fn finite_bounded_typed_geometry_only() {
        let original = scene();
        assert!(original.validate().is_ok());
        for bad in [f64::NAN, f64::INFINITY, -1921.0, 1.0] {
            let mut value = original.clone();
            value.marks[0].points[0][0] = bad;
            assert!(value.validate().is_err());
        }
        let mut value = original.clone();
        value.marks[0].kind = "svg".into();
        assert!(value.validate().is_err());
        let mut value = original.clone();
        value.marks = vec![value.marks[0].clone(); 33];
        assert!(value.validate().is_err());
        let mut value = original.clone();
        value.lifetime_ms = 60000;
        assert!(value.validate().is_ok());
        value.lifetime_ms = 60001;
        assert!(value.validate().is_err());
        let mut value = original.clone();
        value.reveal_from = 2;
        assert!(value.validate().is_err());
        let mut value = original.clone();
        value.marks[0].text = Some("<script>".into());
        assert!(value.validate().is_err());
        value.marks[0].kind = "label".into();
        value.marks[0].points.truncate(1);
        assert!(value.validate().is_ok());
    }
    #[test]
    fn executable_and_unknown_fields_are_rejected() {
        let mut value = serde_json::to_value(scene()).unwrap();
        value["marks"][0]["onclick"] = serde_json::json!("alert(1)");
        assert!(serde_json::from_value::<Scene>(value).is_err());
        let mut value = serde_json::to_value(scene()).unwrap();
        value["html"] = serde_json::json!("<script>");
        assert!(serde_json::from_value::<Scene>(value).is_err());
    }
}
