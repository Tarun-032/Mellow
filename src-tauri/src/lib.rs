#[cfg(desktop)]
mod cursor;
#[cfg(desktop)]
mod sidecar;

#[cfg(desktop)]
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Emitter, Listener, Manager, PhysicalPosition, PhysicalSize, WebviewUrl,
    WebviewWindowBuilder,
};
#[cfg(desktop)]
use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut, ShortcutState};

/// Push-to-talk key state: true on press, false on release.
#[cfg(desktop)]
const PTT: &str = "ptt";

/// Frontend -> Rust: pet right-click. Payload `{ speak, quiet }` labels menu items.
#[cfg(desktop)]
const PET_MENU: &str = "pet-menu";

/// Rust -> frontend: mute/unmute; shell owns the sidecar socket.
#[cfg(desktop)]
const TOGGLE_SPEAK: &str = "toggle-speak";

/// Rust -> frontend: new conversation (clears bubble + model context via shell).
#[cfg(desktop)]
const NEW_CHAT: &str = "new-chat";

/// Rust -> frontend: open a pet panel ("pomodoro" or "reminders").
#[cfg(desktop)]
const OPEN_PANEL: &str = "open-panel";

/// Rust -> frontend: tuck/untuck the pet; frontend owns placement.
#[cfg(desktop)]
const PET_QUIET: &str = "pet-quiet";

/// Frontend -> Rust: lend focus while a panel needs typing (`{ focus }`).
#[cfg(desktop)]
const PET_FOCUS: &str = "pet-focus";

/// Frontend -> Rust: hide Mellow windows from the next screen capture (`{ hidden }`).
#[cfg(desktop)]
const PET_CAPTURE: &str = "pet-capture";

/// Rust -> frontend: onboarding done; discard nap state from the hidden wait.
#[cfg(desktop)]
const PET_WAKE: &str = "pet-wake";

#[cfg(desktop)]
const MAIN_TRAY: &str = "main-tray";

#[derive(Clone, Copy, serde::Deserialize, serde::Serialize)]
struct PetMonitor {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

/// Monitor containing the system cursor (snapshot for screen turns).
#[cfg(desktop)]
#[tauri::command]
fn cursor_monitor(app: tauri::AppHandle) -> Result<PetMonitor, String> {
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let monitor = app
        .available_monitors()
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|monitor| {
            let position = monitor.position();
            let size = monitor.size();
            cursor.x >= f64::from(position.x)
                && cursor.x < f64::from(position.x) + f64::from(size.width)
                && cursor.y >= f64::from(position.y)
                && cursor.y < f64::from(position.y) + f64::from(size.height)
        })
        .or_else(|| app.primary_monitor().ok().flatten())
        .ok_or_else(|| "no monitor is available".to_string())?;
    let position = monitor.position();
    let size = monitor.size();
    Ok(PetMonitor {
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
    })
}

/// Monitor Mellow currently lives on (capture follows the pet).
#[cfg(desktop)]
#[tauri::command]
fn pet_monitor(app: tauri::AppHandle) -> Result<PetMonitor, String> {
    let window = app
        .get_webview_window("pet")
        .ok_or_else(|| "pet window is missing".to_string())?;
    let monitor = window
        .current_monitor()
        .map_err(|error| error.to_string())?
        .or_else(|| app.primary_monitor().ok().flatten())
        .ok_or_else(|| "no monitor is available".to_string())?;
    let position = monitor.position();
    let size = monitor.size();
    Ok(PetMonitor {
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
    })
}

/// Move Mellow to the cursor's monitor (only during an active user drag).
#[cfg(desktop)]
#[tauri::command]
fn move_pet_to_cursor_monitor(app: tauri::AppHandle) -> Result<Option<PetMonitor>, String> {
    let window = app
        .get_webview_window("pet")
        .ok_or_else(|| "pet window is missing".to_string())?;
    let cursor = app.cursor_position().map_err(|error| error.to_string())?;
    let current = window
        .current_monitor()
        .map_err(|error| error.to_string())?;
    let target = app
        .available_monitors()
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|monitor| {
            let p = monitor.position();
            let s = monitor.size();
            cursor.x >= f64::from(p.x)
                && cursor.x < f64::from(p.x) + f64::from(s.width)
                && cursor.y >= f64::from(p.y)
                && cursor.y < f64::from(p.y) + f64::from(s.height)
        });
    let Some(target) = target else {
        return Ok(None);
    };
    if current.as_ref().is_some_and(|monitor| {
        monitor.position().x == target.position().x
            && monitor.position().y == target.position().y
    }) {
        return Ok(None);
    }
    let position = target.position();
    let size = target.size();
    window
        .set_position(PhysicalPosition::new(position.x, position.y))
        .map_err(|error| error.to_string())?;
    window
        .set_size(PhysicalSize::new(size.width, size.height))
        .map_err(|error| error.to_string())?;
    Ok(Some(PetMonitor {
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
    }))
}

/// Hide/unhide Mellow windows from screen capture (WDA_EXCLUDEFROMCAPTURE).
#[cfg(all(desktop, windows))]
fn set_capture_hidden(app: &tauri::AppHandle, hidden: bool) {
    // WDA_EXCLUDEFROMCAPTURE / WDA_NONE, from winuser.h.
    const EXCLUDE: u32 = 0x0000_0011;
    const NONE: u32 = 0x0000_0000;
    #[link(name = "user32")]
    extern "system" {
        fn SetWindowDisplayAffinity(hwnd: isize, affinity: u32) -> i32;
    }
    for (_, window) in app.webview_windows() {
        if let Ok(hwnd) = window.hwnd() {
            // Safe: an HWND we own, and the call only reads a flag on it.
            unsafe { SetWindowDisplayAffinity(hwnd.0 as isize, if hidden { EXCLUDE } else { NONE }) };
        }
    }
}

#[cfg(all(desktop, not(windows)))]
fn set_capture_hidden(_app: &tauri::AppHandle, _hidden: bool) {}

#[cfg(desktop)]
fn open_settings(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("settings") {
        let _ = window.show();
        let _ = window.set_focus();
        return;
    }

    if let Err(error) = WebviewWindowBuilder::new(
        app,
        "settings",
        WebviewUrl::App("index.html".into()),
    )
    .title("Mellow settings")
    .inner_size(960.0, 720.0)
    .min_inner_size(720.0, 600.0)
    .resizable(true)
    .center()
    .build()
    {
        eprintln!("could not open settings: {error}");
    }
}

/// First-run setup window (stays until the final acknowledgement).
#[cfg(desktop)]
fn open_welcome(app: &tauri::AppHandle) {
    if let Some(pet) = app.get_webview_window("pet") {
        let _ = pet.hide();
    }
    if let Some(window) = app.get_webview_window("welcome") {
        let _ = window.show();
        let _ = window.set_focus();
        return;
    }

    if let Err(error) = WebviewWindowBuilder::new(
        app,
        "welcome",
        WebviewUrl::App("index.html".into()),
    )
    .title("Set up Mellow")
    .inner_size(980.0, 700.0)
    .resizable(false)
    .center()
    .build()
    {
        eprintln!("could not open the setup wizard: {error}");
    }
}

#[cfg(desktop)]
fn mellow_data_dir() -> Option<std::path::PathBuf> {
    std::env::var("APPDATA")
        .ok()
        .map(|base| std::path::Path::new(&base).join("Mellow"))
}

#[cfg(desktop)]
fn config_saved_in(directory: &std::path::Path) -> bool {
    directory.join("config.json").exists()
}

#[cfg(desktop)]
fn onboarding_complete_in(directory: &std::path::Path) -> bool {
    directory.join("onboarding-complete").exists()
}

#[cfg(desktop)]
fn config_saved() -> bool {
    mellow_data_dir().is_some_and(|directory| config_saved_in(&directory))
}

/// True only after the final onboarding acknowledgement (not mere config save).
#[cfg(desktop)]
fn onboarded() -> bool {
    mellow_data_dir().is_some_and(|directory| onboarding_complete_in(&directory))
}

#[cfg(desktop)]
fn show_pet(app: &tauri::AppHandle) {
    if !onboarded() {
        open_welcome(app);
        return;
    }
    if let Some(win) = app.get_webview_window("pet") {
        // Reassert topmost/non-focusable/click-through after Windows drops them.
        let _ = win.set_always_on_top(true);
        let _ = win.set_focusable(false);
        let _ = win.set_ignore_cursor_events(true);
        let _ = win.show();
    }
}

#[cfg(desktop)]
fn build_tray_menu(
    app: &tauri::AppHandle,
    include_show: bool,
) -> tauri::Result<Menu<tauri::Wry>> {
    let settings = MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Mellow", true, None::<&str>)?;
    if include_show {
        let show = MenuItem::with_id(app, "show", "Show pet", true, None::<&str>)?;
        Menu::with_items(app, &[&show, &settings, &quit])
    } else {
        Menu::with_items(app, &[&settings, &quit])
    }
}

#[cfg(desktop)]
fn refresh_tray_menu(app: &tauri::AppHandle) -> tauri::Result<()> {
    if let Some(tray) = app.tray_by_id(MAIN_TRAY) {
        tray.set_menu(Some(build_tray_menu(app, onboarded())?))?;
    }
    Ok(())
}

/// Finish onboarding: show pet first, then close the wizard.
#[cfg(desktop)]
#[tauri::command]
fn complete_onboarding(app: tauri::AppHandle) -> Result<(), String> {
    if !config_saved() {
        return Err("Mellow could not find the saved setup. Try again.".to_string());
    }

    let pet = app.get_webview_window("pet").ok_or_else(|| {
        "Mellow's pet window is not available. Restart Mellow and try again.".to_string()
    })?;
    let data_dir = mellow_data_dir()
        .ok_or_else(|| "Mellow could not locate your Windows app-data folder.".to_string())?;
    std::fs::create_dir_all(&data_dir).map_err(|error| error.to_string())?;
    std::fs::write(data_dir.join("onboarding-complete"), b"1\n")
        .map_err(|error| error.to_string())?;
    if let Err(error) = refresh_tray_menu(&app) {
        eprintln!("could not unlock the tray's Show pet item: {error}");
    }
    pet.set_always_on_top(true)
        .map_err(|error| error.to_string())?;
    pet.set_focusable(false).map_err(|error| error.to_string())?;
    pet.set_ignore_cursor_events(true)
        .map_err(|error| error.to_string())?;
    pet.show().map_err(|error| error.to_string())?;
    app.emit(PET_WAKE, ()).map_err(|error| error.to_string())?;

    if let Some(welcome) = app.get_webview_window("welcome") {
        welcome.close().map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Native pet context menu (DOM would fall through click-through pixels).
#[cfg(desktop)]
fn popup_pet_menu(handle: &tauri::AppHandle, speak: bool, quiet: bool) {
    let app = handle.clone();
    let _ = handle.run_on_main_thread(move || {
        let Some(win) = app.get_webview_window("pet") else {
            return;
        };
        // Prefixed: tray menu already owns "settings".
        let build = || -> tauri::Result<Menu<tauri::Wry>> {
            let settings =
                MenuItem::with_id(&app, "pet_settings", "Settings", true, None::<&str>)?;
            let mute = MenuItem::with_id(
                &app,
                "pet_mute",
                if speak { "Mute voice" } else { "Unmute voice" },
                true,
                None::<&str>,
            )?;
            let pomodoro =
                MenuItem::with_id(&app, "pet_pomodoro", "Pomodoro…", true, None::<&str>)?;
            let reminders =
                MenuItem::with_id(&app, "pet_reminders", "Reminders…", true, None::<&str>)?;
            // Conversation action, grouped with the panels.
            let new_chat =
                MenuItem::with_id(&app, "pet_new_chat", "New conversation", true, None::<&str>)?;
            // Quiet = tuck at edge; Hide = remove the window.
            let quiet_item = MenuItem::with_id(
                &app,
                "pet_quiet",
                if quiet { "Come back" } else { "Stay quiet" },
                true,
                None::<&str>,
            )?;
            let hide = MenuItem::with_id(&app, "pet_hide", "Hide pet", true, None::<&str>)?;
            let quit = MenuItem::with_id(&app, "pet_quit", "Quit Mellow", true, None::<&str>)?;
            Menu::with_items(
                &app,
                &[
                    &pomodoro,
                    &reminders,
                    &new_chat,
                    &settings,
                    &mute,
                    &quiet_item,
                    &hide,
                    &quit,
                ],
            )
        };
        match build() {
            // Lend focus for the modal popup, then take it back.
            Ok(menu) => {
                let _ = win.set_focusable(true);
                let _ = win.set_focus();
                let _ = win.popup_menu(&menu);
                let _ = win.set_focusable(false);
            }
            Err(error) => eprintln!("could not build the pet menu: {error}"),
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[allow(unused_mut)]
    let mut builder = tauri::Builder::default().plugin(tauri_plugin_opener::init());

    #[cfg(desktop)]
    let guide_state = cursor::GuideState::default();
    #[cfg(desktop)]
    let guide_motion = guide_state.clone();

    // Sidecar child; killed on exit so a stale process cannot hold the port.
    let mellowd: std::sync::Arc<std::sync::Mutex<Option<std::process::Child>>> =
        Default::default();
    let started = mellowd.clone();

    // Register PTT here: JS StrictMode races left the hotkey unbound.
    #[cfg(desktop)]
    {
        // ponytail: hardcoded; move to config later.
        let ptt = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
        builder = builder.plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_shortcut(ptt)
                .expect("ctrl+shift+space is not a valid shortcut")
                .with_handler(move |app, shortcut, event| {
                    if shortcut == &ptt {
                        let down = event.state() == ShortcutState::Pressed;
                        let _ = app.emit(PTT, down);
                    }
                })
                .build(),
        );
    }

    #[cfg(desktop)]
    let builder = builder.manage(guide_state);

    builder
        .invoke_handler(tauri::generate_handler![
            pet_monitor,
            cursor_monitor,
            move_pet_to_cursor_monitor,
            complete_onboarding,
            cursor::guide_ready,
            cursor::guide_set_target,
            cursor::guide_clear,
            cursor::guide_set_dialogue,
            cursor::guide_set_quiet,
            cursor::guide_set_reduced_motion
        ])
        .setup(move |_app| {
            // Size the screen-sized overlay to the primary monitor at runtime.
            #[cfg(desktop)]
            {
                let win = _app
                    .get_webview_window("pet")
                    .expect("no window labelled `pet` — check tauri.conf.json");

                if let Ok(Some(mon)) = _app.primary_monitor() {
                    let p = mon.position();
                    win.set_position(PhysicalPosition::new(p.x, p.y))?;
                    win.set_size(*mon.size())?;
                }

                // Click-through until the frontend turns it off on the dog.
                win.set_ignore_cursor_events(true)?;
                win.set_focusable(false)?;
                win.set_always_on_top(true)?;

                // Tiny guide window: follows cursor without moving the pet.
                let guide = WebviewWindowBuilder::new(
                    _app,
                    "guide",
                    WebviewUrl::App("index.html".into()),
                )
                .title("Mellow guide")
                .inner_size(36.0, 36.0)
                .resizable(false)
                .decorations(false)
                .transparent(true)
                .always_on_top(true)
                .skip_taskbar(true)
                .shadow(false)
                .focusable(false)
                .visible(false)
                .build()?;
                guide.set_ignore_cursor_events(true)?;
                guide.set_focusable(false)?;

                // Separate bubble window so explanations can sit on other displays.
                let guide_bubble = WebviewWindowBuilder::new(
                    _app,
                    "guide-bubble",
                    WebviewUrl::App("index.html".into()),
                )
                .title("Mellow guide explanation")
                .inner_size(420.0, 220.0)
                .resizable(false)
                .decorations(false)
                .transparent(true)
                .always_on_top(true)
                .skip_taskbar(true)
                .shadow(false)
                .focusable(false)
                .visible(false)
                .build()?;
                guide_bubble.set_ignore_cursor_events(true)?;
                guide_bubble.set_focusable(false)?;

                cursor::spawn(_app.handle().clone(), guide_motion.clone());

                // Start sidecar before the frontend connects.
                *started.lock().unwrap() = sidecar::spawn(_app.handle())
                    .map_err(std::io::Error::other)?;

                // Wizard until onboarded; pet window is already sized/ready.
                if onboarded() {
                    show_pet(_app.handle());
                } else {
                    open_welcome(_app.handle());
                }

                let handle = _app.handle().clone();
                _app.listen(PET_MENU, move |event| {
                    // Malformed payload: assume voice is on.
                    let payload =
                        serde_json::from_str::<serde_json::Value>(event.payload()).ok();
                    let flag = |key: &str, fallback: bool| {
                        payload
                            .as_ref()
                            .and_then(|v| v.get(key).and_then(|f| f.as_bool()))
                            .unwrap_or(fallback)
                    };
                    // Malformed: assume not quiet.
                    popup_pet_menu(&handle, flag("speak", true), flag("quiet", false));
                });

                let capture_handle = _app.handle().clone();
                _app.listen(PET_CAPTURE, move |event| {
                    // Malformed: default to visible in captures.
                    let hidden = serde_json::from_str::<serde_json::Value>(event.payload())
                        .ok()
                        .and_then(|v| v.get("hidden").and_then(|h| h.as_bool()))
                        .unwrap_or(false);
                    let app = capture_handle.clone();
                    let _ = capture_handle.run_on_main_thread(move || {
                        set_capture_hidden(&app, hidden);
                    });
                });

                let focus_handle = _app.handle().clone();
                _app.listen(PET_FOCUS, move |event| {
                    // Malformed: default to non-focusable.
                    let wants = serde_json::from_str::<serde_json::Value>(event.payload())
                        .ok()
                        .and_then(|v| v.get("focus").and_then(|f| f.as_bool()))
                        .unwrap_or(false);
                    let app = focus_handle.clone();
                    let _ = focus_handle.run_on_main_thread(move || {
                        if let Some(win) = app.get_webview_window("pet") {
                            let _ = win.set_focusable(wants);
                            if wants {
                                let _ = win.set_focus();
                            }
                        }
                    });
                });

                // Pet-menu ids only; tray has its own handler below.
                _app.on_menu_event(|app, event| match event.id().as_ref() {
                    "pet_settings" => open_settings(app),
                    "pet_pomodoro" => {
                        let _ = app.emit(OPEN_PANEL, "pomodoro");
                    }
                    "pet_reminders" => {
                        let _ = app.emit(OPEN_PANEL, "reminders");
                    }
                    "pet_new_chat" => {
                        let _ = app.emit(NEW_CHAT, ());
                    }
                    "pet_mute" => {
                        let _ = app.emit(TOGGLE_SPEAK, ());
                    }
                    "pet_quiet" => {
                        let _ = app.emit(PET_QUIET, ());
                    }
                    "pet_hide" => {
                        if let Some(win) = app.get_webview_window("pet") {
                            let _ = win.hide();
                        }
                    }
                    "pet_quit" => app.exit(0),
                    _ => {}
                });

                // Hide "Show pet" until onboarding finishes.
                let menu = build_tray_menu(_app.handle(), onboarded())?;
                TrayIconBuilder::with_id(MAIN_TRAY)
                    .icon(
                        _app
                            .default_window_icon()
                            .expect("Mellow needs a bundled tray icon")
                            .clone(),
                    )
                    .tooltip("Mellow")
                    .menu(&menu)
                    .on_menu_event(|app, event| match event.id().as_ref() {
                        "show" => show_pet(app),
                        "settings" => open_settings(app),
                        "quit" => app.exit(0),
                        _ => {}
                    })
                    .build(_app)?;
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |_app, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(mut child) = mellowd.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}

#[cfg(all(test, desktop))]
mod onboarding_tests {
    use super::{config_saved_in, onboarding_complete_in};

    #[test]
    fn saved_config_does_not_unlock_pet_before_final_acknowledgement() {
        let directory = std::env::temp_dir().join(format!(
            "mellow-onboarding-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("config.json"), b"{}").unwrap();

        assert!(config_saved_in(&directory));
        assert!(!onboarding_complete_in(&directory));

        std::fs::write(directory.join("onboarding-complete"), b"1\n").unwrap();
        assert!(onboarding_complete_in(&directory));

        std::fs::remove_dir_all(directory).unwrap();
    }
}
