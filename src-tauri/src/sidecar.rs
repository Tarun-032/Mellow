//! Own and verify the Python service that provides Mellow's AI runtime.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, Instant};
use tauri::Manager;

/// Must match mellowd/main.py's HOST and PORT.
const ADDR: &str = "127.0.0.1:8765";
const SERVICE: &str = "mellowd";
const PROTOCOL: u64 = 1;
const READY_TIMEOUT: Duration = Duration::from_secs(30);

fn port_taken(addr: SocketAddr) -> bool {
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn validate_health(body: &str) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|error| format!("invalid health JSON: {error}"))?;
    let service = value.get("service").and_then(|item| item.as_str());
    let protocol = value.get("protocol").and_then(|item| item.as_u64());
    let version = value.get("version").and_then(|item| item.as_str());
    let ok = value.get("ok").and_then(|item| item.as_bool());
    if ok == Some(true)
        && service == Some(SERVICE)
        && protocol == Some(PROTOCOL)
        && version == Some(env!("CARGO_PKG_VERSION"))
    {
        return Ok(());
    }
    Err(format!(
        "expected {SERVICE} protocol {PROTOCOL} version {}, got service={service:?} protocol={protocol:?} version={version:?}",
        env!("CARGO_PKG_VERSION")
    ))
}

/// Prove that the listener is this build's sidecar, not merely an open port.
fn health(addr: SocketAddr) -> Result<(), String> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(500))
        .map_err(|error| format!("not listening: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("could not set health timeout: {error}"))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("could not set health timeout: {error}"))?;
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .map_err(|error| format!("health request failed: {error}"))?;

    let mut response = Vec::with_capacity(512);
    stream
        .take(64 * 1024)
        .read_to_end(&mut response)
        .map_err(|error| format!("health response failed: {error}"))?;
    let response = String::from_utf8(response)
        .map_err(|error| format!("health response was not UTF-8: {error}"))?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| "health response had no HTTP body".to_string())?;
    let status = headers.lines().next().unwrap_or_default();
    if !status.contains(" 200 ") {
        return Err(format!("health returned {status}"));
    }
    validate_health(body)
}

/// Build the development or installed release command.
fn command(app: &tauri::AppHandle) -> Result<Command, String> {
    if cfg!(debug_assertions) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let python = root.join(".venv/Scripts/python.exe");
        if !python.exists() {
            return Err(format!(
                "no development interpreter at {}",
                python.display()
            ));
        }
        let mut cmd = Command::new(python);
        cmd.args(["-m", "mellowd"]).current_dir(root);
        Ok(cmd)
    } else {
        let resources = app
            .path()
            .resource_dir()
            .map_err(|error| format!("could not resolve installed resources: {error}"))?;
        let exe = resources.join("mellowd").join("mellowd.exe");
        if !exe.exists() {
            return Err(format!("bundled sidecar is missing at {}", exe.display()));
        }
        let mut cmd = Command::new(exe);
        #[cfg(windows)]
        {
            // CREATE_NO_WINDOW: never flash a console behind the pet.
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        Ok(cmd)
    }
}

/// Start mellowd and refuse any unverified process already using its port.
pub fn spawn(app: &tauri::AppHandle) -> Result<Option<Child>, String> {
    let addr: SocketAddr = ADDR.parse().expect("ADDR is a literal");
    if port_taken(addr) {
        health(addr).map_err(|error| {
            format!(
                "port {ADDR} is occupied by a service Mellow cannot trust: {error}. \
                 Stop that process and start Mellow again."
            )
        })?;
        eprintln!("[mellow] using the verified mellowd already on {ADDR}");
        return Ok(None);
    }

    let mut child = command(app)?
        .spawn()
        .map_err(|error| format!("could not start mellowd: {error}"))?;
    println!("[mellow] started mellowd (pid {})", child.id());

    let started = Instant::now();
    while started.elapsed() < READY_TIMEOUT {
        if health(addr).is_ok() {
            println!("[mellow] mellowd identity and health verified");
            return Ok(Some(child));
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("could not inspect mellowd: {error}"))?
        {
            return Err(format!("mellowd exited before it became healthy ({status})"));
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    Err(format!(
        "mellowd did not become healthy within {} seconds",
        READY_TIMEOUT.as_secs()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_identity_requires_service_protocol_and_version() {
        let good = format!(
            r#"{{"ok":true,"service":"mellowd","protocol":1,"version":"{}"}}"#,
            env!("CARGO_PKG_VERSION")
        );
        assert!(validate_health(&good).is_ok());
        assert!(validate_health(r#"{"ok":true}"#).is_err());
        assert!(validate_health(
            r#"{"ok":true,"service":"mellowd","protocol":2,"version":"1.0.0"}"#
        )
        .is_err());
    }
}
