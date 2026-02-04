import "./style.css";
import { Game } from "./Game";

window.addEventListener("DOMContentLoaded", () => {
  new Game();
  setupWakeLock();
});

function setupWakeLock() {
  if (!("wakeLock" in navigator)) return;

  let wakeLock: WakeLockSentinel | null = null;
  let wakeLockEnabled = false;

  const requestWakeLock = async () => {
    if (!wakeLockEnabled) return;
    try {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", () => {
        wakeLock = null;
      });
    } catch {
      // Ignore if the browser blocks the request.
    }
  };

  const onUserGesture = () => {
    if (wakeLockEnabled) return;
    wakeLockEnabled = true;
    requestWakeLock();
  };

  window.addEventListener("pointerdown", onUserGesture, { once: true });
  window.addEventListener("keydown", onUserGesture, { once: true });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && wakeLockEnabled) {
      requestWakeLock();
    }
  });
}
