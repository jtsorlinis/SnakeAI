type WakeLockSentinelLike = {
  released: boolean;
  release: () => Promise<void>;
  addEventListener: (
    type: "release",
    listener: EventListenerOrEventListenerObject,
  ) => void;
};

type WakeLockApi = {
  request: (type: "screen") => Promise<WakeLockSentinelLike>;
};

let wakeLock: WakeLockSentinelLike | null = null;
let wakeLockInitialized = false;

function getWakeLockApi(): WakeLockApi | null {
  const wakeLockApi = (navigator as Navigator & { wakeLock?: WakeLockApi })
    .wakeLock;
  return wakeLockApi ?? null;
}

async function requestWakeLock(): Promise<void> {
  if (document.visibilityState !== "visible") {
    return;
  }

  if (wakeLock && !wakeLock.released) {
    return;
  }

  const wakeLockApi = getWakeLockApi();
  if (!wakeLockApi) {
    return;
  }

  try {
    const sentinel = await wakeLockApi.request("screen");
    wakeLock = sentinel;
    sentinel.addEventListener("release", () => {
      wakeLock = null;
      if (document.visibilityState === "visible") {
        void requestWakeLock();
      }
    });
  } catch (error) {
    console.debug("Wake lock request failed.", error);
  }
}

async function releaseWakeLock(): Promise<void> {
  if (!wakeLock) {
    return;
  }

  const current = wakeLock;
  wakeLock = null;

  try {
    await current.release();
  } catch (error) {
    console.debug("Wake lock release failed.", error);
  }
}

function handleVisibilityChange(): void {
  if (document.visibilityState === "visible") {
    void requestWakeLock();
  } else {
    void releaseWakeLock();
  }
}

function handleFocus(): void {
  void requestWakeLock();
}

export function enableWakeLock(): void {
  if (wakeLockInitialized) {
    return;
  }

  if (!getWakeLockApi()) {
    return;
  }

  wakeLockInitialized = true;

  document.addEventListener("visibilitychange", handleVisibilityChange);
  window.addEventListener("focus", handleFocus);

  const requestOnInteraction = () => {
    void requestWakeLock();
  };
  window.addEventListener("pointerdown", requestOnInteraction, { once: true });
  window.addEventListener("keydown", requestOnInteraction, { once: true });

  void requestWakeLock();
}
