const CHECKPOINT_DB_NAME = "snake-es-checkpoint-db";
const CHECKPOINT_DB_VERSION = 1;
const CHECKPOINT_STORE_NAME = "checkpoints";
const CHECKPOINT_STATS_KEY = "latest-stats";

export const CHECKPOINT_MODEL_KEY = "snake-es-model";

export type CheckpointStats = {
  version: 2;
  savedAtMs: number;
  gridSize: number;
  episodeCount: number;
  totalSteps: number;
  bestScore: number;
  bestReturn: number;
  rewardHistory: number[];
  fitnessMean: number;
  fitnessStd: number;
  fitnessBest: number;
  updateNorm: number;
  weightNorm: number;
  updates: number;
};

function ensureIndexedDbAvailable(): IDBFactory {
  if (typeof indexedDB === "undefined") {
    throw new Error("IndexedDB is not available in this environment.");
  }
  return indexedDB;
}

function openCheckpointDb(): Promise<IDBDatabase> {
  const indexedDb = ensureIndexedDbAvailable();

  return new Promise((resolve, reject) => {
    const request = indexedDb.open(CHECKPOINT_DB_NAME, CHECKPOINT_DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(CHECKPOINT_STORE_NAME)) {
        db.createObjectStore(CHECKPOINT_STORE_NAME);
      }
    };

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onerror = () => {
      reject(
        request.error ??
          new Error("Failed to open IndexedDB for checkpoint storage."),
      );
    };
  });
}

function putValue(
  db: IDBDatabase,
  key: string,
  value: CheckpointStats,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHECKPOINT_STORE_NAME, "readwrite");
    const store = tx.objectStore(CHECKPOINT_STORE_NAME);
    const request = store.put(value, key);

    request.onerror = () => {
      reject(request.error ?? new Error("Failed to write checkpoint stats."));
    };

    tx.oncomplete = () => {
      resolve();
    };
    tx.onerror = () => {
      reject(tx.error ?? new Error("IndexedDB transaction failed."));
    };
    tx.onabort = () => {
      reject(tx.error ?? new Error("IndexedDB transaction was aborted."));
    };
  });
}

function getValue(db: IDBDatabase, key: string): Promise<CheckpointStats | null> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHECKPOINT_STORE_NAME, "readonly");
    const store = tx.objectStore(CHECKPOINT_STORE_NAME);
    const request = store.get(key);

    request.onsuccess = () => {
      resolve((request.result as CheckpointStats | undefined) ?? null);
    };
    request.onerror = () => {
      reject(request.error ?? new Error("Failed to read checkpoint stats."));
    };

    tx.onabort = () => {
      reject(tx.error ?? new Error("IndexedDB read transaction was aborted."));
    };
  });
}

export async function saveCheckpointStats(stats: CheckpointStats): Promise<void> {
  const db = await openCheckpointDb();
  try {
    await putValue(db, CHECKPOINT_STATS_KEY, stats);
  } finally {
    db.close();
  }
}

export async function loadCheckpointStats(): Promise<CheckpointStats | null> {
  const db = await openCheckpointDb();
  try {
    return await getValue(db, CHECKPOINT_STATS_KEY);
  } finally {
    db.close();
  }
}
