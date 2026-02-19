import type { Point } from "./types";

export const MIN_GRID_SIZE = 10;
export const MAX_GRID_SIZE = 30;
export const GRID_SIZE_STEP = 5;
export const DEFAULT_GRID_SIZE = 10;

export let GRID_SIZE = DEFAULT_GRID_SIZE;
export const TILE_SIZE = 24;
export let BOARD_SIZE = 0;
export let MAX_SCORE = 0;
export let BASE_HUNGER = 0;

export const NET_WIDTH = 480;
export const NET_HEIGHT = 480;
export const CHART_WIDTH = 300;
export const CHART_HEIGHT = 140;

export const OBS_CHANNELS = 3;
export const OUTPUTS = 3;

export const OBS_LABELS = ["Head", "Body", "Food"] as const;
export const OUTPUT_LABELS = ["Straight", "Turn left", "Turn right"] as const;

export const NORMAL_STEPS_PER_SECOND = 30;
export const TURBO_TIME_BUDGET_MS = 12;

export const TRAIN_ENVS = 24;
export const REPLAY_CAPACITY = 15_000;
export const TRAIN_START_SIZE = 5_000;
export const BATCH_SIZE = 32;
export const TRAIN_EVERY_STEPS = 2;
export const GRADIENT_STEPS = 1;
export const TARGET_UPDATE_STEPS = 1_000;
export const N_STEP_RETURNS = 3;

export const GAMMA = 0.99;
export const LEARNING_RATE = 0.0005;
export const ADAM_BETA1 = 0.9;
export const ADAM_BETA2 = 0.999;
export const ADAM_EPSILON = 1e-8;

export const PRIORITY_ALPHA = 0.6;
export const PRIORITY_BETA_START = 0.4;
export const PRIORITY_BETA_END = 1;
export const PRIORITY_BETA_DECAY_STEPS = 2_000_000;
export const PRIORITY_EPSILON = 1e-3;

export const EPSILON_START = 1;
export const EPSILON_END = 0.1;
export const EPSILON_DECAY_STEPS = 2_000_000;

export const REWARD_EAT = 1;
export const REWARD_COLLISION_DEATH = -1;
export const REWARD_STARVATION_DEATH = 0;

export function rewardStepPenalty(): number {
  return -1 / (GRID_SIZE * GRID_SIZE);
}

export const DIRS: ReadonlyArray<Point> = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
];

function recalculateGridConfig(): void {
  BOARD_SIZE = GRID_SIZE * TILE_SIZE;
  MAX_SCORE = GRID_SIZE * GRID_SIZE - 3;
  BASE_HUNGER = GRID_SIZE * GRID_SIZE;
}

recalculateGridConfig();

export function setGridSize(nextSize: number): number {
  const clamped = Math.min(MAX_GRID_SIZE, Math.max(MIN_GRID_SIZE, nextSize));
  const snapped =
    MIN_GRID_SIZE +
    Math.round((clamped - MIN_GRID_SIZE) / GRID_SIZE_STEP) * GRID_SIZE_STEP;
  const normalized = Math.min(MAX_GRID_SIZE, Math.max(MIN_GRID_SIZE, snapped));

  if (normalized !== GRID_SIZE) {
    GRID_SIZE = normalized;
    recalculateGridConfig();
  }

  return GRID_SIZE;
}

export function observationSize(gridSize = GRID_SIZE): number {
  return OBS_CHANNELS * gridSize * gridSize;
}
