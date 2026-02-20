import type { Point } from "./types";

export const MIN_GRID_SIZE = 20;
export const MAX_GRID_SIZE = 20;
export const GRID_SIZE_STEP = 5;
export const DEFAULT_GRID_SIZE = 20;

export let GRID_SIZE = DEFAULT_GRID_SIZE;
export const TILE_SIZE = 24;
export let BOARD_SIZE = 0;
export let MAX_SCORE = 0;
export let BASE_HUNGER = 0;

export const NET_WIDTH = 480;
export const NET_HEIGHT = 480;
export const CHART_WIDTH = 300;
export const CHART_HEIGHT = 140;

export const OBS_CHANNELS = 5;
export const OUTPUTS = 3;

export const OBS_LABELS = [
  "Food",
  "Head",
  "Body age",
  "Dir X",
  "Dir Y",
] as const;
export const OUTPUT_LABELS = ["Straight", "Turn left", "Turn right"] as const;

export const NORMAL_STEPS_PER_SECOND = 30;
export const TURBO_TIME_BUDGET_MS = 12;

export const TRAIN_ENVS = 24;
export const PPO_ROLLOUT_STEPS = 64;
export const PPO_EPOCHS = 4;
export const PPO_MINIBATCH_SIZE = 256;

export const GAMMA = 0.99;
export const GAE_LAMBDA = 0.95;
export const LEARNING_RATE = 0.0003;
export const ADAM_BETA1 = 0.9;
export const ADAM_BETA2 = 0.999;
export const ADAM_EPSILON = 1e-8;
export const PPO_CLIP_RANGE = 0.2;
export const PPO_VALUE_LOSS_COEF = 0.5;
export const PPO_ENTROPY_COEF = 0.01;
export const PPO_TARGET_KL = 0.03;

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
