import type { Point } from "./types";

export const MIN_GRID_SIZE = 10;
export const MAX_GRID_SIZE = 30;
export const GRID_SIZE_STEP = 5;
export const DEFAULT_GRID_SIZE = 10;

export let GRID_SIZE = DEFAULT_GRID_SIZE;
export const TILE_SIZE = 24;
export let BOARD_SIZE = 0;
export let MAX_SCORE = 0;

export const NET_WIDTH = 480;
export const NET_HEIGHT = 480;
export const CHART_WIDTH = 300;
export const CHART_HEIGHT = 140;

export const POP_SIZE = 180;
export const TOURNAMENT_SIZE = 4;
export const NEAT_COMPATIBILITY_THRESHOLD = 2.2;
export const NEAT_COMPATIBILITY_GENE_COEFF = 1.0;
export const NEAT_COMPATIBILITY_WEIGHT_COEFF = 0.4;
export const NEAT_WEIGHT_MUTATION_RATE = 0.9;
export const NEAT_WEIGHT_MUTATION_SIZE = 0.35;
export const NEAT_BIAS_MUTATION_RATE = 0.3;
export const NEAT_ADD_CONNECTION_RATE = 0.08;
export const NEAT_ADD_NODE_RATE = 0.02;
export const NEAT_DISABLE_INHERITED_GENE_RATE = 0.9;
export const NEAT_SURVIVAL_RATIO = 0.25;
export const NEAT_MAX_NODES = 24;
export let BASE_HUNGER = 0;

export const NORMAL_STEPS_PER_SECOND = 30;
export const TURBO_TIME_BUDGET_MS = 12;

export const INPUTS = 10;
export const OUTPUTS = 3;

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

export const NET_HOVER_RADIUS = 8;

export const INPUT_LABELS = [
  "Block Ahead",
  "Block Left",
  "Block Right",
  "Tail Ahead",
  "Tail Left",
  "Tail Right",
  "Food Ahead",
  "Food Side",
  "Dir X",
  "Dir Y",
] as const;

export const OUTPUT_LABELS = ["Straight", "Turn left", "Turn right"] as const;

export const DIRS: ReadonlyArray<Point> = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
];
