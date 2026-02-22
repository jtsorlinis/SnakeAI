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
export const MULTI_VIEW_COUNT = 25;
export const MULTI_VIEW_COLUMNS = 5;

export const POP_SIZE = 180;
export const ELITE_COUNT = 8;
export const TOURNAMENT_SIZE = 4;
export const MUTATION_RATE = 0.08;
export const MUTATION_SIZE = 0.35;
export let BASE_HUNGER = 0;

export const NORMAL_STEPS_PER_SECOND = 15;
export const TURBO_TIME_BUDGET_MS = 12;

export const INPUTS = 10;
export const OUTPUTS = 3;
export const BASE_HIDDEN_UNITS = 8;

export const HIDDEN_LAYERS = 1;
export let HIDDEN_LAYER_UNITS: number[] = [];
export let IH_COUNT = 0;
export let HH_COUNT = 0;
export let H_BIAS_COUNT = 0;
export let HO_COUNT = 0;
export let O_BIAS_COUNT = 0;

export let OFFSET_IH = 0;
export let OFFSET_HH = 0;
export let OFFSET_H_BIAS = 0;
export let OFFSET_HO = 0;
export let OFFSET_O_BIAS = 0;
export let GENE_COUNT = 0;

function recalculateGridConfig(): void {
  BOARD_SIZE = GRID_SIZE * TILE_SIZE;
  MAX_SCORE = GRID_SIZE * GRID_SIZE - 3;
  BASE_HUNGER = GRID_SIZE * GRID_SIZE;
}

function buildHiddenLayerUnits(layerCount: number): number[] {
  const units: number[] = [];
  let width = BASE_HIDDEN_UNITS;

  for (let layer = 0; layer < layerCount; layer++) {
    units.push(width);
    width = Math.max(2, Math.round((2 * width) / 3));
  }

  return units;
}

function recalculateTopology(): void {
  HIDDEN_LAYER_UNITS = buildHiddenLayerUnits(HIDDEN_LAYERS);

  IH_COUNT = HIDDEN_LAYER_UNITS.length > 0 ? INPUTS * HIDDEN_LAYER_UNITS[0] : 0;
  HH_COUNT = 0;
  for (let layer = 1; layer < HIDDEN_LAYER_UNITS.length; layer++) {
    HH_COUNT += HIDDEN_LAYER_UNITS[layer - 1] * HIDDEN_LAYER_UNITS[layer];
  }
  H_BIAS_COUNT = HIDDEN_LAYER_UNITS.reduce((sum, units) => sum + units, 0);
  const outputInputs =
    HIDDEN_LAYER_UNITS.length > 0
      ? HIDDEN_LAYER_UNITS[HIDDEN_LAYER_UNITS.length - 1]
      : INPUTS;
  HO_COUNT = outputInputs * OUTPUTS;
  O_BIAS_COUNT = OUTPUTS;

  OFFSET_IH = 0;
  OFFSET_HH = OFFSET_IH + IH_COUNT;
  OFFSET_H_BIAS = OFFSET_HH + HH_COUNT;
  OFFSET_HO = OFFSET_H_BIAS + H_BIAS_COUNT;
  OFFSET_O_BIAS = OFFSET_HO + HO_COUNT;
  GENE_COUNT = OFFSET_O_BIAS + O_BIAS_COUNT;
}

recalculateGridConfig();
recalculateTopology();

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
