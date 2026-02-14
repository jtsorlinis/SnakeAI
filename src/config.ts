import type { Point } from "./types";

export const GRID_SIZE = 20;
export const TILE_SIZE = 24;
export const BOARD_SIZE = GRID_SIZE * TILE_SIZE;
export const MAX_SCORE = GRID_SIZE * GRID_SIZE - 3;

export const NET_WIDTH = 480;
export const NET_HEIGHT = 480;
export const CHART_WIDTH = 300;
export const CHART_HEIGHT = 140;

export const POP_SIZE = 180;
export const ELITE_COUNT = 8;
export const TOURNAMENT_SIZE = 4;
export const MUTATION_RATE = 0.08;
export const MUTATION_SIZE = 0.35;
export const BASE_HUNGER = GRID_SIZE * GRID_SIZE;

export const NORMAL_STEPS_PER_SECOND = 30;
export const TURBO_TIME_BUDGET_MS = 12;

export const INPUTS = 11;
export const HIDDEN = 10;
export const OUTPUTS = 3;

export const IH_COUNT = INPUTS * HIDDEN;
export const H_BIAS_COUNT = HIDDEN;
export const HO_COUNT = HIDDEN * OUTPUTS;
export const O_BIAS_COUNT = OUTPUTS;

export const OFFSET_IH = 0;
export const OFFSET_H_BIAS = OFFSET_IH + IH_COUNT;
export const OFFSET_HO = OFFSET_H_BIAS + H_BIAS_COUNT;
export const OFFSET_O_BIAS = OFFSET_HO + HO_COUNT;
export const GENE_COUNT = OFFSET_O_BIAS + O_BIAS_COUNT;

export const NET_HOVER_RADIUS = 8;

export const INPUT_LABELS = [
  "Front blocked",
  "Left blocked",
  "Right blocked",
  "1/Dist front",
  "1/Dist left",
  "1/Dist right",
  "Food x > head x",
  "Food y > head y",
  "Dir up",
  "Dir right",
  "Dir down",
] as const;

export const OUTPUT_LABELS = ["Straight", "Turn left", "Turn right"] as const;

export const DIRS: ReadonlyArray<Point> = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
];
