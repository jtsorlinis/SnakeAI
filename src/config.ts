import type { Point } from "./types";

export const MIN_GRID_SIZE = 10;
export const MAX_GRID_SIZE = 30;
export const GRID_SIZE_STEP = 5;
export const DEFAULT_GRID_SIZE = 10;

export let GRID_SIZE = DEFAULT_GRID_SIZE;
export let MAX_SCORE = 0;

// Drawing-space sizes; canvases are scaled to their CSS size and pixel ratio.
export const BOARD_VIEW_SIZE = 480;
export const NET_WIDTH = 480;
export const NET_HEIGHT = 480;
export const MULTI_VIEW_COUNT = 25;
export const MULTI_VIEW_COLUMNS = 5;
// Extra snakes replaying the showcased policy for the replay average.
export const REPLAY_AGENT_COUNT = 8;

export const POP_SIZE = 180;
export const ELITE_COUNT = 8;
export const TOURNAMENT_SIZE = 4;
// GA children are a copy of one parent with Gaussian noise of this size added
// to every weight. Changing a few weights by a lot, and mixing two parents,
// left over half of populations stuck on food chasing that traps itself.
export const MUTATION_SIGMA = 0.12;
// Each GA/CMA-ES policy plays this many games per generation and is ranked on
// the mean. A single game is mostly food-placement luck.
export const EVAL_GAMES = 3;
// Extra games a candidate plays before it can replace the showcased policy,
// so one lucky evaluation can't take over the display for good.
export const VALIDATION_GAMES = 10;
// Some runs settle on a pattern that rarely clears the board (edge-hugging
// around 60, or a sweep that dies near the end) and almost never escape. After
// this many GA/CMA-ES generations a population's leader is tested on fresh
// games, and if it clears fewer than half of them the population is reseeded.
export const RESTART_AFTER = 300;
export const RESTART_CHECK_GAMES = 20;
export const ES_PARENT_COUNT = 12;
export const CMA_INITIAL_SIGMA = 0.55;
export const ROLLOUT_BATCH_SIZE = 180;
export const PPO_GAMMA = 1;
// The policy only sees local sensors, so the critic cannot tell a snake that is
// boxing itself in from one that is safe. Bootstrapping from it (lambda < 1)
// rewards chasing food greedily; Monte Carlo advantages credit the late deaths
// that greedy chasing causes.
export const PPO_GAE_LAMBDA = 1;
export const PPO_ROLLOUT_HORIZON = 128;
export const PPO_CLIP_EPSILON = 0.2;
export const PPO_EPOCHS = 4;
export const PPO_MINIBATCH_SIZE = 256;
export const PPO_POLICY_LEARNING_RATE = 1.2e-3;
export const PPO_VALUE_LEARNING_RATE = 1e-3;
export const PPO_VALUE_COEFFICIENT = 0.5;
// The entropy bonus fades linearly from this to zero over the first
// PPO_ENTROPY_FADE_UPDATES of a run. Kept on, the policy still picks a move
// it doesn't prefer on 5-10% of steps after 10000 updates and never settles
// on the sweep that clears the board.
export const PPO_ENTROPY_COEFFICIENT = 0.01;
export const PPO_ENTROPY_FADE_UPDATES = 500;
// PPO's reward for clearing the board; each food is worth 1. Near the end a
// snake expects to die soon, so finishing has to be worth much more than the
// last few food. It doesn't help a run that never clears.
export const PPO_CLEAR_REWARD = 50;
export const PPO_MAX_GRAD_NORM = 1;
// PPO updates between greedy validations of the policy for the display.
export const PPO_VALIDATE_EVERY = 10;
// PPO's budget to learn to clear the board before its networks are reset.
// Runs that get there are usually clearing over half their games by then.
export const PPO_RESTART_AFTER = 1000;
// Runs that go on to clear the board have mostly settled on their moves this
// many updates in (rollout policy entropy, in nats, at or below the limit).
// Runs still choosing more randomly than that rarely get there, so they reset
// early. The limit depends on PPO_ENTROPY_COEFFICIENT and its fade.
export const PPO_SETTLE_CHECK_AFTER = 300;
export const PPO_SETTLED_ENTROPY = 0.2;
// Board PPO: the policy reads the whole board, re-drawn from the snake's point
// of view, instead of the ten sensors. It uses the rollout batch, horizon, clip
// range, epochs and minibatch size above.
// A first layer of 128 showed no clear gain (90% of games cleared against
// 88%, four seeds each) and took about 70% longer per update.
export const VISION_HIDDEN_UNITS: readonly number[] = [64, 64];
// With a critic that sees the whole board, bootstrapped advantages are sound.
// The discount is this close to 1 on purpose. At 0.998, food sooner was worth
// enough more that small risks paid: the policy rushed, and cleared about 89%
// of games however long it trained. A discount of 1 did no better than this.
export const VISION_GAMMA = 0.9995;
export const VISION_GAE_LAMBDA = 0.95;
// The learning rate drops to the settled value once the policy clears the
// board in at least half of its validation games. Without the drop, training
// at this discount fell apart for a while every few million steps. Dropping
// it on a schedule instead left one run in four stuck at a score of about 90
// without ever clearing the board.
export const VISION_LEARNING_RATE = 3e-4;
export const VISION_SETTLED_LEARNING_RATE = 1e-4;
// Constant. Cutting it to 0.001 once the policy was clearing didn't stop
// training from falling apart.
export const VISION_ENTROPY_COEFFICIENT = 0.01;
export const VISION_MAX_GRAD_NORM = 0.5;
// Board PPO's reward for clearing the board; each food is worth 1 and dying
// costs 1. The 50 that helps sensor PPO made training unstable here.
export const VISION_CLEAR_REWARD = 1;
// How much of a running average of the policy weights each update keeps; the
// rest comes from the live weights. The average plays more steadily than the
// live weights, so it is what gets validated and shown.
export const VISION_POLICY_AVERAGE_DECAY = 0.95;
// A Board PPO update is too slow to run in one go, so it is spread over
// simulation steps, this many samples at a time (about the cost of moving
// the whole rollout batch once).
export const VISION_TRAIN_SAMPLES_PER_STEP = 60;
export let BASE_HUNGER = 0;

export const NORMAL_STEPS_PER_SECOND = 15;
export const TURBO_TIME_BUDGET_MS = 12;

export const INPUTS = 10;
export const OUTPUTS = 3;
// Extra observations only the PPO critic sees: snake length and
// remaining hunger. The return from a state depends heavily on both, but the
// policy's local sensors cannot see them.
export const CRITIC_CONTEXT_INPUTS = 2;
export const CRITIC_INPUTS = INPUTS + CRITIC_CONTEXT_INPUTS;
export const CRITIC_HIDDEN_UNITS: readonly number[] = [8];
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
export let POLICY_PARAM_COUNT = 0;

function recalculateGridConfig(): void {
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
  POLICY_PARAM_COUNT = OFFSET_O_BIAS + O_BIAS_COUNT;
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
