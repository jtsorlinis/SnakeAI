export type Point = { x: number; y: number };

export type PolicyParams = Float32Array;
export type TrainerAlgorithm = "ga" | "cmaes" | "ppo" | "boardppo";
export type PolicyPlaybackMode = "greedy" | "stochastic";

export type ReplayStats = {
  games: number;
  averageScore: number;
  wins: number;
};

export type TerminalReason = "collision" | "hunger" | "solved" | null;

export type Agent = {
  policy: PolicyParams;
  body: Point[];
  dir: number;
  food: Point;
  alive: boolean;
  score: number;
  steps: number;
  hunger: number;
  fitness: number;
  terminalReason: TerminalReason;
};

export type NetworkActivations = {
  input: Float32Array;
  hidden: Float32Array[];
  output: Float32Array;
  best: number;
};

// What Board PPO's policy is looking at: the board around the snake's head,
// turned so the snake faces up. Cells are listed row by row.
export type VisionView = {
  // Cells per side.
  size: number;
  // 1 where the cell is on the board, 0 beyond a wall.
  inside: Uint8Array;
  // 0 for an empty cell, otherwise how long the body stays there, scaled so
  // the head is 1 and the tail tip is close to 0.
  body: Float32Array;
  // Cell index of the food.
  food: number;
  // Preference for going straight, left and right; they sum to 1.
  probabilities: Float32Array;
  best: number;
};

export type TrainerState = {
  algorithm: TrainerAlgorithm;
  boardAgent: Agent;
  boardAgents: readonly Agent[];
  fitnessHistory: readonly number[];
  iteration: number;
  iterationLabel: string;
  alive: number;
  batchSize: number;
  batchSizeLabel: string;
  bestEverScore: number;
  bestEverFitness: number;
  replay: ReplayStats;
  staleIterations: number;
  staleLabel: string;
  historyLabel: string;
  policySourceLabel: string;
  playbackMode: PolicyPlaybackMode;
  playbackModeEnabled: boolean;
  network: {
    policy: PolicyParams | null;
    activations: NetworkActivations | null;
  };
  // Set by Board PPO, whose network is too large to draw weight by weight.
  vision?: VisionView | null;
};

export interface TrainerController {
  reset(): void;
  simulate(stepCount: number): void;
  getState(randomBoardCount?: number): TrainerState;
  onGridSizeChanged(): void;
  setPlaybackMode(mode: PolicyPlaybackMode): void;
  getPlaybackMode(): PolicyPlaybackMode;
  supportsPlaybackMode(): boolean;
}
