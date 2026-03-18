export type Point = { x: number; y: number };

export type PolicyParams = Float32Array;
export type TrainerAlgorithm =
  | "a2c"
  | "ppo"
  | "reinforce"
  | "openai-es"
  | "pso"
  | "ga"
  | "es"
  | "cmaes";
export type PolicyPlaybackMode = "greedy" | "stochastic";

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
