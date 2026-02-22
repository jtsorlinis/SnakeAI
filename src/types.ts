export type Point = { x: number; y: number };

export type PolicyParams = Float32Array;

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
};

export type NetworkActivations = {
  input: Float32Array;
  hidden: Float32Array[];
  output: Float32Array;
  best: number;
};

export type TrainerState = {
  boardAgent: Agent;
  boardAgents: readonly Agent[];
  fitnessHistory: readonly number[];
  ppoUpdate: number;
  alive: number;
  rolloutBatchSize: number;
  bestEverScore: number;
  bestEverFitness: number;
  updatesSinceBest: number;
  network: {
    policy: PolicyParams | null;
    activations: NetworkActivations | null;
  };
};
