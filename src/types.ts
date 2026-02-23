export type Point = { x: number; y: number };

export type Genome = Float32Array;

export type Agent = {
  genome: Genome;
  body: Point[];
  dir: number;
  food: Point;
  alive: boolean;
  score: number;
  steps: number;
  stepsSinceFood: number;
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
  generation: number;
  evaluationEpisode: number;
  evaluationEpisodeTarget: number;
  alive: number;
  populationSize: number;
  bestEverScore: number;
  bestEverFitness: number;
  staleGenerations: number;
  network: {
    genome: Genome | null;
    activations: NetworkActivations | null;
  };
};
