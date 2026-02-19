export type Point = { x: number; y: number };

export type NodeType = "input" | "hidden" | "output";

export type NodeGene = {
  id: number;
  type: NodeType;
  layer: number;
  bias: number;
  ioIndex?: number;
};

export type ConnectionGene = {
  innovation: number;
  from: number;
  to: number;
  weight: number;
  enabled: boolean;
};

export type Genome = {
  nodes: NodeGene[];
  connections: ConnectionGene[];
};

export type Agent = {
  genome: Genome;
  body: Point[];
  dir: number;
  food: Point;
  alive: boolean;
  score: number;
  steps: number;
  hunger: number;
  fitness: number;
  speciesId: number;
};

export type NetworkActivationNode = {
  id: number;
  type: NodeType;
  layer: number;
  bias: number;
  ioIndex?: number;
  label: string;
  value: number;
};

export type NetworkActivationEdge = {
  from: number;
  to: number;
  weight: number;
  enabled: boolean;
  label: string;
};

export type NetworkActivations = {
  nodes: NetworkActivationNode[];
  edges: NetworkActivationEdge[];
  output: Float32Array;
  best: number;
};

export type TrainerState = {
  boardAgent: Agent;
  fitnessHistory: readonly number[];
  generation: number;
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
