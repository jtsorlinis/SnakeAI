export type Point = { x: number; y: number };

export type Agent = {
  body: Point[];
  dir: number;
  food: Point;
  alive: boolean;
  score: number;
  steps: number;
  hunger: number;
  episodeReturn: number;
};

export type Transition = {
  state: Float32Array;
  action: number;
  reward: number;
  nextState: Float32Array;
  done: boolean;
};

export type NetworkState = {
  observation: Float32Array | null;
  policy: Float32Array | null;
  value: number;
  action: number;
};

export type TrainerState = {
  boardAgent: Agent;
  rewardHistory: readonly number[];
  episodeCount: number;
  totalSteps: number;
  stepsPerSecond: number;
  bestScore: number;
  avgReturn: number;
  bestReturn: number;
  totalLoss: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  approxKl: number;
  clipFraction: number;
  updates: number;
  rolloutProgress: number;
  network: NetworkState;
};
