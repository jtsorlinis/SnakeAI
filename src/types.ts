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
  state: Uint8Array;
  action: number;
  reward: number;
  nextState: Uint8Array;
  done: boolean;
};

export type NetworkState = {
  observation: Uint8Array | null;
  qValues: Float32Array | null;
  action: number;
};

export type TrainerState = {
  boardAgent: Agent;
  rewardHistory: readonly number[];
  episodeCount: number;
  totalSteps: number;
  epsilon: number;
  replaySize: number;
  avgReturn: number;
  bestReturn: number;
  loss: number;
  network: NetworkState;
};
