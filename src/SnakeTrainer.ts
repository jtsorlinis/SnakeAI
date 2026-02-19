import {
  BATCH_SIZE,
  EPSILON_DECAY_STEPS,
  EPSILON_END,
  EPSILON_START,
  GAMMA,
  GRADIENT_STEPS,
  N_STEP_RETURNS,
  OUTPUTS,
  PRIORITY_BETA_DECAY_STEPS,
  PRIORITY_BETA_END,
  PRIORITY_BETA_START,
  REPLAY_CAPACITY,
  TARGET_UPDATE_STEPS,
  TRAIN_ENVS,
  TRAIN_EVERY_STEPS,
  TRAIN_START_SIZE,
  observationSize,
} from "./config";
import { argMax, ConvDQN } from "./ConvDQN";
import { ReplayBuffer } from "./ReplayBuffer";
import { SnakeEnvironment } from "./SnakeEnvironment";
import type { Agent, TrainerState, Transition } from "./types";

const HISTORY_LIMIT = 1000;
const AVG_WINDOW = 100;

type NStepEntry = Transition;

export class SnakeTrainer {
  private readonly environment = new SnakeEnvironment();
  private readonly replay = new ReplayBuffer(REPLAY_CAPACITY);
  private readonly nStepDiscount = Math.pow(GAMMA, N_STEP_RETURNS);

  private online = new ConvDQN();
  private target = new ConvDQN();

  private trainingAgents: Agent[] = [];
  private trainingStates: Uint8Array[] = [];
  private nStepQueues: NStepEntry[][] = [];

  private showcaseAgent: Agent = this.environment.createAgent();
  private showcaseObservation: Uint8Array = new Uint8Array(observationSize());
  private showcaseQValues = new Float32Array(OUTPUTS);
  private showcaseAction = 0;

  private terminalState = new Uint8Array(observationSize());
  private readonly actionQScratch = new Float32Array(OUTPUTS);

  private rewardHistory: number[] = [];
  private episodeCount = 0;
  private totalSteps = 0;
  private epsilon = EPSILON_START;
  private priorityBeta = PRIORITY_BETA_START;
  private loss = 0;
  private bestReturn = Number.NEGATIVE_INFINITY;
  private gradientSteps = 0;
  private lastTargetSyncGradientStep = 0;
  private stepsPerSecond = 0;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.online.dispose();
    this.target.dispose();
    this.online = new ConvDQN();
    this.target = new ConvDQN();
    this.target.copyWeightsFrom(this.online);

    this.replay.clear();

    this.trainingAgents = [];
    this.trainingStates = [];
    this.nStepQueues = [];

    for (let i = 0; i < TRAIN_ENVS; i++) {
      const agent = this.environment.createAgent();
      const state = this.environment.observe(agent);
      this.trainingAgents.push(agent);
      this.trainingStates.push(state);
      this.nStepQueues.push([]);
    }

    this.showcaseAgent = this.environment.createAgent();
    this.showcaseObservation = new Uint8Array(observationSize());
    this.environment.observe(this.showcaseAgent, this.showcaseObservation);
    this.online.predict(this.showcaseObservation, this.showcaseQValues);
    this.showcaseAction = argMax(this.showcaseQValues);

    this.terminalState = new Uint8Array(observationSize());

    this.rewardHistory = [];
    this.episodeCount = 0;
    this.totalSteps = 0;
    this.epsilon = EPSILON_START;
    this.priorityBeta = PRIORITY_BETA_START;
    this.loss = 0;
    this.bestReturn = Number.NEGATIVE_INFINITY;
    this.gradientSteps = 0;
    this.lastTargetSyncGradientStep = 0;
    this.stepsPerSecond = 0;
  }

  public simulate(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      this.stepTrainingEnvironments();
      this.trainOnlineNetwork();
      this.stepShowcase();
      this.epsilon = this.currentEpsilon(this.totalSteps);
      this.priorityBeta = this.currentPriorityBeta(this.totalSteps);
    }
  }

  public simulateShowcase(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      this.stepShowcase();
    }
  }

  public getState(): TrainerState {
    const avgReturn = this.averageReturn();

    return {
      boardAgent: this.showcaseAgent,
      rewardHistory: this.rewardHistory,
      episodeCount: this.episodeCount,
      totalSteps: this.totalSteps,
      epsilon: this.epsilon,
      priorityBeta: this.priorityBeta,
      replaySize: this.replay.size,
      stepsPerSecond: this.stepsPerSecond,
      avgReturn,
      bestReturn:
        this.bestReturn === Number.NEGATIVE_INFINITY ? 0 : this.bestReturn,
      loss: this.loss,
      network: {
        observation: this.showcaseObservation,
        qValues: this.showcaseQValues,
        action: this.showcaseAction,
      },
    };
  }

  public onGridSizeChanged(): void {
    this.reset();
  }

  public setStepsPerSecond(value: number): void {
    this.stepsPerSecond = value;
  }

  private stepTrainingEnvironments(): void {
    for (let i = 0; i < this.trainingAgents.length; i++) {
      const agent = this.trainingAgents[i];
      const state = this.trainingStates[i];

      const action = this.selectAction(state, true);
      const result = this.environment.step(agent, action);
      agent.episodeReturn += result.reward;

      const nextState = result.done
        ? this.terminalState
        : this.environment.observe(agent);

      this.nStepQueues[i].push({
        state,
        action,
        reward: result.reward,
        nextState,
        done: result.done,
      });
      this.flushNStepQueue(this.nStepQueues[i]);
      this.totalSteps += 1;

      if (result.done) {
        this.recordEpisodeReturn(agent.episodeReturn);
        this.episodeCount += 1;
        this.environment.resetAgent(agent);
        this.trainingStates[i] = this.environment.observe(agent);
      } else {
        this.trainingStates[i] = nextState;
      }
    }
  }

  private flushNStepQueue(queue: NStepEntry[]): void {
    while (queue.length > 0) {
      const queueEndsWithDone = queue[queue.length - 1].done;
      if (!queueEndsWithDone && queue.length < N_STEP_RETURNS) {
        return;
      }

      let reward = 0;
      let discount = 1;
      let nextState = queue[0].nextState;
      let done = false;
      const steps = Math.min(N_STEP_RETURNS, queue.length);

      for (let i = 0; i < steps; i++) {
        const transition = queue[i];
        reward += discount * transition.reward;
        discount *= GAMMA;
        nextState = transition.nextState;
        done = transition.done;
        if (done) {
          break;
        }
      }

      const first = queue[0];
      this.replay.push({
        state: first.state,
        action: first.action,
        reward,
        nextState,
        done,
      });

      queue.shift();
    }
  }

  private selectAction(state: Uint8Array, explore: boolean): number {
    if (explore && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * OUTPUTS);
    }

    this.online.predict(state, this.actionQScratch);
    return argMax(this.actionQScratch);
  }

  private trainOnlineNetwork(): void {
    if (this.replay.size < TRAIN_START_SIZE) {
      return;
    }

    if (this.totalSteps % TRAIN_EVERY_STEPS !== 0) {
      return;
    }

    let latestLoss = this.loss;

    for (let i = 0; i < GRADIENT_STEPS; i++) {
      const sample = this.replay.sample(BATCH_SIZE, this.priorityBeta);
      if (sample.transitions.length === 0) {
        break;
      }

      const result = this.online.trainBatch(
        sample.transitions,
        sample.weights,
        this.target,
        this.nStepDiscount,
      );
      latestLoss = result.loss;
      this.replay.updatePriorities(sample.indices, result.tdErrors);
      this.gradientSteps += 1;
    }

    this.loss =
      this.loss === 0 ? latestLoss : this.loss * 0.97 + latestLoss * 0.03;

    if (
      this.gradientSteps - this.lastTargetSyncGradientStep >=
      TARGET_UPDATE_STEPS
    ) {
      this.target.copyWeightsFrom(this.online);
      this.lastTargetSyncGradientStep = this.gradientSteps;
    }
  }

  private stepShowcase(): void {
    if (!this.showcaseAgent.alive) {
      this.environment.resetAgent(this.showcaseAgent);
    }

    this.environment.observe(this.showcaseAgent, this.showcaseObservation);
    this.online.predict(this.showcaseObservation, this.showcaseQValues);
    const action = argMax(this.showcaseQValues);
    this.environment.step(this.showcaseAgent, action);

    if (!this.showcaseAgent.alive) {
      this.environment.resetAgent(this.showcaseAgent);
    }

    this.environment.observe(this.showcaseAgent, this.showcaseObservation);
    this.online.predict(this.showcaseObservation, this.showcaseQValues);
    this.showcaseAction = argMax(this.showcaseQValues);
  }

  private recordEpisodeReturn(value: number): void {
    this.rewardHistory.push(value);
    if (this.rewardHistory.length > HISTORY_LIMIT) {
      this.rewardHistory.shift();
    }

    if (value > this.bestReturn) {
      this.bestReturn = value;
    }
  }

  private averageReturn(): number {
    if (this.rewardHistory.length === 0) {
      return 0;
    }

    const windowSize = Math.min(AVG_WINDOW, this.rewardHistory.length);
    let sum = 0;

    for (
      let i = this.rewardHistory.length - windowSize;
      i < this.rewardHistory.length;
      i++
    ) {
      sum += this.rewardHistory[i];
    }

    return sum / windowSize;
  }

  private currentEpsilon(steps: number): number {
    const progress = Math.min(1, steps / EPSILON_DECAY_STEPS);
    return EPSILON_START + (EPSILON_END - EPSILON_START) * progress;
  }

  private currentPriorityBeta(steps: number): number {
    const progress = Math.min(1, steps / PRIORITY_BETA_DECAY_STEPS);
    return (
      PRIORITY_BETA_START +
      (PRIORITY_BETA_END - PRIORITY_BETA_START) * progress
    );
  }
}
