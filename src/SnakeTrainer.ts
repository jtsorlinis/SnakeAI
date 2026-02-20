import {
  GAE_LAMBDA,
  GAMMA,
  GRID_SIZE,
  OUTPUTS,
  PPO_CLIP_RANGE,
  PPO_ENTROPY_COEF,
  PPO_EPOCHS,
  PPO_MINIBATCH_SIZE,
  PPO_ROLLOUT_STEPS,
  PPO_TARGET_KL,
  PPO_VALUE_LOSS_COEF,
  TRAIN_ENVS,
  observationSize,
} from "./config";
import {
  CHECKPOINT_MODEL_KEY,
  type CheckpointStats,
  loadCheckpointStats,
  saveCheckpointStats,
} from "./checkpointStore";
import { argMax, ConvDQN } from "./ConvDQN";
import { SnakeEnvironment } from "./SnakeEnvironment";
import type { Agent, TrainerState } from "./types";

const HISTORY_LIMIT = 1000;
const AVG_WINDOW = 100;
const METRIC_EMA = 0.1;
const PPO_MINIBATCHES_PER_SIM_TICK = 1;

type PendingPpoUpdate = {
  batchSize: number;
  indices: Int32Array;
  epoch: number;
  cursor: number;
  accumTotalLoss: number;
  accumPolicyLoss: number;
  accumValueLoss: number;
  accumEntropy: number;
  accumApproxKl: number;
  accumClipFraction: number;
  batches: number;
};

const CHECKPOINT_VERSION = 1;

function shuffleIndices(indices: Int32Array): void {
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }
}

function ema(previous: number, next: number): number {
  if (!Number.isFinite(next)) {
    return previous;
  }
  return previous === 0 ? next : previous * (1 - METRIC_EMA) + next * METRIC_EMA;
}

export class SnakeTrainer {
  private readonly environment = new SnakeEnvironment();
  private online = new ConvDQN();

  private trainingAgents: Agent[] = [];
  private trainingStates: Float32Array[] = [];

  private rolloutStates: Float32Array[] = [];
  private rolloutActions = new Int32Array(0);
  private rolloutRewards = new Float32Array(0);
  private rolloutDones = new Uint8Array(0);
  private rolloutValues = new Float32Array(0);
  private rolloutLogProbs = new Float32Array(0);
  private rolloutAdvantages = new Float32Array(0);
  private rolloutReturns = new Float32Array(0);
  private rolloutStep = 0;
  private pendingPpoUpdate: PendingPpoUpdate | null = null;

  private showcaseAgent: Agent = this.environment.createAgent();
  private showcaseObservation: Float32Array = new Float32Array(observationSize());
  private showcasePolicy = new Float32Array(OUTPUTS);
  private showcaseLogits = new Float32Array(OUTPUTS);
  private showcaseAction = 0;
  private showcaseValue = 0;

  private rewardHistory: number[] = [];
  private episodeCount = 0;
  private totalSteps = 0;
  private stepsPerSecond = 0;
  private bestScoreValue = 0;
  private bestReturnValue = Number.NEGATIVE_INFINITY;

  private totalLoss = 0;
  private policyLoss = 0;
  private valueLoss = 0;
  private entropy = 0;
  private approxKl = 0;
  private clipFraction = 0;
  private updates = 0;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.online.dispose();
    this.online = new ConvDQN();

    this.trainingAgents = [];
    this.trainingStates = [];

    for (let i = 0; i < TRAIN_ENVS; i++) {
      const agent = this.environment.createAgent();
      this.trainingAgents.push(agent);
      this.trainingStates.push(this.environment.observe(agent));
    }

    const rolloutSize = PPO_ROLLOUT_STEPS * TRAIN_ENVS;
    this.rolloutStates = new Array<Float32Array>(rolloutSize);
    this.rolloutActions = new Int32Array(rolloutSize);
    this.rolloutRewards = new Float32Array(rolloutSize);
    this.rolloutDones = new Uint8Array(rolloutSize);
    this.rolloutValues = new Float32Array(rolloutSize);
    this.rolloutLogProbs = new Float32Array(rolloutSize);
    this.rolloutAdvantages = new Float32Array(rolloutSize);
    this.rolloutReturns = new Float32Array(rolloutSize);
    this.rolloutStep = 0;
    this.pendingPpoUpdate = null;

    this.showcaseAgent = this.environment.createAgent();
    this.showcaseObservation = new Float32Array(observationSize());
    this.showcasePolicy = new Float32Array(OUTPUTS);
    this.showcaseLogits = new Float32Array(OUTPUTS);
    this.refreshShowcasePrediction();

    this.rewardHistory = [];
    this.episodeCount = 0;
    this.totalSteps = 0;
    this.stepsPerSecond = 0;
    this.bestScoreValue = 0;
    this.bestReturnValue = Number.NEGATIVE_INFINITY;

    this.totalLoss = 0;
    this.policyLoss = 0;
    this.valueLoss = 0;
    this.entropy = 0;
    this.approxKl = 0;
    this.clipFraction = 0;
    this.updates = 0;
  }

  public simulate(stepCount: number): number {
    let envSteps = 0;

    for (let i = 0; i < stepCount; i++) {
      if (this.pendingPpoUpdate) {
        this.runPendingPpoUpdate(PPO_MINIBATCHES_PER_SIM_TICK);
      } else {
        this.stepTrainingEnvironments();
        envSteps += TRAIN_ENVS;

        if (this.rolloutStep >= PPO_ROLLOUT_STEPS) {
          this.beginPpoUpdate();
        }
      }

      this.stepShowcase();
    }

    return envSteps;
  }

  public simulateShowcase(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      this.stepShowcase();
    }
  }

  public getState(): TrainerState {
    return {
      boardAgent: this.showcaseAgent,
      rewardHistory: this.rewardHistory,
      episodeCount: this.episodeCount,
      totalSteps: this.totalSteps,
      stepsPerSecond: this.stepsPerSecond,
      bestScore: this.bestScoreValue,
      avgReturn: this.averageReturn(),
      bestReturn:
        this.bestReturnValue === Number.NEGATIVE_INFINITY
          ? 0
          : this.bestReturnValue,
      totalLoss: this.totalLoss,
      policyLoss: this.policyLoss,
      valueLoss: this.valueLoss,
      entropy: this.entropy,
      approxKl: this.approxKl,
      clipFraction: this.clipFraction,
      updates: this.updates,
      rolloutProgress: this.rolloutStep / PPO_ROLLOUT_STEPS,
      network: {
        observation: this.showcaseObservation,
        policy: this.showcasePolicy,
        value: this.showcaseValue,
        action: this.showcaseAction,
      },
    };
  }

  public onGridSizeChanged(): void {
    this.reset();
  }

  public async saveCheckpoint(): Promise<CheckpointStats> {
    const snapshot = this.createCheckpointSnapshot();
    await this.online.saveToIndexedDb(CHECKPOINT_MODEL_KEY);
    await saveCheckpointStats(snapshot);
    return snapshot;
  }

  public async loadCheckpoint(): Promise<CheckpointStats | null> {
    const snapshot = await loadCheckpointStats();
    if (!snapshot) {
      return null;
    }

    if (snapshot.version !== CHECKPOINT_VERSION) {
      console.warn(
        `Skipping checkpoint load due to version mismatch: ${snapshot.version}.`,
      );
      return null;
    }

    if (snapshot.gridSize !== GRID_SIZE) {
      console.warn(
        `Skipping checkpoint load due to grid mismatch: saved ${snapshot.gridSize}, current ${GRID_SIZE}.`,
      );
      return null;
    }

    const modelLoaded = await this.online.loadFromIndexedDb(CHECKPOINT_MODEL_KEY);
    if (!modelLoaded) {
      return null;
    }

    this.applyCheckpointSnapshot(snapshot);
    this.refreshShowcasePrediction();
    return snapshot;
  }

  public setStepsPerSecond(value: number): void {
    this.stepsPerSecond = value;
  }

  private stepTrainingEnvironments(): void {
    const rolloutOffset = this.rolloutStep * TRAIN_ENVS;

    for (let i = 0; i < TRAIN_ENVS; i++) {
      const agent = this.trainingAgents[i];
      const state = this.trainingStates[i];

      const decision = this.online.act(state, true);
      const result = this.environment.step(agent, decision.action);
      agent.episodeReturn += result.reward;

      const writeIndex = rolloutOffset + i;
      this.rolloutStates[writeIndex] = state;
      this.rolloutActions[writeIndex] = decision.action;
      this.rolloutRewards[writeIndex] = result.reward;
      this.rolloutDones[writeIndex] = result.done ? 1 : 0;
      this.rolloutValues[writeIndex] = decision.value;
      this.rolloutLogProbs[writeIndex] = decision.logProb;
      this.totalSteps += 1;

      if (result.done) {
        this.recordEpisode(agent.episodeReturn, agent.score);
        this.episodeCount += 1;
        this.environment.resetAgent(agent);
      }

      this.trainingStates[i] = this.environment.observe(agent);
    }

    this.rolloutStep += 1;
  }

  private beginPpoUpdate(): void {
    const batchSize = this.rolloutStep * TRAIN_ENVS;
    if (batchSize <= 0) {
      return;
    }

    const lastValues = new Float32Array(TRAIN_ENVS);
    for (let i = 0; i < TRAIN_ENVS; i++) {
      lastValues[i] = this.online.act(this.trainingStates[i], false).value;
    }

    this.computeGeneralizedAdvantages(lastValues);
    this.normalizeAdvantages(batchSize);

    const indices = new Int32Array(batchSize);
    for (let i = 0; i < batchSize; i++) {
      indices[i] = i;
    }
    shuffleIndices(indices);

    this.pendingPpoUpdate = {
      batchSize,
      indices,
      epoch: 0,
      cursor: 0,
      accumTotalLoss: 0,
      accumPolicyLoss: 0,
      accumValueLoss: 0,
      accumEntropy: 0,
      accumApproxKl: 0,
      accumClipFraction: 0,
      batches: 0,
    };
  }

  private runPendingPpoUpdate(maxMiniBatches: number): void {
    let processed = 0;

    while (processed < maxMiniBatches && this.pendingPpoUpdate) {
      const update = this.pendingPpoUpdate;
      const start = update.cursor;
      const end = Math.min(update.batchSize, start + PPO_MINIBATCH_SIZE);

      if (end <= start) {
        if (update.epoch >= PPO_EPOCHS - 1) {
          this.finishPpoUpdate();
          break;
        }

        update.epoch += 1;
        update.cursor = 0;
        shuffleIndices(update.indices);
        continue;
      }

      const miniBatchSize = end - start;

      const miniStates = new Array<ArrayLike<number>>(miniBatchSize);
      const miniActions = new Int32Array(miniBatchSize);
      const miniOldLogProbs = new Float32Array(miniBatchSize);
      const miniAdvantages = new Float32Array(miniBatchSize);
      const miniReturns = new Float32Array(miniBatchSize);

      for (let i = 0; i < miniBatchSize; i++) {
        const sourceIndex = update.indices[start + i];
        miniStates[i] = this.rolloutStates[sourceIndex];
        miniActions[i] = this.rolloutActions[sourceIndex];
        miniOldLogProbs[i] = this.rolloutLogProbs[sourceIndex];
        miniAdvantages[i] = this.rolloutAdvantages[sourceIndex];
        miniReturns[i] = this.rolloutReturns[sourceIndex];
      }

      const result = this.online.trainBatch({
        observations: miniStates,
        actions: miniActions,
        oldLogProbs: miniOldLogProbs,
        advantages: miniAdvantages,
        returns: miniReturns,
        clipRange: PPO_CLIP_RANGE,
        valueLossCoef: PPO_VALUE_LOSS_COEF,
        entropyCoef: PPO_ENTROPY_COEF,
      });

      update.accumTotalLoss += result.totalLoss;
      update.accumPolicyLoss += result.policyLoss;
      update.accumValueLoss += result.valueLoss;
      update.accumEntropy += result.entropy;
      update.accumApproxKl += result.approxKl;
      update.accumClipFraction += result.clipFraction;
      update.batches += 1;
      update.cursor = end;
      processed += 1;

      if (result.approxKl > PPO_TARGET_KL) {
        this.finishPpoUpdate();
        break;
      }

      if (update.cursor >= update.batchSize) {
        if (update.epoch >= PPO_EPOCHS - 1) {
          this.finishPpoUpdate();
          break;
        }

        update.epoch += 1;
        update.cursor = 0;
        shuffleIndices(update.indices);
      }
    }
  }

  private finishPpoUpdate(): void {
    if (!this.pendingPpoUpdate) {
      return;
    }

    if (this.pendingPpoUpdate.batches > 0) {
      const batchCount = this.pendingPpoUpdate.batches;
      this.totalLoss = ema(
        this.totalLoss,
        this.pendingPpoUpdate.accumTotalLoss / batchCount,
      );
      this.policyLoss = ema(
        this.policyLoss,
        this.pendingPpoUpdate.accumPolicyLoss / batchCount,
      );
      this.valueLoss = ema(
        this.valueLoss,
        this.pendingPpoUpdate.accumValueLoss / batchCount,
      );
      this.entropy = ema(
        this.entropy,
        this.pendingPpoUpdate.accumEntropy / batchCount,
      );
      this.approxKl = ema(
        this.approxKl,
        this.pendingPpoUpdate.accumApproxKl / batchCount,
      );
      this.clipFraction = ema(
        this.clipFraction,
        this.pendingPpoUpdate.accumClipFraction / batchCount,
      );
      this.updates += 1;
    }

    this.pendingPpoUpdate = null;
    this.rolloutStep = 0;
  }

  private computeGeneralizedAdvantages(lastValues: Float32Array): void {
    for (let env = 0; env < TRAIN_ENVS; env++) {
      let gae = 0;
      let nextValue = lastValues[env];

      for (let step = this.rolloutStep - 1; step >= 0; step--) {
        const index = step * TRAIN_ENVS + env;
        const nonTerminal = this.rolloutDones[index] === 1 ? 0 : 1;

        const delta =
          this.rolloutRewards[index] +
          GAMMA * nextValue * nonTerminal -
          this.rolloutValues[index];

        gae = delta + GAMMA * GAE_LAMBDA * nonTerminal * gae;

        this.rolloutAdvantages[index] = gae;
        this.rolloutReturns[index] = gae + this.rolloutValues[index];
        nextValue = this.rolloutValues[index];
      }
    }
  }

  private normalizeAdvantages(batchSize: number): void {
    let sum = 0;
    for (let i = 0; i < batchSize; i++) {
      sum += this.rolloutAdvantages[i];
    }

    const mean = sum / batchSize;
    let variance = 0;
    for (let i = 0; i < batchSize; i++) {
      const centered = this.rolloutAdvantages[i] - mean;
      variance += centered * centered;
    }

    const std = Math.sqrt(variance / batchSize + 1e-8);
    for (let i = 0; i < batchSize; i++) {
      this.rolloutAdvantages[i] = (this.rolloutAdvantages[i] - mean) / std;
    }
  }

  private stepShowcase(): void {
    if (!this.showcaseAgent.alive) {
      this.environment.resetAgent(this.showcaseAgent);
    }

    this.refreshShowcasePrediction();
    this.environment.step(this.showcaseAgent, this.showcaseAction);

    if (!this.showcaseAgent.alive) {
      this.environment.resetAgent(this.showcaseAgent);
    }

    this.refreshShowcasePrediction();
  }

  private refreshShowcasePrediction(): void {
    this.environment.observe(this.showcaseAgent, this.showcaseObservation);
    const prediction = this.online.predict(
      this.showcaseObservation,
      this.showcasePolicy,
      this.showcaseLogits,
    );
    this.showcaseValue = prediction.value;
    this.showcaseAction = argMax(this.showcasePolicy);
  }

  private recordEpisode(episodeReturn: number, episodeScore: number): void {
    this.rewardHistory.push(episodeReturn);
    if (this.rewardHistory.length > HISTORY_LIMIT) {
      this.rewardHistory.shift();
    }

    if (this.bestReturnValue < episodeReturn) {
      this.bestReturnValue = episodeReturn;
    }

    if (this.bestScoreValue < episodeScore) {
      this.bestScoreValue = episodeScore;
    }
  }

  private createCheckpointSnapshot(): CheckpointStats {
    return {
      version: CHECKPOINT_VERSION,
      savedAtMs: Date.now(),
      gridSize: GRID_SIZE,
      episodeCount: this.episodeCount,
      totalSteps: this.totalSteps,
      bestScore: this.bestScoreValue,
      bestReturn:
        this.bestReturnValue === Number.NEGATIVE_INFINITY
          ? 0
          : this.bestReturnValue,
      rewardHistory: this.rewardHistory.slice(),
      totalLoss: this.totalLoss,
      policyLoss: this.policyLoss,
      valueLoss: this.valueLoss,
      entropy: this.entropy,
      approxKl: this.approxKl,
      clipFraction: this.clipFraction,
      updates: this.updates,
    };
  }

  private applyCheckpointSnapshot(snapshot: CheckpointStats): void {
    this.rewardHistory = snapshot.rewardHistory.slice();
    this.episodeCount = Math.max(0, snapshot.episodeCount);
    this.totalSteps = Math.max(0, snapshot.totalSteps);
    this.bestScoreValue = Math.max(0, snapshot.bestScore);
    this.bestReturnValue = Number.isFinite(snapshot.bestReturn)
      ? snapshot.bestReturn
      : Number.NEGATIVE_INFINITY;
    this.totalLoss = snapshot.totalLoss;
    this.policyLoss = snapshot.policyLoss;
    this.valueLoss = snapshot.valueLoss;
    this.entropy = snapshot.entropy;
    this.approxKl = snapshot.approxKl;
    this.clipFraction = snapshot.clipFraction;
    this.updates = Math.max(0, snapshot.updates);
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
}
