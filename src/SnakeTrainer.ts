import {
  ES_EVAL_ENVS,
  ES_EVAL_STEPS,
  ES_GENERATION_STEPS,
  ES_LEARNING_RATE,
  ES_NOISE_STD,
  ES_POPULATION_SIZE,
  GRID_SIZE,
  OUTPUTS,
  REWARD_HISTORY_LIMIT,
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

const AVG_WINDOW = 100;
const METRIC_EMA = 0.1;
const CHECKPOINT_VERSION = 2;

function ema(previous: number, next: number): number {
  if (!Number.isFinite(next)) {
    return previous;
  }
  return previous === 0 ? next : previous * (1 - METRIC_EMA) + next * METRIC_EMA;
}

function fillRandomNormal(target: Float32Array): void {
  for (let i = 0; i < target.length; i += 2) {
    const u1 = Math.max(1e-7, Math.random());
    const u2 = Math.random();
    const magnitude = Math.sqrt(-2 * Math.log(u1));
    const angle = 2 * Math.PI * u2;

    target[i] = magnitude * Math.cos(angle);
    if (i + 1 < target.length) {
      target[i + 1] = magnitude * Math.sin(angle);
    }
  }
}

function l2Norm(values: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i] * values[i];
  }
  return Math.sqrt(sum);
}

export class SnakeTrainer {
  private readonly environment = new SnakeEnvironment();
  private online = new ConvDQN();

  private trainingAgents: Agent[] = [];
  private trainingStates: Float32Array[] = [];
  private trainingActions = new Int32Array(0);
  private generationStep = 0;

  private showcaseAgent: Agent = this.environment.createAgent();
  private showcaseObservation: Float32Array = new Float32Array(observationSize());
  private showcasePolicy = new Float32Array(OUTPUTS);
  private showcaseLogits = new Float32Array(OUTPUTS);
  private showcaseAction = 0;

  private baseWeights = new Float32Array(0);
  private candidateWeights = new Float32Array(0);
  private noiseScratch = new Float32Array(0);
  private gradientScratch = new Float32Array(0);

  private rewardHistory: number[] = [];
  private episodeCount = 0;
  private totalSteps = 0;
  private stepsPerSecond = 0;
  private bestScoreValue = 0;
  private bestReturnValue = Number.NEGATIVE_INFINITY;

  private fitnessMean = 0;
  private fitnessStd = 0;
  private fitnessBest = Number.NEGATIVE_INFINITY;
  private updateNorm = 0;
  private weightNorm = 0;
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

    this.trainingActions = new Int32Array(TRAIN_ENVS);
    this.generationStep = 0;

    const parameterCount = this.online.parameterCount();
    this.baseWeights = new Float32Array(parameterCount);
    this.candidateWeights = new Float32Array(parameterCount);
    this.noiseScratch = new Float32Array(parameterCount);
    this.gradientScratch = new Float32Array(parameterCount);
    this.online.exportWeightsFlat(this.baseWeights);

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

    this.fitnessMean = 0;
    this.fitnessStd = 0;
    this.fitnessBest = Number.NEGATIVE_INFINITY;
    this.updateNorm = 0;
    this.weightNorm = l2Norm(this.baseWeights);
    this.updates = 0;
  }

  public simulate(stepCount: number): number {
    let envSteps = 0;

    for (let i = 0; i < stepCount; i++) {
      this.stepTrainingEnvironments();
      envSteps += TRAIN_ENVS;

      if (this.generationStep >= ES_GENERATION_STEPS) {
        this.runEvolutionUpdate();
        this.generationStep = 0;
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
      fitnessMean: this.fitnessMean,
      fitnessStd: this.fitnessStd,
      fitnessBest:
        this.fitnessBest === Number.NEGATIVE_INFINITY ? 0 : this.fitnessBest,
      updateNorm: this.updateNorm,
      weightNorm: this.weightNorm,
      updates: this.updates,
      generationProgress: this.generationStep / ES_GENERATION_STEPS,
      network: {
        observation: this.showcaseObservation,
        policy: this.showcasePolicy,
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

    this.online.exportWeightsFlat(this.baseWeights);
    this.applyCheckpointSnapshot(snapshot);
    this.refreshShowcasePrediction();
    return snapshot;
  }

  public setStepsPerSecond(value: number): void {
    this.stepsPerSecond = value;
  }

  private stepTrainingEnvironments(): void {
    this.online.actBatch(this.trainingStates, true, this.trainingActions);

    for (let i = 0; i < TRAIN_ENVS; i++) {
      const agent = this.trainingAgents[i];
      const action = this.trainingActions[i];
      const result = this.environment.step(agent, action);
      agent.episodeReturn += result.reward;
      this.totalSteps += 1;

      if (result.done) {
        this.recordEpisode(agent.episodeReturn, agent.score);
        this.episodeCount += 1;
        this.environment.resetAgent(agent);
      }

      this.environment.observe(agent, this.trainingStates[i]);
    }

    this.generationStep += 1;
  }

  private runEvolutionUpdate(): void {
    if (ES_NOISE_STD <= 0 || ES_EVAL_STEPS <= 0 || ES_EVAL_ENVS <= 0) {
      return;
    }

    const populationSize =
      ES_POPULATION_SIZE >= 2
        ? ES_POPULATION_SIZE + (ES_POPULATION_SIZE % 2)
        : 2;
    const pairCount = Math.max(1, populationSize / 2);

    this.online.exportWeightsFlat(this.baseWeights);
    this.gradientScratch.fill(0);

    const referenceAgents = this.captureEvaluationAgents();
    const rewards = new Float32Array(populationSize);
    let rewardCursor = 0;
    let bestCandidate = Number.NEGATIVE_INFINITY;

    for (let pair = 0; pair < pairCount; pair++) {
      fillRandomNormal(this.noiseScratch);

      this.online.addScaledNoise(
        this.baseWeights,
        this.noiseScratch,
        ES_NOISE_STD,
        this.candidateWeights,
      );
      const rewardPlus = this.evaluateCandidate(referenceAgents, this.candidateWeights);
      rewards[rewardCursor] = rewardPlus;
      rewardCursor += 1;

      this.online.addScaledNoise(
        this.baseWeights,
        this.noiseScratch,
        -ES_NOISE_STD,
        this.candidateWeights,
      );
      const rewardMinus = this.evaluateCandidate(referenceAgents, this.candidateWeights);
      rewards[rewardCursor] = rewardMinus;
      rewardCursor += 1;

      if (rewardPlus > bestCandidate) {
        bestCandidate = rewardPlus;
      }
      if (rewardMinus > bestCandidate) {
        bestCandidate = rewardMinus;
      }

      const rewardDiff = rewardPlus - rewardMinus;
      for (let i = 0; i < this.gradientScratch.length; i++) {
        this.gradientScratch[i] += rewardDiff * this.noiseScratch[i];
      }
    }

    const updateScale = ES_LEARNING_RATE / (2 * pairCount * ES_NOISE_STD);
    let updateSumSquares = 0;

    for (let i = 0; i < this.baseWeights.length; i++) {
      const delta = updateScale * this.gradientScratch[i];
      this.baseWeights[i] += delta;
      updateSumSquares += delta * delta;
    }

    this.online.importWeightsFlat(this.baseWeights);

    let rewardSum = 0;
    for (let i = 0; i < rewards.length; i++) {
      rewardSum += rewards[i];
    }
    const rewardMean = rewardSum / rewards.length;

    let rewardVariance = 0;
    for (let i = 0; i < rewards.length; i++) {
      const centered = rewards[i] - rewardMean;
      rewardVariance += centered * centered;
    }
    const rewardStd = Math.sqrt(rewardVariance / rewards.length);

    this.fitnessMean = ema(this.fitnessMean, rewardMean);
    this.fitnessStd = ema(this.fitnessStd, rewardStd);
    if (bestCandidate > this.fitnessBest) {
      this.fitnessBest = bestCandidate;
    }
    this.updateNorm = ema(this.updateNorm, Math.sqrt(updateSumSquares));
    this.weightNorm = ema(this.weightNorm, l2Norm(this.baseWeights));
    this.updates += 1;
  }

  private captureEvaluationAgents(): Agent[] {
    const count = Math.max(1, Math.min(ES_EVAL_ENVS, this.trainingAgents.length));
    const agents = new Array<Agent>(count);

    for (let i = 0; i < count; i++) {
      agents[i] = this.cloneAgent(this.trainingAgents[i]);
    }

    return agents;
  }

  private evaluateCandidate(
    referenceAgents: readonly Agent[],
    candidateWeights: Float32Array,
  ): number {
    this.online.importWeightsFlat(candidateWeights);

    const agents = new Array<Agent>(referenceAgents.length);
    const observations = new Array<Float32Array>(referenceAgents.length);
    const actions = new Int32Array(referenceAgents.length);

    for (let i = 0; i < referenceAgents.length; i++) {
      agents[i] = this.cloneAgent(referenceAgents[i]);
      observations[i] = this.environment.observe(agents[i]);
    }

    let rewardSum = 0;

    for (let step = 0; step < ES_EVAL_STEPS; step++) {
      this.online.actBatch(observations, false, actions);

      for (let env = 0; env < agents.length; env++) {
        const result = this.environment.step(agents[env], actions[env]);
        rewardSum += result.reward;

        if (result.done) {
          this.environment.resetAgent(agents[env]);
        }

        this.environment.observe(agents[env], observations[env]);
      }
    }

    return rewardSum / (ES_EVAL_STEPS * agents.length);
  }

  private cloneAgent(agent: Agent): Agent {
    return {
      body: agent.body.map((part) => ({ x: part.x, y: part.y })),
      dir: agent.dir,
      food: { x: agent.food.x, y: agent.food.y },
      alive: agent.alive,
      score: agent.score,
      steps: agent.steps,
      hunger: agent.hunger,
      episodeReturn: agent.episodeReturn,
    };
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
    this.online.predict(
      this.showcaseObservation,
      this.showcasePolicy,
      this.showcaseLogits,
    );
    this.showcaseAction = argMax(this.showcasePolicy);
  }

  private recordEpisode(episodeReturn: number, episodeScore: number): void {
    this.rewardHistory.push(episodeReturn);
    if (this.rewardHistory.length > REWARD_HISTORY_LIMIT + 512) {
      this.rewardHistory.splice(
        0,
        this.rewardHistory.length - REWARD_HISTORY_LIMIT,
      );
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
      fitnessMean: this.fitnessMean,
      fitnessStd: this.fitnessStd,
      fitnessBest:
        this.fitnessBest === Number.NEGATIVE_INFINITY ? 0 : this.fitnessBest,
      updateNorm: this.updateNorm,
      weightNorm: this.weightNorm,
      updates: this.updates,
    };
  }

  private applyCheckpointSnapshot(snapshot: CheckpointStats): void {
    this.rewardHistory = snapshot.rewardHistory.slice(-REWARD_HISTORY_LIMIT);
    this.episodeCount = Math.max(0, snapshot.episodeCount);
    this.totalSteps = Math.max(0, snapshot.totalSteps);
    this.bestScoreValue = Math.max(0, snapshot.bestScore);
    this.bestReturnValue = Number.isFinite(snapshot.bestReturn)
      ? snapshot.bestReturn
      : Number.NEGATIVE_INFINITY;

    this.fitnessMean = snapshot.fitnessMean;
    this.fitnessStd = snapshot.fitnessStd;
    this.fitnessBest = Number.isFinite(snapshot.fitnessBest)
      ? snapshot.fitnessBest
      : Number.NEGATIVE_INFINITY;
    this.updateNorm = snapshot.updateNorm;
    this.weightNorm = snapshot.weightNorm;
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
