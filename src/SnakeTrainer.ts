import {
  DQN_EPSILON_DECAY_STEPS,
  DQN_EPSILON_END,
  DQN_EPSILON_START,
  DQN_TRAINING_STEPS_PER_UPDATE,
  GRID_SIZE,
  ROLLOUT_BATCH_SIZE,
} from "./config";
import { DQNOptimizer } from "./DQNOptimizer";
import { NeuralNetwork } from "./NeuralNetwork";
import { SnakeEnvironment } from "./SnakeEnvironment";
import type { Agent, PolicyParams, TrainerState } from "./types";

export class SnakeTrainer {
  private readonly dqn = new DQNOptimizer();
  private readonly network = new NeuralNetwork();
  private readonly environment = new SnakeEnvironment(this.network);

  private rolloutAgents: Agent[] = [];
  private dqnUpdate = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessUpdate = 1;
  private fitnessHistory: number[] = [];
  private epsilon = DQN_EPSILON_START;
  private replaySize = 0;
  private lastTrainingLoss = 0;
  private totalEnvSteps = 0;

  private showcasePolicy: PolicyParams | null = null;
  private showcaseAgent: Agent | null = null;
  private randomBoardAgents: Agent[] = [];
  private randomBoardDqnUpdate = -1;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.dqn.reset();
    this.rolloutAgents = [];
    this.dqnUpdate = 1;
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessUpdate = 1;
    this.fitnessHistory = [];
    this.epsilon = DQN_EPSILON_START;
    this.replaySize = 0;
    this.lastTrainingLoss = 0;
    this.totalEnvSteps = 0;
    this.showcasePolicy = null;
    this.showcaseAgent = null;
    this.invalidateRandomBoardAgents();

    this.initializeRolloutBatch();

    if (this.rolloutAgents.length > 0) {
      this.setShowcasePolicy(this.dqn.getPolicyParams());
    }
  }

  public simulate(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      this.epsilon = this.explorationRate();
      let alive = 0;

      for (const agent of this.rolloutAgents) {
        if (!agent.alive) {
          continue;
        }

        const observation = this.environment.observe(agent);
        const sample = this.dqn.sampleAction(observation, this.epsilon);
        const previousScore = agent.score;
        this.environment.step(agent, sample.action);
        const done = !agent.alive;
        const nextObservation = done
          ? observation
          : this.environment.observe(agent);

        this.dqn.addTransition({
          observation,
          action: sample.action,
          reward: this.computeReward(agent, previousScore),
          nextObservation,
          done,
        });
        this.totalEnvSteps += 1;

        if (agent.alive) {
          alive += 1;
        }
      }

      if (alive === 0) {
        this.runDqnUpdate();
      }

      if (this.showcaseAgent) {
        this.environment.step(this.showcaseAgent);
      }

      if (!this.showcaseAgent?.alive && this.showcasePolicy) {
        this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
      }
    }

    this.replaySize = this.dqn.getReplaySize();
  }

  public getState(randomBoardCount = 0): TrainerState {
    let alive = 0;
    for (const agent of this.rolloutAgents) {
      if (agent.alive) {
        alive += 1;
      }
    }

    const boardAgent = this.showcaseAgent ?? this.rolloutAgents[0];
    const network = this.showcasePolicy
      ? {
          policy: this.showcasePolicy,
          activations: this.environment.computeNetworkActivations(
            this.showcasePolicy,
            this.showcaseAgent,
          ),
        }
      : { policy: null, activations: null };

    return {
      boardAgent,
      boardAgents: this.getRandomBoardAgents(randomBoardCount),
      fitnessHistory: this.fitnessHistory,
      dqnUpdate: this.dqnUpdate,
      alive,
      rolloutBatchSize: ROLLOUT_BATCH_SIZE,
      bestEverScore: this.bestEverScore,
      bestEverFitness: this.bestEverFitness,
      updatesSinceBest: Math.max(0, this.dqnUpdate - this.bestFitnessUpdate),
      epsilon: this.epsilon,
      replaySize: this.replaySize,
      loss: this.lastTrainingLoss,
      network,
    };
  }

  public onGridSizeChanged(): void {
    this.initializeRolloutBatch();

    if (this.showcasePolicy) {
      this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
    } else {
      this.showcaseAgent = null;
    }

    this.invalidateRandomBoardAgents();
  }

  private setShowcasePolicy(policy: PolicyParams): void {
    this.showcasePolicy = new Float32Array(policy);
    this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
  }

  private initializeRolloutBatch(): void {
    const policy = this.dqn.getPolicyParams();
    this.rolloutAgents = [];

    for (let i = 0; i < ROLLOUT_BATCH_SIZE; i++) {
      this.rolloutAgents.push(this.environment.createAgent(policy));
    }
  }

  private invalidateRandomBoardAgents(): void {
    this.randomBoardAgents = [];
    this.randomBoardDqnUpdate = -1;
  }

  private sampleAgents(candidates: Agent[], sampleCount: number): Agent[] {
    const count = Math.min(sampleCount, candidates.length);
    const pool = candidates.slice();
    const sampled: Agent[] = [];

    for (let i = 0; i < count; i++) {
      const pick = i + Math.floor(Math.random() * (pool.length - i));
      [pool[i], pool[pick]] = [pool[pick], pool[i]];
      sampled.push(pool[i]);
    }

    return sampled;
  }

  private getRandomBoardAgents(randomBoardCount: number): readonly Agent[] {
    if (randomBoardCount <= 0) {
      return [];
    }

    const requiredCount = Math.min(randomBoardCount, this.rolloutAgents.length);
    const updateChanged = this.randomBoardDqnUpdate !== this.dqnUpdate;
    const sizeChanged = this.randomBoardAgents.length !== requiredCount;

    if (updateChanged || sizeChanged) {
      const aliveFirst = this.sampleAgents(
        this.rolloutAgents.filter((agent) => agent.alive),
        requiredCount,
      );
      const remaining = requiredCount - aliveFirst.length;
      if (remaining > 0) {
        const aliveSet = new Set(aliveFirst);
        const fallback = this.sampleAgents(
          this.rolloutAgents.filter((agent) => !aliveSet.has(agent)),
          remaining,
        );
        this.randomBoardAgents = aliveFirst.concat(fallback);
      } else {
        this.randomBoardAgents = aliveFirst;
      }
      this.randomBoardDqnUpdate = this.dqnUpdate;
    }

    const currentlyAlive = new Set(
      this.randomBoardAgents.filter((agent) => agent.alive),
    );
    const replacementPool = this.rolloutAgents.filter(
      (agent) => agent.alive && !currentlyAlive.has(agent),
    );

    for (let i = 0; i < this.randomBoardAgents.length; i++) {
      const current = this.randomBoardAgents[i];
      if (current.alive || replacementPool.length === 0) {
        continue;
      }

      const pick = Math.floor(Math.random() * replacementPool.length);
      const replacement = replacementPool[pick];
      this.randomBoardAgents[i] = replacement;
      currentlyAlive.add(replacement);
      replacementPool[pick] = replacementPool[replacementPool.length - 1];
      replacementPool.pop();
    }

    return this.randomBoardAgents;
  }

  private runDqnUpdate(): void {
    let best = this.rolloutAgents[0];
    for (const agent of this.rolloutAgents) {
      agent.fitness = this.fitness(agent);
      if (agent.fitness > best.fitness) {
        best = agent;
      }
    }

    this.bestEverScore = Math.max(this.bestEverScore, best.score);
    if (this.dqnUpdate === 1 || best.fitness > this.bestEverFitness) {
      this.bestEverFitness = best.fitness;
      this.bestFitnessUpdate = this.dqnUpdate;
      this.setShowcasePolicy(best.policy);
    }

    this.fitnessHistory.push(best.fitness);
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory.shift();
    }

    const stats = this.dqn.train(DQN_TRAINING_STEPS_PER_UPDATE);
    this.lastTrainingLoss = stats.loss;
    this.replaySize = stats.replaySize;

    this.initializeRolloutBatch();
    this.dqnUpdate += 1;
    this.invalidateRandomBoardAgents();
  }

  private explorationRate(): number {
    const denominator = Math.max(1, DQN_EPSILON_DECAY_STEPS);
    const progress = Math.min(1, Math.max(0, this.totalEnvSteps / denominator));
    return DQN_EPSILON_START + (DQN_EPSILON_END - DQN_EPSILON_START) * progress;
  }

  private computeReward(agent: Agent, previousScore: number): number {
    let reward = agent.score - previousScore;
    reward -= 1 / (GRID_SIZE * GRID_SIZE);
    if (!agent.alive && agent.hunger > 0) {
      reward -= 1;
    }
    return reward;
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = !agent.alive && agent.hunger > 0 ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }
}
