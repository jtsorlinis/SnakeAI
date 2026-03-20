import {
  GRID_SIZE,
  PPO_ROLLOUT_HORIZON,
  ROLLOUT_BATCH_SIZE,
} from "./config";
import { NeuralNetwork } from "./NeuralNetwork";
import { PPOOptimizer } from "./PPOOptimizer";
import { SnakeEnvironment } from "./SnakeEnvironment";
import type {
  Agent,
  PolicyParams,
  PolicyPlaybackMode,
  TrainerController,
  TrainerState,
} from "./types";
import type { PPORollout, PPOTransition } from "./PPOOptimizer";

export class PPOTrainer implements TrainerController {
  private readonly ppo = new PPOOptimizer();
  private readonly network = new NeuralNetwork();
  private readonly environment = new SnakeEnvironment(this.network);

  private rolloutAgents: Agent[] = [];
  private trajectories: PPOTransition[][] = [];
  private rolloutSteps = 0;
  private ppoUpdate = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessUpdate = 1;
  private fitnessHistory: number[] = [];
  private completedEpisodeBestScore = 0;
  private completedEpisodeBestFitness = Number.NEGATIVE_INFINITY;

  private displayPolicy: PolicyParams | null = null;
  private displayAgent: Agent | null = null;
  private randomBoardAgents: Agent[] = [];
  private randomBoardPpoUpdate = -1;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.ppo.reset();
    this.rolloutAgents = [];
    this.trajectories = [];
    this.rolloutSteps = 0;
    this.ppoUpdate = 1;
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessUpdate = 1;
    this.fitnessHistory = [];
    this.resetCompletedEpisodeStats();
    this.displayPolicy = null;
    this.displayAgent = null;
    this.invalidateRandomBoardAgents();

    this.initializeRolloutBatch();

    if (this.rolloutAgents.length > 0) {
      this.setDisplayPolicy(this.ppo.getPolicyParams());
    }
  }

  public simulate(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      for (let index = 0; index < this.rolloutAgents.length; index++) {
        const agent = this.rolloutAgents[index];
        const observation = this.environment.observe(agent);
        const sample = this.ppo.sampleAction(observation);
        const previousScore = agent.score;
        this.environment.step(agent, sample.action);
        const done = !agent.alive;

        this.trajectories[index].push({
          observation,
          action: sample.action,
          reward: this.computeReward(agent, previousScore),
          done,
          logProb: sample.logProb,
          value: sample.value,
        });

        if (done) {
          this.recordCompletedEpisode(agent);
          this.rolloutAgents[index] = this.environment.createAgent(
            this.ppo.getPolicyParams(),
          );
        }
      }

      this.rolloutSteps += 1;
      if (this.rolloutSteps >= PPO_ROLLOUT_HORIZON) {
        this.runPpoUpdate();
      }

      if (this.displayAgent) {
        const action = this.environment.selectAction(
          this.displayAgent,
          "greedy",
        );
        this.environment.step(this.displayAgent, action);
      }

      if (!this.displayAgent?.alive && this.displayPolicy) {
        this.displayAgent = this.environment.createAgent(this.displayPolicy);
      }
    }
  }

  public getState(randomBoardCount = 0): TrainerState {
    let alive = 0;
    for (const agent of this.rolloutAgents) {
      if (agent.alive) {
        alive += 1;
      }
    }

    const boardAgent = this.displayAgent ?? this.rolloutAgents[0];
    const network = this.displayPolicy
      ? {
          policy: this.displayPolicy,
          activations: this.environment.computeNetworkActivations(
            this.displayPolicy,
            this.displayAgent,
          ),
        }
      : { policy: null, activations: null };

    return {
      algorithm: "ppo",
      boardAgent,
      boardAgents: this.getRandomBoardAgents(randomBoardCount),
      fitnessHistory: this.fitnessHistory,
      iteration: this.ppoUpdate,
      iterationLabel: "PPO update",
      alive,
      batchSize: ROLLOUT_BATCH_SIZE,
      batchSizeLabel: "Rollout batch",
      bestEverScore: this.bestEverScore,
      bestEverFitness: this.bestEverFitness,
      staleIterations: Math.max(0, this.ppoUpdate - this.bestFitnessUpdate),
      staleLabel: "Updates since best",
      historyLabel: "PPO best fitness history",
      policySourceLabel: "Best policy",
      playbackMode: "greedy",
      playbackModeEnabled: false,
      network,
    };
  }

  public onGridSizeChanged(): void {
    this.initializeRolloutBatch();

    if (this.displayPolicy) {
      this.displayAgent = this.environment.createAgent(this.displayPolicy);
    } else {
      this.displayAgent = null;
    }

    this.invalidateRandomBoardAgents();
  }

  public setPlaybackMode(_mode: PolicyPlaybackMode): void {}

  public getPlaybackMode(): PolicyPlaybackMode {
    return "greedy";
  }

  public supportsPlaybackMode(): boolean {
    return false;
  }

  private setDisplayPolicy(policy: PolicyParams): void {
    this.displayPolicy = new Float32Array(policy);
    this.displayAgent = this.environment.createAgent(this.displayPolicy);
  }

  private initializeRolloutBatch(): void {
    const policy = this.ppo.getPolicyParams();
    this.rolloutAgents = [];
    this.trajectories = [];
    this.rolloutSteps = 0;
    this.resetCompletedEpisodeStats();

    for (let i = 0; i < ROLLOUT_BATCH_SIZE; i++) {
      this.rolloutAgents.push(this.environment.createAgent(policy));
      this.trajectories.push([]);
    }
  }

  private resetCompletedEpisodeStats(): void {
    this.completedEpisodeBestScore = 0;
    this.completedEpisodeBestFitness = Number.NEGATIVE_INFINITY;
  }

  private recordCompletedEpisode(agent: Agent): void {
    agent.fitness = this.fitness(agent);
    this.completedEpisodeBestScore = Math.max(
      this.completedEpisodeBestScore,
      agent.score,
    );
    this.completedEpisodeBestFitness = Math.max(
      this.completedEpisodeBestFitness,
      agent.fitness,
    );
  }

  private invalidateRandomBoardAgents(): void {
    this.randomBoardAgents = [];
    this.randomBoardPpoUpdate = -1;
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
    const updateChanged = this.randomBoardPpoUpdate !== this.ppoUpdate;
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
      this.randomBoardPpoUpdate = this.ppoUpdate;
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

  private runPpoUpdate(): void {
    let bestScore = this.completedEpisodeBestScore;
    let bestFitness = this.completedEpisodeBestFitness;

    for (const agent of this.rolloutAgents) {
      agent.fitness = this.fitness(agent);
      bestScore = Math.max(bestScore, agent.score);
      bestFitness = Math.max(bestFitness, agent.fitness);
    }

    this.bestEverScore = Math.max(this.bestEverScore, bestScore);
    if (this.ppoUpdate === 1 || bestFitness > this.bestEverFitness) {
      this.bestEverFitness = bestFitness;
      this.bestFitnessUpdate = this.ppoUpdate;
      this.setDisplayPolicy(this.ppo.getPolicyParams());
    }

    this.fitnessHistory.push(bestFitness);
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory.shift();
    }

    this.ppo.train(this.buildRollouts());

    this.syncRolloutAgentPolicies();
    this.clearTrajectories();
    this.rolloutSteps = 0;
    this.resetCompletedEpisodeStats();
    this.ppoUpdate += 1;
    this.invalidateRandomBoardAgents();
  }

  private buildRollouts(): PPORollout[] {
    const rollouts: PPORollout[] = [];

    for (let index = 0; index < this.trajectories.length; index++) {
      const transitions = this.trajectories[index];
      const lastTransition = transitions[transitions.length - 1];
      let bootstrapValue = 0;

      if (lastTransition && !lastTransition.done) {
        const observation = this.environment.observe(this.rolloutAgents[index]);
        bootstrapValue = this.ppo.estimateValue(observation);
      }

      rollouts.push({ transitions, bootstrapValue });
    }

    return rollouts;
  }

  private syncRolloutAgentPolicies(): void {
    const policy = this.ppo.getPolicyParams();
    for (const agent of this.rolloutAgents) {
      agent.policy = new Float32Array(policy);
    }
  }

  private clearTrajectories(): void {
    for (const trajectory of this.trajectories) {
      trajectory.length = 0;
    }
  }

  private computeReward(agent: Agent, previousScore: number): number {
    let reward = agent.score - previousScore;
    reward -= 1 / (GRID_SIZE * GRID_SIZE);
    if (agent.terminalReason === "solved") {
      reward += 1;
    } else if (agent.terminalReason === "collision") {
      reward -= 1;
    }
    return reward;
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = agent.terminalReason === "collision" ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }
}
