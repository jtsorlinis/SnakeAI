import {
  PPO_RESTART_AFTER,
  PPO_ROLLOUT_HORIZON,
  PPO_SETTLE_CHECK_AFTER,
  PPO_SETTLED_ENTROPY,
  PPO_VALIDATE_EVERY,
  RESTART_CHECK_GAMES,
  ROLLOUT_BATCH_SIZE,
  VALIDATION_GAMES,
} from "./config";
import { NeuralNetwork } from "./NeuralNetwork";
import { scoreFitness, validatePolicy } from "./PolicyValidation";
import { PPOOptimizer } from "./PPOOptimizer";
import { ReplayEvaluator } from "./ReplayEvaluator";
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
  private readonly replay = new ReplayEvaluator(this.environment);

  private rolloutAgents: Agent[] = [];
  private trajectories: PPOTransition[][] = [];
  private rolloutSteps = 0;
  private ppoUpdate = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessUpdate = 1;
  // When the current networks were initialised, and whether they have passed
  // the board-clearing check.
  private runStartUpdate = 1;
  private runClearsBoard = false;
  private policyEntropy = Number.POSITIVE_INFINITY;
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
    this.runStartUpdate = 1;
    this.runClearsBoard = false;
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

      this.replay.step();
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
      replay: this.replay.getStats(),
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
    this.runStartUpdate = this.ppoUpdate;
    this.runClearsBoard = false;

    if (this.displayPolicy) {
      this.displayAgent = this.environment.createAgent(this.displayPolicy);
    } else {
      this.displayAgent = null;
    }
    this.replay.setPolicy(this.displayPolicy);

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
    this.replay.setPolicy(this.displayPolicy);
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
    agent.fitness = scoreFitness(agent);
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
      agent.fitness = scoreFitness(agent);
      bestScore = Math.max(bestScore, agent.score);
      bestFitness = Math.max(bestFitness, agent.fitness);
    }

    this.bestEverScore = Math.max(this.bestEverScore, bestScore);
    // Rollout episodes are sampled stochastically and span many updates, so
    // the best one says little about the current policy. Score it greedily on
    // fresh games instead; ties go to the newer policy.
    if (this.ppoUpdate === 1 || this.ppoUpdate % PPO_VALIDATE_EVERY === 0) {
      const policy = this.ppo.getPolicyParams();
      const validated = validatePolicy(this.environment, policy, VALIDATION_GAMES);
      if (validated.fitness >= this.bestEverFitness) {
        this.bestEverFitness = validated.fitness;
        this.bestFitnessUpdate = this.ppoUpdate;
        this.setDisplayPolicy(policy);
      }
    }

    this.fitnessHistory.push(bestFitness);
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory.shift();
    }

    if (this.shouldRestart()) {
      this.ppo.reset();
      this.initializeRolloutBatch();
      this.runStartUpdate = this.ppoUpdate;
    } else {
      this.policyEntropy = this.ppo.train(this.buildRollouts()).entropy;

      this.syncRolloutAgentPolicies();
      this.clearTrajectories();
      this.rolloutSteps = 0;
      this.resetCompletedEpisodeStats();
    }
    this.ppoUpdate += 1;
    this.invalidateRandomBoardAgents();
  }

  // Most runs never learn to clear the board. Networks that are still
  // choosing moves too randomly partway in, or that haven't learned it within
  // their budget (judged by one check on fresh games), are reinitialised. The
  // display keeps the best policy found so far.
  private shouldRestart(): boolean {
    if (this.runClearsBoard) {
      return false;
    }

    const age = this.ppoUpdate - this.runStartUpdate;
    if (age === PPO_SETTLE_CHECK_AFTER) {
      return this.policyEntropy > PPO_SETTLED_ENTROPY;
    }
    if (age < PPO_RESTART_AFTER) {
      return false;
    }

    const check = validatePolicy(
      this.environment,
      this.ppo.getPolicyParams(),
      RESTART_CHECK_GAMES,
    );
    this.runClearsBoard = check.clearRate >= 0.5;
    return !this.runClearsBoard;
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

  // No per-step penalty, for the same reason as scoreFitness.
  private computeReward(agent: Agent, previousScore: number): number {
    let reward = agent.score - previousScore;
    if (agent.terminalReason === "solved") {
      reward += 1;
    } else if (agent.terminalReason === "collision") {
      reward -= 1;
    }
    return reward;
  }
}
