import { GRID_SIZE, ROLLOUT_BATCH_SIZE } from "./config";
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
import type { PPOTransition } from "./PPOOptimizer";

export class PPOTrainer implements TrainerController {
  private readonly ppo = new PPOOptimizer();
  private readonly network = new NeuralNetwork();
  private readonly environment = new SnakeEnvironment(this.network);

  private rolloutAgents: Agent[] = [];
  private trajectories: PPOTransition[][] = [];
  private ppoUpdate = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessUpdate = 1;
  private fitnessHistory: number[] = [];

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
    this.ppoUpdate = 1;
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessUpdate = 1;
    this.fitnessHistory = [];
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
      let alive = 0;

      for (let index = 0; index < this.rolloutAgents.length; index++) {
        const agent = this.rolloutAgents[index];
        if (!agent.alive) {
          continue;
        }

        const observation = this.environment.observe(agent);
        const sample = this.ppo.sampleAction(observation);
        const previousScore = agent.score;
        this.environment.step(agent, sample.action);

        this.trajectories[index].push({
          observation,
          action: sample.action,
          reward: this.computeReward(agent, previousScore),
          done: !agent.alive,
          logProb: sample.logProb,
          value: sample.value,
        });

        if (agent.alive) {
          alive += 1;
        }
      }

      if (alive === 0) {
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

    for (let i = 0; i < ROLLOUT_BATCH_SIZE; i++) {
      this.rolloutAgents.push(this.environment.createAgent(policy));
      this.trajectories.push([]);
    }
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
    let best = this.rolloutAgents[0];
    for (const agent of this.rolloutAgents) {
      agent.fitness = this.fitness(agent);
      if (agent.fitness > best.fitness) {
        best = agent;
      }
    }

    this.bestEverScore = Math.max(this.bestEverScore, best.score);
    if (this.ppoUpdate === 1 || best.fitness > this.bestEverFitness) {
      this.bestEverFitness = best.fitness;
      this.bestFitnessUpdate = this.ppoUpdate;
      this.setDisplayPolicy(this.ppo.getPolicyParams());
    }

    this.fitnessHistory.push(best.fitness);
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory.shift();
    }

    this.ppo.train(this.trajectories);

    this.initializeRolloutBatch();
    this.ppoUpdate += 1;
    this.invalidateRandomBoardAgents();
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
