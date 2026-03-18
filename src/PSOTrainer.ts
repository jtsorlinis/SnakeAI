import { POP_SIZE } from "./config";
import { NeuralNetwork } from "./NeuralNetwork";
import { ParticleSwarmOptimizer } from "./ParticleSwarmOptimizer";
import { SnakeEnvironment } from "./SnakeEnvironment";
import type {
  Agent,
  PolicyParams,
  PolicyPlaybackMode,
  TrainerController,
  TrainerState,
} from "./types";

export class PSOTrainer implements TrainerController {
  private readonly pso = new ParticleSwarmOptimizer();
  private readonly network = new NeuralNetwork();
  private readonly environment = new SnakeEnvironment(this.network);

  private population: Agent[] = [];
  private generation = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessGeneration = 1;
  private fitnessHistory: number[] = [];

  private showcasePolicy: PolicyParams | null = null;
  private showcaseAgent: Agent | null = null;
  private randomBoardAgents: Agent[] = [];
  private randomBoardGeneration = -1;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.pso.reset();
    this.population = [];
    this.generation = 1;
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessGeneration = 1;
    this.fitnessHistory = [];
    this.showcasePolicy = null;
    this.showcaseAgent = null;
    this.invalidateRandomBoardAgents();

    this.population = this.pso
      .initializePopulation()
      .map((policy) => this.environment.createAgent(policy));

    if (this.population.length > 0) {
      this.setShowcasePolicy(this.population[0].policy);
    }
  }

  public simulate(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      let alive = 0;

      for (const agent of this.population) {
        if (!agent.alive) {
          continue;
        }

        this.environment.step(agent);
        if (agent.alive) {
          alive += 1;
        }
      }

      if (alive === 0) {
        this.evolve();
      }

      if (this.showcaseAgent) {
        this.environment.step(this.showcaseAgent);
      }

      if (!this.showcaseAgent?.alive && this.showcasePolicy) {
        this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
      }
    }
  }

  public getState(randomBoardCount = 0): TrainerState {
    let alive = 0;
    for (const agent of this.population) {
      if (agent.alive) {
        alive += 1;
      }
    }

    const boardAgent = this.showcaseAgent ?? this.population[0];
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
      algorithm: "pso",
      boardAgent,
      boardAgents: this.getRandomBoardAgents(randomBoardCount),
      fitnessHistory: this.fitnessHistory,
      iteration: this.generation,
      iterationLabel: "Generation",
      alive,
      batchSize: POP_SIZE,
      batchSizeLabel: "Swarm",
      bestEverScore: this.bestEverScore,
      bestEverFitness: this.bestEverFitness,
      staleIterations: Math.max(0, this.generation - this.bestFitnessGeneration),
      staleLabel: "Generations since best",
      historyLabel: "PSO best fitness history",
      policySourceLabel: "Best policy",
      playbackMode: "greedy",
      playbackModeEnabled: false,
      network,
    };
  }

  public onGridSizeChanged(): void {
    this.population = this.population.map((agent) =>
      this.environment.createAgent(agent.policy),
    );

    if (this.showcasePolicy) {
      this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
    } else {
      this.showcaseAgent = null;
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

  private setShowcasePolicy(policy: PolicyParams): void {
    this.showcasePolicy = new Float32Array(policy);
    this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
  }

  private invalidateRandomBoardAgents(): void {
    this.randomBoardAgents = [];
    this.randomBoardGeneration = -1;
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

    const requiredCount = Math.min(randomBoardCount, this.population.length);
    const generationChanged = this.randomBoardGeneration !== this.generation;
    const sizeChanged = this.randomBoardAgents.length !== requiredCount;

    if (generationChanged || sizeChanged) {
      const aliveFirst = this.sampleAgents(
        this.population.filter((agent) => agent.alive),
        requiredCount,
      );
      const remaining = requiredCount - aliveFirst.length;
      if (remaining > 0) {
        const aliveSet = new Set(aliveFirst);
        const fallback = this.sampleAgents(
          this.population.filter((agent) => !aliveSet.has(agent)),
          remaining,
        );
        this.randomBoardAgents = aliveFirst.concat(fallback);
      } else {
        this.randomBoardAgents = aliveFirst;
      }
      this.randomBoardGeneration = this.generation;
    }

    const currentlyAlive = new Set(
      this.randomBoardAgents.filter((agent) => agent.alive),
    );
    const replacementPool = this.population.filter(
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

  private evolve(): void {
    const { best, nextPolicies } = this.pso.evolve(this.population);

    this.bestEverScore = Math.max(this.bestEverScore, best.score);
    if (this.generation === 1 || best.fitness > this.bestEverFitness) {
      this.bestEverFitness = best.fitness;
      this.bestFitnessGeneration = this.generation;
      this.setShowcasePolicy(best.policy);
    }

    this.fitnessHistory.push(best.fitness);
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory.shift();
    }

    this.population = nextPolicies.map((policy) =>
      this.environment.createAgent(policy),
    );
    this.generation += 1;
    this.invalidateRandomBoardAgents();
  }
}
