import { GRID_SIZE, POP_SIZE } from "./config";
import { NeuralNetwork } from "./NeuralNetwork";
import { PPOOptimizer } from "./PPOOptimizer";
import { SnakeEnvironment } from "./SnakeEnvironment";
import type { Agent, Genome, TrainerState } from "./types";
import type { PPOTransition } from "./PPOOptimizer";

export class SnakeTrainer {
  private readonly ppo = new PPOOptimizer();
  private readonly network = new NeuralNetwork();
  private readonly environment = new SnakeEnvironment(this.network);

  private population: Agent[] = [];
  private trajectories: PPOTransition[][] = [];
  private generation = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessGen = 1;
  private fitnessHistory: number[] = [];

  private showcaseGenome: Genome | null = null;
  private showcaseAgent: Agent | null = null;
  private randomBoardAgents: Agent[] = [];
  private randomBoardGeneration = -1;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.ppo.reset();
    this.population = [];
    this.trajectories = [];
    this.generation = 1;
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessGen = 1;
    this.fitnessHistory = [];
    this.showcaseGenome = null;
    this.showcaseAgent = null;
    this.invalidateRandomBoardAgents();

    this.initializePopulation();

    if (this.population.length > 0) {
      this.setShowcaseGenome(this.ppo.getPolicyGenome());
    }
  }

  public simulate(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      let alive = 0;

      for (let index = 0; index < this.population.length; index++) {
        const agent = this.population[index];
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
        this.evolve();
      }

      if (this.showcaseAgent) {
        this.environment.step(this.showcaseAgent);
      }

      if (!this.showcaseAgent?.alive && this.showcaseGenome) {
        this.showcaseAgent = this.environment.createAgent(this.showcaseGenome);
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
    const network = this.showcaseGenome
      ? {
          genome: this.showcaseGenome,
          activations: this.environment.computeNetworkActivations(
            this.showcaseGenome,
            this.showcaseAgent,
          ),
        }
      : { genome: null, activations: null };

    return {
      boardAgent,
      boardAgents: this.getRandomBoardAgents(randomBoardCount),
      fitnessHistory: this.fitnessHistory,
      generation: this.generation,
      alive,
      populationSize: POP_SIZE,
      bestEverScore: this.bestEverScore,
      bestEverFitness: this.bestEverFitness,
      staleGenerations: Math.max(0, this.generation - this.bestFitnessGen),
      network,
    };
  }

  public onGridSizeChanged(): void {
    this.initializePopulation();

    if (this.showcaseGenome) {
      this.showcaseAgent = this.environment.createAgent(this.showcaseGenome);
    } else {
      this.showcaseAgent = null;
    }

    this.invalidateRandomBoardAgents();
  }

  private setShowcaseGenome(genome: Genome): void {
    this.showcaseGenome = new Float32Array(genome);
    this.showcaseAgent = this.environment.createAgent(this.showcaseGenome);
  }

  private initializePopulation(): void {
    const policy = this.ppo.getPolicyGenome();
    this.population = [];
    this.trajectories = [];

    for (let i = 0; i < POP_SIZE; i++) {
      this.population.push(this.environment.createAgent(policy));
      this.trajectories.push([]);
    }
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
    let best = this.population[0];
    for (const agent of this.population) {
      agent.fitness = this.fitness(agent);
      if (agent.fitness > best.fitness) {
        best = agent;
      }
    }

    this.bestEverScore = Math.max(this.bestEverScore, best.score);
    if (this.generation === 1 || best.fitness > this.bestEverFitness) {
      this.bestEverFitness = best.fitness;
      this.bestFitnessGen = this.generation;
      this.setShowcaseGenome(best.genome);
    }

    this.fitnessHistory.push(best.fitness);
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory.shift();
    }

    this.ppo.train(this.trajectories);

    this.initializePopulation();
    this.generation += 1;
    this.invalidateRandomBoardAgents();
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
