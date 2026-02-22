import {
  EPISODES_PER_GENOME_LATE,
  EPISODES_PER_GENOME_START,
  EPISODES_PER_GENOME_SWITCH_GENERATION,
  POP_SIZE,
} from "./config";
import {
  GeneticAlgorithm,
  type EvolutionCandidate,
} from "./GeneticAlgorithm";
import { NeuralNetwork } from "./NeuralNetwork";
import { SnakeEnvironment } from "./SnakeEnvironment";
import type { Agent, Genome, TrainerState } from "./types";

export class SnakeTrainer {
  private readonly ga = new GeneticAlgorithm();
  private readonly network = new NeuralNetwork();
  private readonly environment = new SnakeEnvironment(this.network);

  private population: Agent[] = [];
  private generation = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessGen = 1;
  private fitnessHistory: number[] = [];
  private evaluationEpisode = 1;
  private evaluationEpisodeTarget = EPISODES_PER_GENOME_START;
  private evaluationFitnessSums: number[] = [];
  private evaluationScoreSums: number[] = [];
  private evaluationBestScores: number[] = [];

  private showcaseGenome: Genome | null = null;
  private showcaseAgent: Agent | null = null;
  private randomBoardAgents: Agent[] = [];
  private randomBoardGeneration = -1;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.population = [];
    this.generation = 1;
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessGen = 1;
    this.fitnessHistory = [];
    this.showcaseGenome = null;
    this.showcaseAgent = null;
    this.invalidateRandomBoardAgents();

    for (let i = 0; i < POP_SIZE; i++) {
      this.population.push(
        this.environment.createAgent(this.ga.randomGenome()),
      );
    }

    if (this.population.length > 0) {
      this.setShowcaseGenome(this.population[0].genome);
    }

    this.beginGenerationEvaluation();
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
        this.finishEpisode();
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
      evaluationEpisode: this.evaluationEpisode,
      evaluationEpisodeTarget: this.evaluationEpisodeTarget,
      alive,
      populationSize: POP_SIZE,
      bestEverScore: this.bestEverScore,
      bestEverFitness: this.bestEverFitness,
      staleGenerations: Math.max(0, this.generation - this.bestFitnessGen),
      network,
    };
  }

  public onGridSizeChanged(): void {
    this.population = this.population.map((agent) =>
      this.environment.createAgent(agent.genome),
    );

    if (this.showcaseGenome) {
      this.showcaseAgent = this.environment.createAgent(this.showcaseGenome);
    } else {
      this.showcaseAgent = null;
    }

    this.beginGenerationEvaluation();
    this.invalidateRandomBoardAgents();
  }

  private setShowcaseGenome(genome: Genome): void {
    this.showcaseGenome = new Float32Array(genome);
    this.showcaseAgent = this.environment.createAgent(this.showcaseGenome);
  }

  private invalidateRandomBoardAgents(): void {
    this.randomBoardAgents = [];
    this.randomBoardGeneration = -1;
  }

  private episodesPerGenomeForGeneration(generation: number): number {
    if (generation >= EPISODES_PER_GENOME_SWITCH_GENERATION) {
      return EPISODES_PER_GENOME_LATE;
    }

    return EPISODES_PER_GENOME_START;
  }

  private beginGenerationEvaluation(): void {
    this.evaluationEpisode = 1;
    this.evaluationEpisodeTarget = this.episodesPerGenomeForGeneration(
      this.generation,
    );
    this.evaluationFitnessSums = new Array(this.population.length).fill(0);
    this.evaluationScoreSums = new Array(this.population.length).fill(0);
    this.evaluationBestScores = new Array(this.population.length).fill(0);
  }

  private accumulateEpisodeResults(): void {
    for (let i = 0; i < this.population.length; i++) {
      const agent = this.population[i]!;
      this.evaluationFitnessSums[i] += this.ga.evaluateFitness(agent);
      this.evaluationScoreSums[i] += agent.score;
      this.evaluationBestScores[i] = Math.max(
        this.evaluationBestScores[i],
        agent.score,
      );
    }
  }

  private finishEpisode(): void {
    this.accumulateEpisodeResults();

    if (this.evaluationEpisode < this.evaluationEpisodeTarget) {
      this.evaluationEpisode += 1;
      this.population = this.population.map((agent) =>
        this.environment.createAgent(agent.genome),
      );
      this.invalidateRandomBoardAgents();
      return;
    }

    this.evolveGeneration();
  }

  private buildEvolutionCandidates(): EvolutionCandidate[] {
    return this.population.map((agent, index) => ({
      genome: agent.genome,
      fitness: this.evaluationFitnessSums[index] / this.evaluationEpisodeTarget,
      score: this.evaluationScoreSums[index] / this.evaluationEpisodeTarget,
    }));
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

  private evolveGeneration(): void {
    const candidates = this.buildEvolutionCandidates();
    const { best, nextGenomes } = this.ga.evolve(candidates, this.generation);
    const generationBestScore = this.evaluationBestScores.reduce(
      (max, score) => Math.max(max, score),
      0,
    );

    this.bestEverScore = Math.max(this.bestEverScore, generationBestScore);
    if (this.generation === 1 || best.fitness > this.bestEverFitness) {
      this.bestEverFitness = best.fitness;
      this.bestFitnessGen = this.generation;
      this.setShowcaseGenome(best.genome);
    }

    this.fitnessHistory.push(best.fitness);
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory.shift();
    }

    this.population = nextGenomes.map((genome) =>
      this.environment.createAgent(genome),
    );
    this.generation += 1;
    this.beginGenerationEvaluation();
    this.invalidateRandomBoardAgents();
  }
}
