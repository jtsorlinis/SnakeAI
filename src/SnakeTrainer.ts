import { POP_SIZE } from "./config";
import { GeneticAlgorithm } from "./GeneticAlgorithm";
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

  private showcaseGenome: Genome | null = null;
  private showcaseAgent: Agent | null = null;

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

    for (let i = 0; i < POP_SIZE; i++) {
      this.population.push(
        this.environment.createAgent(this.ga.randomGenome()),
      );
    }

    if (this.population.length > 0) {
      this.setShowcaseGenome(this.population[0].genome);
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

      if (!this.showcaseAgent?.alive && this.showcaseGenome) {
        this.showcaseAgent = this.environment.createAgent(this.showcaseGenome);
      }
    }
  }

  public getState(): TrainerState {
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
    this.population = this.population.map((agent) =>
      this.environment.createAgent(agent.genome),
    );

    if (this.showcaseGenome) {
      this.showcaseAgent = this.environment.createAgent(this.showcaseGenome);
    } else {
      this.showcaseAgent = null;
    }
  }

  private setShowcaseGenome(genome: Genome): void {
    this.showcaseGenome = new Float32Array(genome);
    this.showcaseAgent = this.environment.createAgent(this.showcaseGenome);
  }

  private evolve(): void {
    const { best, nextGenomes } = this.ga.evolve(this.population);

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

    this.population = nextGenomes.map((genome) =>
      this.environment.createAgent(genome),
    );
    this.generation += 1;
  }
}
