import {
  ELITE_COUNT,
  GENE_COUNT,
  GRID_SIZE,
  MUTATION_RATE,
  MUTATION_SIZE,
  POP_SIZE,
  TOURNAMENT_SIZE,
} from "./config";
import type { Agent, Genome } from "./types";

const TOURNAMENT_POOL_RATIO = 0.4;

export type EvolutionResult = {
  best: Agent;
  nextGenomes: Genome[];
};

export class GeneticAlgorithm {
  public randomGenome(): Genome {
    const genome = new Float32Array(GENE_COUNT);
    for (let i = 0; i < genome.length; i++) {
      genome[i] = this.randomRange(-1, 1);
    }
    return genome;
  }

  public evolve(population: Agent[]): EvolutionResult {
    for (const agent of population) {
      agent.fitness = this.fitness(agent);
    }

    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const nextGenomes: Genome[] = [];

    for (let i = 0; i < ELITE_COUNT; i++) {
      nextGenomes.push(ranked[i].genome);
    }

    while (nextGenomes.length < POP_SIZE) {
      const parentA = this.pickParent(ranked);
      const parentB = this.pickParent(ranked);
      const child = this.crossover(parentA.genome, parentB.genome);
      this.mutate(child);
      nextGenomes.push(child);
    }

    return {
      best: ranked[0],
      nextGenomes,
    };
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
  }

  private randomIndex(maxExclusive: number): number {
    return Math.floor(Math.random() * maxExclusive);
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = !agent.alive && agent.hunger > 0 ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }

  private crossover(a: Genome, b: Genome): Genome {
    const child = new Float32Array(GENE_COUNT);
    for (let i = 0; i < GENE_COUNT; i++) {
      child[i] = Math.random() < 0.5 ? a[i] : b[i];
    }
    return child;
  }

  private mutate(genome: Genome): void {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < MUTATION_RATE) {
        genome[i] += this.randomRange(-MUTATION_SIZE, MUTATION_SIZE);
      }
    }
  }

  private pickParent(ranked: Agent[]): Agent {
    const poolSize = Math.max(
      2,
      Math.floor(ranked.length * TOURNAMENT_POOL_RATIO),
    );
    let winner = ranked[this.randomIndex(poolSize)];

    for (let i = 1; i < TOURNAMENT_SIZE; i++) {
      const challenger = ranked[this.randomIndex(poolSize)];
      if (challenger.fitness > winner.fitness) {
        winner = challenger;
      }
    }

    return winner;
  }
}
