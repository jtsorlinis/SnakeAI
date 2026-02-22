import {
  ELITE_COUNT,
  GENE_COUNT,
  GRID_SIZE,
  MUTATION_RATE,
  MUTATION_SIZE,
  POP_SIZE,
} from "./config";
import type { Agent, Genome } from "./types";

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
    const eliteCount = Math.min(ELITE_COUNT, ranked.length);

    for (let i = 0; i < eliteCount; i++) {
      nextGenomes.push(new Float32Array(ranked[i].genome));
    }

    while (nextGenomes.length < POP_SIZE) {
      nextGenomes.push(this.spawnChild(ranked, eliteCount));
    }

    return {
      best: ranked[0],
      nextGenomes,
    };
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = !agent.alive && agent.hunger > 0 ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }

  private mutateInPlace(genome: Genome): void {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < MUTATION_RATE) {
        genome[i] += this.randomRange(-MUTATION_SIZE, MUTATION_SIZE);
      }
    }
  }

  private spawnChild(ranked: Agent[], eliteCount: number): Genome {
    const parent = ranked[Math.floor(Math.random() * eliteCount)];
    const child = new Float32Array(parent.genome);
    this.mutateInPlace(child);
    return child;
  }
}
