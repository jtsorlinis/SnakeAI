import {
  ES_PARENT_COUNT,
  ES_SIGMA,
  GENE_COUNT,
  GRID_SIZE,
  POP_SIZE,
} from "./config";
import type { Agent, Genome } from "./types";

export type EvolutionResult = {
  best: Agent;
  nextGenomes: Genome[];
};

export class SimpleEvolutionStrategy {
  private spareNormal: number | null = null;

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
    const parentCount = Math.max(1, Math.min(ES_PARENT_COUNT, ranked.length));
    const parents = ranked.slice(0, parentCount);
    const nextGenomes: Genome[] = [new Float32Array(ranked[0].genome)];

    while (nextGenomes.length < POP_SIZE) {
      const parent = parents[this.randomIndex(parentCount)];
      nextGenomes.push(this.sampleChild(parent.genome));
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

  private sampleChild(parent: Genome): Genome {
    const child = new Float32Array(parent);
    for (let i = 0; i < child.length; i++) {
      child[i] += this.randomNormal() * ES_SIGMA;
    }
    return child;
  }

  private randomNormal(): number {
    if (this.spareNormal !== null) {
      const normal = this.spareNormal;
      this.spareNormal = null;
      return normal;
    }

    let u = 0;
    let v = 0;
    while (u === 0) {
      u = Math.random();
    }
    while (v === 0) {
      v = Math.random();
    }

    const mag = Math.sqrt(-2.0 * Math.log(u));
    const theta = 2.0 * Math.PI * v;
    this.spareNormal = mag * Math.sin(theta);
    return mag * Math.cos(theta);
  }
}
