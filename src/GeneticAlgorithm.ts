import {
  CROSSOVER_RATE,
  ELITE_RATIO,
  GENE_COUNT,
  GRID_SIZE,
  MUTATION_RATE,
  MUTATION_SIGMA_DECAY,
  MUTATION_SIGMA_MIN,
  MUTATION_SIGMA_START,
  TOURNAMENT_SIZE,
} from "./config";
import type { Agent, Genome } from "./types";

export type EvolutionCandidate = {
  genome: Genome;
  fitness: number;
  score: number;
};

export type EvolutionResult = {
  best: EvolutionCandidate;
  nextGenomes: Genome[];
};

export class GeneticAlgorithm {
  private hasSpareGaussian = false;
  private spareGaussian = 0;

  public randomGenome(): Genome {
    const genome = new Float32Array(GENE_COUNT);
    for (let i = 0; i < genome.length; i++) {
      genome[i] = this.randomRange(-1, 1);
    }
    return genome;
  }

  public evaluateFitness(agent: Agent): number {
    const area = GRID_SIZE * GRID_SIZE;
    const score = agent.score;

    // Main objective: getting food (quadratic term helps push past low-score plateaus)
    const foodReward = 20 * score + 10 * score * score;

    // Small survival bonus (capped so "just living" can't dominate)
    const survivalBonus = 0.005 * Math.min(agent.steps, area);

    // Penalize stalling (time since last food), grows faster the longer it stalls
    const stallLinear = 0.02 * agent.stepsSinceFood;
    const stallQuadratic =
      0.01 * (agent.stepsSinceFood * agent.stepsSinceFood) / area;
    const stallPenalty = stallLinear + stallQuadratic;

    // Death type penalties (starvation should hurt more than collision)
    const collided = !agent.alive && agent.hunger > 0;
    const starved = !agent.alive && agent.hunger === 0;

    const collisionPenalty = collided ? 2 : 0;
    const starvationPenalty = starved ? 8 : 0;

    return (
      foodReward +
      survivalBonus -
      stallPenalty -
      collisionPenalty -
      starvationPenalty
    );
  }

  public evolve(
    population: readonly EvolutionCandidate[],
    generation: number,
  ): EvolutionResult {
    if (population.length === 0) {
      throw new Error("Cannot evolve an empty population");
    }

    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const nextGenomes: Genome[] = [];
    const targetSize = population.length;
    const eliteCount = Math.max(1, Math.round(targetSize * ELITE_RATIO));
    const mutationSigma = this.mutationSigma(generation);

    for (let i = 0; i < Math.min(eliteCount, targetSize); i++) {
      nextGenomes.push(new Float32Array(ranked[i].genome));
    }

    while (nextGenomes.length < targetSize) {
      const parentA = this.pickParent(ranked);
      const parentB = this.pickParent(ranked);
      const child = this.makeChild(parentA.genome, parentB.genome);
      this.mutate(child, mutationSigma);
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

  private mutationSigma(generation: number): number {
    const generationNumber = Math.max(1, generation);
    return Math.max(
      MUTATION_SIGMA_MIN,
      MUTATION_SIGMA_START *
        Math.pow(MUTATION_SIGMA_DECAY, generationNumber - 1),
    );
  }

  private makeChild(a: Genome, b: Genome): Genome {
    if (Math.random() >= CROSSOVER_RATE) {
      return new Float32Array(a);
    }

    return this.crossover(a, b);
  }

  private crossover(a: Genome, b: Genome): Genome {
    const child = new Float32Array(GENE_COUNT);
    for (let i = 0; i < GENE_COUNT; i++) {
      child[i] = Math.random() < 0.5 ? a[i] : b[i];
    }
    return child;
  }

  private mutate(genome: Genome, sigma: number): void {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < MUTATION_RATE) {
        genome[i] += this.randomGaussian(0, sigma);
      }
    }
  }

  private randomGaussian(mean: number, standardDeviation: number): number {
    if (this.hasSpareGaussian) {
      this.hasSpareGaussian = false;
      return mean + standardDeviation * this.spareGaussian;
    }

    let u = 0;
    let v = 0;
    let magnitude = 0;

    do {
      u = this.randomRange(-1, 1);
      v = this.randomRange(-1, 1);
      magnitude = u * u + v * v;
    } while (magnitude <= Number.EPSILON || magnitude >= 1);

    const factor = Math.sqrt((-2 * Math.log(magnitude)) / magnitude);
    this.spareGaussian = v * factor;
    this.hasSpareGaussian = true;

    return mean + standardDeviation * (u * factor);
  }

  private pickParent(
    ranked: readonly EvolutionCandidate[],
  ): EvolutionCandidate {
    let winner = ranked[this.randomIndex(ranked.length)];

    for (let i = 1; i < TOURNAMENT_SIZE; i++) {
      const challenger = ranked[this.randomIndex(ranked.length)];
      if (challenger.fitness > winner.fitness) {
        winner = challenger;
      }
    }

    return winner;
  }
}
