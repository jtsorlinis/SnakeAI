import {
  CMA_INITIAL_SIGMA,
  CMA_MAX_SIGMA,
  CMA_MIN_SIGMA,
  CMA_MIN_VARIANCE,
  GENE_COUNT,
  GRID_SIZE,
  POP_SIZE,
} from "./config";
import type { Agent, Genome } from "./types";

export type EvolutionResult = {
  best: Agent;
  nextGenomes: Genome[];
};

export class CmaEs {
  private readonly dimension = GENE_COUNT;
  private readonly lambda = POP_SIZE;
  private readonly mu = Math.max(1, Math.floor(this.lambda / 2));
  private readonly weights: Float64Array;
  private readonly mueff: number;

  private readonly cc: number;
  private readonly cs: number;
  private readonly c1: number;
  private readonly cmu: number;
  private readonly damps: number;
  private readonly chiN: number;

  private mean = new Float32Array(this.dimension);
  private diagC = new Float32Array(this.dimension);
  private pc = new Float32Array(this.dimension);
  private ps = new Float32Array(this.dimension);
  private sigma = CMA_INITIAL_SIGMA;
  private generation = 0;
  private spareNormal: number | null = null;

  constructor() {
    const rawWeights = new Float64Array(this.mu);
    for (let i = 0; i < this.mu; i++) {
      rawWeights[i] = Math.log(this.mu + 0.5) - Math.log(i + 1);
    }

    let weightSum = 0;
    for (let i = 0; i < this.mu; i++) {
      weightSum += rawWeights[i];
    }

    this.weights = new Float64Array(this.mu);
    let sqWeightSum = 0;
    for (let i = 0; i < this.mu; i++) {
      const normalized = rawWeights[i] / weightSum;
      this.weights[i] = normalized;
      sqWeightSum += normalized * normalized;
    }
    this.mueff = 1 / sqWeightSum;

    const n = this.dimension;
    this.cc = (4 + this.mueff / n) / (n + 4 + (2 * this.mueff) / n);
    this.cs = (this.mueff + 2) / (n + this.mueff + 5);
    this.c1 = 2 / ((n + 1.3) * (n + 1.3) + this.mueff);
    this.cmu = Math.min(
      1 - this.c1,
      (2 * (this.mueff - 2 + 1 / this.mueff)) /
        ((n + 2) * (n + 2) + this.mueff),
    );
    this.damps =
      1 +
      2 * Math.max(0, Math.sqrt((this.mueff - 1) / (n + 1)) - 1) +
      this.cs;
    this.chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
  }

  public reset(): Genome[] {
    this.mean = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      this.mean[i] = this.randomRange(-1, 1);
    }

    this.diagC = new Float32Array(this.dimension);
    this.diagC.fill(1);
    this.pc = new Float32Array(this.dimension);
    this.ps = new Float32Array(this.dimension);
    this.sigma = CMA_INITIAL_SIGMA;
    this.generation = 0;
    this.spareNormal = null;

    return this.samplePopulation();
  }

  public evolve(population: Agent[]): EvolutionResult {
    for (const agent of population) {
      agent.fitness = this.fitness(agent);
    }

    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const oldMean = new Float32Array(this.mean);
    const yW = new Float64Array(this.dimension);

    for (let rank = 0; rank < this.mu; rank++) {
      const genome = ranked[rank].genome;
      const weight = this.weights[rank];
      for (let gene = 0; gene < this.dimension; gene++) {
        yW[gene] += weight * ((genome[gene] - oldMean[gene]) / this.sigma);
      }
    }

    for (let gene = 0; gene < this.dimension; gene++) {
      this.mean[gene] = oldMean[gene] + this.sigma * yW[gene];
    }

    const psFactor = Math.sqrt(this.cs * (2 - this.cs) * this.mueff);
    for (let gene = 0; gene < this.dimension; gene++) {
      const variance = Math.max(CMA_MIN_VARIANCE, this.diagC[gene]);
      const invSqrtCy = yW[gene] / Math.sqrt(variance);
      this.ps[gene] =
        (1 - this.cs) * this.ps[gene] + psFactor * invSqrtCy;
    }

    const normPs = this.vectorNorm(this.ps);
    const hsig =
      normPs /
        Math.sqrt(1 - Math.pow(1 - this.cs, 2 * (this.generation + 1))) /
        this.chiN <
      1.4 + 2 / (this.dimension + 1);
    const hsigValue = hsig ? 1 : 0;

    const pcFactor = Math.sqrt(this.cc * (2 - this.cc) * this.mueff);
    for (let gene = 0; gene < this.dimension; gene++) {
      this.pc[gene] =
        (1 - this.cc) * this.pc[gene] + hsigValue * pcFactor * yW[gene];
    }

    const covarianceDecay =
      1 -
      this.c1 -
      this.cmu +
      this.c1 * (1 - hsigValue) * this.cc * (2 - this.cc);

    for (let gene = 0; gene < this.dimension; gene++) {
      let rankMu = 0;
      for (let rank = 0; rank < this.mu; rank++) {
        const y = (ranked[rank].genome[gene] - oldMean[gene]) / this.sigma;
        rankMu += this.weights[rank] * y * y;
      }

      const nextVariance =
        covarianceDecay * this.diagC[gene] +
        this.c1 * this.pc[gene] * this.pc[gene] +
        this.cmu * rankMu;

      this.diagC[gene] = Math.max(CMA_MIN_VARIANCE, nextVariance);
    }

    this.sigma *= Math.exp(
      (this.cs / this.damps) * (normPs / this.chiN - 1),
    );
    this.sigma = Math.min(CMA_MAX_SIGMA, Math.max(CMA_MIN_SIGMA, this.sigma));
    this.generation += 1;

    return {
      best: ranked[0],
      nextGenomes: this.samplePopulation(),
    };
  }

  private samplePopulation(): Genome[] {
    const genomes: Genome[] = [];
    for (let i = 0; i < this.lambda; i++) {
      genomes.push(this.randomGenomeFromDistribution());
    }
    return genomes;
  }

  private randomGenomeFromDistribution(): Genome {
    const genome = new Float32Array(this.dimension);
    for (let gene = 0; gene < this.dimension; gene++) {
      const variance = Math.max(CMA_MIN_VARIANCE, this.diagC[gene]);
      const stdDev = this.sigma * Math.sqrt(variance);
      genome[gene] = this.mean[gene] + stdDev * this.randomNormal();
    }
    return genome;
  }

  private randomNormal(): number {
    if (this.spareNormal !== null) {
      const sample = this.spareNormal;
      this.spareNormal = null;
      return sample;
    }

    let u = 0;
    let v = 0;
    while (u === 0) {
      u = Math.random();
    }
    while (v === 0) {
      v = Math.random();
    }

    const magnitude = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    this.spareNormal = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }

  private vectorNorm(values: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
      sum += values[i] * values[i];
    }
    return Math.sqrt(sum);
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
}
