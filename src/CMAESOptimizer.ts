import { CMA_INITIAL_SIGMA, GENE_COUNT, GRID_SIZE, POP_SIZE } from "./config";
import type { Agent, Genome } from "./types";

const MIN_EIGENVALUE = 1e-12;
const MIN_SIGMA = 1e-4;
const MAX_SIGMA = 5;
const MAX_JACOBI_SWEEPS = 50;
const JACOBI_EPSILON = 1e-12;

type Sample = {
  genome: Genome;
  normalizedStep: Float64Array;
  step: Float64Array;
};

export type EvolutionResult = {
  best: Agent;
  nextGenomes: Genome[];
};

export class CMAESOptimizer {
  private readonly dimension = GENE_COUNT;
  private readonly lambda = POP_SIZE;
  private readonly mu = Math.max(1, Math.floor(this.lambda / 2));
  private readonly weights = this.buildRecombinationWeights(this.mu);
  private readonly mueff = this.computeEffectiveSampleSize(this.weights);
  private readonly cc =
    (4 + this.mueff / this.dimension) /
    (this.dimension + 4 + (2 * this.mueff) / this.dimension);
  private readonly cs =
    (this.mueff + 2) / (this.dimension + this.mueff + 5);
  private readonly c1 =
    2 / ((this.dimension + 1.3) * (this.dimension + 1.3) + this.mueff);
  private readonly cmu = Math.min(
    1 - this.c1,
    (2 * (this.mueff - 2 + 1 / this.mueff)) /
      ((this.dimension + 2) * (this.dimension + 2) + this.mueff),
  );
  private readonly damps =
    1 +
    2 * Math.max(0, Math.sqrt((this.mueff - 1) / (this.dimension + 1)) - 1) +
    this.cs;
  private readonly chiN =
    Math.sqrt(this.dimension) *
    (1 - 1 / (4 * this.dimension) + 1 / (21 * this.dimension * this.dimension));

  private generation = 0;
  private sigma = CMA_INITIAL_SIGMA;

  private readonly mean = new Float64Array(this.dimension);
  private readonly covariance = new Float64Array(this.dimension * this.dimension);
  private readonly eigenVectors = new Float64Array(this.dimension * this.dimension);
  private readonly eigenValues = new Float64Array(this.dimension);
  private readonly eigenScales = new Float64Array(this.dimension);
  private readonly ps = new Float64Array(this.dimension);
  private readonly pc = new Float64Array(this.dimension);

  private readonly weightedZ = new Float64Array(this.dimension);
  private readonly weightedY = new Float64Array(this.dimension);
  private readonly transformedZW = new Float64Array(this.dimension);
  private readonly scaledNormal = new Float64Array(this.dimension);
  private readonly rankMuCovariance = new Float64Array(
    this.dimension * this.dimension,
  );

  private readonly jacobiMatrix = new Float64Array(this.dimension * this.dimension);
  private readonly jacobiB = new Float64Array(this.dimension);
  private readonly jacobiZ = new Float64Array(this.dimension);

  private currentSamples: Sample[] = [];
  private gaussianSpare: number | null = null;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.generation = 0;
    this.sigma = CMA_INITIAL_SIGMA;
    this.gaussianSpare = null;
    this.currentSamples = [];

    const n = this.dimension;
    for (let i = 0; i < n; i++) {
      this.mean[i] = this.randomRange(-1, 1);
      this.ps[i] = 0;
      this.pc[i] = 0;
      this.eigenValues[i] = 1;
      this.eigenScales[i] = 1;

      const rowOffset = i * n;
      for (let j = 0; j < n; j++) {
        const value = i === j ? 1 : 0;
        this.covariance[rowOffset + j] = value;
        this.eigenVectors[rowOffset + j] = value;
      }
    }
  }

  public samplePopulation(): Genome[] {
    this.currentSamples = this.generateSamples();
    return this.currentSamples.map((sample) => sample.genome);
  }

  public evolve(population: Agent[]): EvolutionResult {
    if (population.length !== this.lambda) {
      throw new Error(
        `CMA-ES expected population of ${this.lambda}, got ${population.length}.`,
      );
    }
    if (this.currentSamples.length !== this.lambda) {
      throw new Error(
        "CMA-ES population not initialized. Call samplePopulation() before evolve().",
      );
    }

    const ranked = population
      .map((agent, index) => {
        agent.fitness = this.fitness(agent);
        return { agent, index };
      })
      .sort((a, b) => b.agent.fitness - a.agent.fitness);

    this.weightedZ.fill(0);
    this.weightedY.fill(0);

    for (let rank = 0; rank < this.mu; rank++) {
      const weight = this.weights[rank];
      const sample = this.currentSamples[ranked[rank].index];
      for (let i = 0; i < this.dimension; i++) {
        this.weightedZ[i] += weight * sample.normalizedStep[i];
        this.weightedY[i] += weight * sample.step[i];
      }
    }

    for (let i = 0; i < this.dimension; i++) {
      this.mean[i] += this.sigma * this.weightedY[i];
    }

    this.multiplyBWithVector(this.weightedZ, this.transformedZW);

    const psFactor = Math.sqrt(this.cs * (2 - this.cs) * this.mueff);
    for (let i = 0; i < this.dimension; i++) {
      this.ps[i] = (1 - this.cs) * this.ps[i] + psFactor * this.transformedZW[i];
    }

    const psNorm = this.norm(this.ps);
    const expectedNorm =
      Math.sqrt(1 - Math.pow(1 - this.cs, 2 * (this.generation + 1))) *
      this.chiN;
    const hsig =
      psNorm / Math.max(1e-12, expectedNorm) < 1.4 + 2 / (this.dimension + 1)
        ? 1
        : 0;

    const pcFactor = Math.sqrt(this.cc * (2 - this.cc) * this.mueff);
    for (let i = 0; i < this.dimension; i++) {
      this.pc[i] =
        (1 - this.cc) * this.pc[i] + hsig * pcFactor * this.weightedY[i];
    }

    this.rankMuCovariance.fill(0);
    for (let rank = 0; rank < this.mu; rank++) {
      const weight = this.weights[rank];
      const step = this.currentSamples[ranked[rank].index].step;
      for (let i = 0; i < this.dimension; i++) {
        const weightedStep = weight * step[i];
        const rowOffset = i * this.dimension;
        for (let j = 0; j < this.dimension; j++) {
          this.rankMuCovariance[rowOffset + j] += weightedStep * step[j];
        }
      }
    }

    const covarianceDecay =
      1 - this.c1 - this.cmu + (1 - hsig) * this.c1 * this.cc * (2 - this.cc);

    for (let i = 0; i < this.dimension; i++) {
      const rowOffset = i * this.dimension;
      for (let j = 0; j < this.dimension; j++) {
        const index = rowOffset + j;
        this.covariance[index] =
          covarianceDecay * this.covariance[index] +
          this.c1 * this.pc[i] * this.pc[j] +
          this.cmu * this.rankMuCovariance[index];
      }
    }

    this.ensureSymmetricCovariance();

    const sigmaScale = Math.exp(
      (this.cs / this.damps) * (psNorm / this.chiN - 1),
    );
    this.sigma = this.clamp(this.sigma * sigmaScale, MIN_SIGMA, MAX_SIGMA);

    this.generation += 1;
    this.updateEigenDecomposition();
    this.currentSamples = this.generateSamples();

    return {
      best: ranked[0].agent,
      nextGenomes: this.currentSamples.map((sample) => sample.genome),
    };
  }

  private generateSamples(): Sample[] {
    const samples: Sample[] = [];

    for (let sampleIndex = 0; sampleIndex < this.lambda; sampleIndex++) {
      const normalizedStep = new Float64Array(this.dimension);
      const step = new Float64Array(this.dimension);
      const genome = new Float32Array(this.dimension);

      for (let i = 0; i < this.dimension; i++) {
        normalizedStep[i] = this.randomNormal();
      }

      this.transformNormalToStep(normalizedStep, step);

      for (let i = 0; i < this.dimension; i++) {
        genome[i] = this.mean[i] + this.sigma * step[i];
      }

      samples.push({ genome, normalizedStep, step });
    }

    return samples;
  }

  private transformNormalToStep(
    normalizedStep: Float64Array,
    stepTarget: Float64Array,
  ): void {
    for (let i = 0; i < this.dimension; i++) {
      this.scaledNormal[i] = this.eigenScales[i] * normalizedStep[i];
    }

    this.multiplyBWithVector(this.scaledNormal, stepTarget);
  }

  private multiplyBWithVector(
    source: Float64Array,
    target: Float64Array,
  ): void {
    const n = this.dimension;
    for (let row = 0; row < n; row++) {
      let sum = 0;
      const rowOffset = row * n;
      for (let col = 0; col < n; col++) {
        sum += this.eigenVectors[rowOffset + col] * source[col];
      }
      target[row] = sum;
    }
  }

  private ensureSymmetricCovariance(): void {
    const n = this.dimension;
    for (let i = 0; i < n; i++) {
      const diagIndex = i * n + i;
      this.covariance[diagIndex] = Math.max(
        MIN_EIGENVALUE,
        this.covariance[diagIndex],
      );

      for (let j = i + 1; j < n; j++) {
        const indexA = i * n + j;
        const indexB = j * n + i;
        const average = 0.5 * (this.covariance[indexA] + this.covariance[indexB]);
        this.covariance[indexA] = average;
        this.covariance[indexB] = average;
      }
    }
  }

  private updateEigenDecomposition(): void {
    this.jacobiMatrix.set(this.covariance);
    this.jacobiEigenDecomposition(
      this.jacobiMatrix,
      this.eigenVectors,
      this.eigenValues,
    );
    this.sortEigenPairsDescending(this.eigenValues, this.eigenVectors);

    for (let i = 0; i < this.dimension; i++) {
      this.eigenValues[i] = Math.max(MIN_EIGENVALUE, this.eigenValues[i]);
      this.eigenScales[i] = Math.sqrt(this.eigenValues[i]);
    }

    this.rebuildCovarianceFromEigenSystem();
  }

  private jacobiEigenDecomposition(
    matrix: Float64Array,
    vectors: Float64Array,
    values: Float64Array,
  ): void {
    const n = this.dimension;
    const b = this.jacobiB;
    const z = this.jacobiZ;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        vectors[i * n + j] = i === j ? 1 : 0;
      }

      const diagonalValue = matrix[i * n + i];
      values[i] = diagonalValue;
      b[i] = diagonalValue;
      z[i] = 0;
    }

    for (let sweep = 0; sweep < MAX_JACOBI_SWEEPS; sweep++) {
      let offDiagonalSum = 0;
      for (let p = 0; p < n - 1; p++) {
        const rowOffset = p * n;
        for (let q = p + 1; q < n; q++) {
          offDiagonalSum += Math.abs(matrix[rowOffset + q]);
        }
      }

      if (offDiagonalSum <= JACOBI_EPSILON) {
        break;
      }

      const threshold =
        sweep < 3 ? (0.2 * offDiagonalSum) / (n * n) : 0;

      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          const pqIndex = p * n + q;
          const apq = matrix[pqIndex];
          const g = 100 * Math.abs(apq);

          if (
            sweep > 3 &&
            Math.abs(values[p]) + g === Math.abs(values[p]) &&
            Math.abs(values[q]) + g === Math.abs(values[q])
          ) {
            matrix[pqIndex] = 0;
            matrix[q * n + p] = 0;
            continue;
          }

          if (Math.abs(apq) <= threshold) {
            continue;
          }

          const deltaEigen = values[q] - values[p];
          let t = 0;
          if (Math.abs(deltaEigen) + g === Math.abs(deltaEigen)) {
            t = apq / deltaEigen;
          } else {
            const theta = 0.5 * deltaEigen / apq;
            t = 1 / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
            if (theta < 0) {
              t = -t;
            }
          }

          const c = 1 / Math.sqrt(1 + t * t);
          const s = t * c;
          const tau = s / (1 + c);
          const shift = t * apq;

          z[p] -= shift;
          z[q] += shift;
          values[p] -= shift;
          values[q] += shift;
          matrix[pqIndex] = 0;
          matrix[q * n + p] = 0;

          for (let j = 0; j < p; j++) {
            this.rotateSymmetricEntries(matrix, j, p, j, q, s, tau);
          }
          for (let j = p + 1; j < q; j++) {
            this.rotateSymmetricEntries(matrix, p, j, j, q, s, tau);
          }
          for (let j = q + 1; j < n; j++) {
            this.rotateSymmetricEntries(matrix, p, j, q, j, s, tau);
          }
          for (let j = 0; j < n; j++) {
            this.rotateGeneralEntries(vectors, j, p, j, q, s, tau);
          }
        }
      }

      for (let i = 0; i < n; i++) {
        b[i] += z[i];
        values[i] = b[i];
        z[i] = 0;
      }
    }
  }

  private rotateSymmetricEntries(
    matrix: Float64Array,
    i: number,
    j: number,
    k: number,
    l: number,
    s: number,
    tau: number,
  ): void {
    const n = this.dimension;
    const indexA = i * n + j;
    const indexB = k * n + l;
    const g = matrix[indexA];
    const h = matrix[indexB];
    const updatedA = g - s * (h + g * tau);
    const updatedB = h + s * (g - h * tau);

    matrix[indexA] = updatedA;
    matrix[j * n + i] = updatedA;
    matrix[indexB] = updatedB;
    matrix[l * n + k] = updatedB;
  }

  private rotateGeneralEntries(
    matrix: Float64Array,
    i: number,
    j: number,
    k: number,
    l: number,
    s: number,
    tau: number,
  ): void {
    const n = this.dimension;
    const indexA = i * n + j;
    const indexB = k * n + l;
    const g = matrix[indexA];
    const h = matrix[indexB];
    matrix[indexA] = g - s * (h + g * tau);
    matrix[indexB] = h + s * (g - h * tau);
  }

  private sortEigenPairsDescending(
    values: Float64Array,
    vectors: Float64Array,
  ): void {
    const n = this.dimension;

    for (let i = 0; i < n - 1; i++) {
      let best = i;
      for (let j = i + 1; j < n; j++) {
        if (values[j] > values[best]) {
          best = j;
        }
      }

      if (best === i) {
        continue;
      }

      const value = values[i];
      values[i] = values[best];
      values[best] = value;

      for (let row = 0; row < n; row++) {
        const leftIndex = row * n + i;
        const rightIndex = row * n + best;
        const temp = vectors[leftIndex];
        vectors[leftIndex] = vectors[rightIndex];
        vectors[rightIndex] = temp;
      }
    }
  }

  private rebuildCovarianceFromEigenSystem(): void {
    const n = this.dimension;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum +=
            this.eigenVectors[i * n + k] *
            this.eigenValues[k] *
            this.eigenVectors[j * n + k];
        }

        const indexA = i * n + j;
        const indexB = j * n + i;
        this.covariance[indexA] = sum;
        this.covariance[indexB] = sum;
      }
    }
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
  }

  private randomNormal(): number {
    if (this.gaussianSpare !== null) {
      const spare = this.gaussianSpare;
      this.gaussianSpare = null;
      return spare;
    }

    let u1 = 0;
    while (u1 <= Number.EPSILON) {
      u1 = Math.random();
    }
    const u2 = Math.random();
    const magnitude = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;

    this.gaussianSpare = magnitude * Math.sin(theta);
    return magnitude * Math.cos(theta);
  }

  private norm(vector: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < vector.length; i++) {
      sum += vector[i] * vector[i];
    }
    return Math.sqrt(sum);
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = !agent.alive && agent.hunger > 0 ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }

  private buildRecombinationWeights(mu: number): Float64Array {
    const weights = new Float64Array(mu);
    let sum = 0;

    for (let i = 0; i < mu; i++) {
      const weight = Math.log(mu + 0.5) - Math.log(i + 1);
      weights[i] = weight;
      sum += weight;
    }

    for (let i = 0; i < mu; i++) {
      weights[i] /= sum;
    }

    return weights;
  }

  private computeEffectiveSampleSize(weights: Float64Array): number {
    let sumSquares = 0;
    for (let i = 0; i < weights.length; i++) {
      sumSquares += weights[i] * weights[i];
    }
    return 1 / sumSquares;
  }
}
