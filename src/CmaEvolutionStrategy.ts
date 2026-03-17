import {
  CMA_INITIAL_SIGMA,
  GRID_SIZE,
  POLICY_PARAM_COUNT,
  POP_SIZE,
} from "./config";
import type { Agent, PolicyParams } from "./types";

export type EvolutionResult = {
  best: Agent;
  nextPolicies: PolicyParams[];
};

export class CmaEvolutionStrategy {
  private readonly dimension = POLICY_PARAM_COUNT;
  private readonly populationSize = POP_SIZE;
  private readonly mu = Math.max(1, Math.floor(this.populationSize / 2));
  private readonly weights = this.createWeights();
  private readonly muEff = this.computeMuEff();
  private readonly cSigma =
    (this.muEff + 2) / (this.dimension + this.muEff + 5);
  private readonly dSigma =
    1 +
    2 *
      Math.max(
        0,
        Math.sqrt((this.muEff - 1) / (this.dimension + 1)) - 1,
      ) +
    this.cSigma;
  private readonly cC =
    (4 + this.muEff / this.dimension) /
    (this.dimension + 4 + (2 * this.muEff) / this.dimension);
  private readonly c1 =
    2 / (Math.pow(this.dimension + 1.3, 2) + this.muEff);
  private readonly cMu = Math.min(
    1 - this.c1,
    (2 * (this.muEff - 2 + 1 / this.muEff)) /
      (Math.pow(this.dimension + 2, 2) + this.muEff),
  );
  private readonly chiN =
    Math.sqrt(this.dimension) *
    (1 -
      1 / (4 * this.dimension) +
      1 / (21 * this.dimension * this.dimension));

  private mean = new Float64Array(this.dimension);
  private covariance = this.createIdentityMatrix();
  private eigenvectors = this.createIdentityMatrix();
  private axisScales = new Float64Array(this.dimension).fill(1);
  private pC = new Float64Array(this.dimension);
  private pSigma = new Float64Array(this.dimension);
  private generation = 0;
  private sigma = CMA_INITIAL_SIGMA;
  private spareNormal: number | null = null;

  constructor() {
    this.resetState();
  }

  public initializePopulation(): PolicyParams[] {
    this.resetState();

    const policies: PolicyParams[] = [];
    for (let sample = 0; sample < this.populationSize; sample++) {
      const policy = new Float32Array(this.dimension);
      for (let gene = 0; gene < this.dimension; gene++) {
        policy[gene] = this.randomRange(-1, 1);
        this.mean[gene] += policy[gene];
      }
      policies.push(policy);
    }

    for (let gene = 0; gene < this.dimension; gene++) {
      this.mean[gene] /= this.populationSize;
    }

    return policies;
  }

  public evolve(population: Agent[]): EvolutionResult {
    for (const agent of population) {
      agent.fitness = this.fitness(agent);
    }

    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const best = ranked[0];
    const previousMean = new Float64Array(this.mean);
    const nextMean = this.recombineMean(ranked);
    const meanShift = this.computeNormalizedMeanShift(previousMean, nextMean);
    const whitenedShift = this.applyInverseSqrtCovariance(meanShift);

    const pSigmaFactor = Math.sqrt(
      this.cSigma * (2 - this.cSigma) * this.muEff,
    );
    for (let i = 0; i < this.dimension; i++) {
      this.pSigma[i] =
        (1 - this.cSigma) * this.pSigma[i] + pSigmaFactor * whitenedShift[i];
    }

    const pSigmaNorm = this.vectorNorm(this.pSigma);
    const sigmaDecay = Math.pow(1 - this.cSigma, 2 * (this.generation + 1));
    const hSigma =
      pSigmaNorm /
        Math.sqrt(Math.max(1e-12, 1 - sigmaDecay)) /
        this.chiN <
      1.4 + 2 / (this.dimension + 1)
        ? 1
        : 0;

    const pCFactor = Math.sqrt(this.cC * (2 - this.cC) * this.muEff);
    for (let i = 0; i < this.dimension; i++) {
      this.pC[i] =
        (1 - this.cC) * this.pC[i] + hSigma * pCFactor * meanShift[i];
    }

    this.updateCovariance(ranked, previousMean, hSigma);

    for (let i = 0; i < this.dimension; i++) {
      this.mean[i] = nextMean[i];
    }
    this.sigma =
      this.sigma *
      Math.exp((this.cSigma / this.dSigma) * (pSigmaNorm / this.chiN - 1));
    this.sigma = Math.max(1e-8, this.sigma);
    this.generation += 1;
    this.decomposeCovariance();

    return {
      best,
      nextPolicies: this.samplePopulation(),
    };
  }

  public getSigma(): number {
    return this.sigma;
  }

  private resetState(): void {
    this.mean = new Float64Array(this.dimension);
    this.covariance = this.createIdentityMatrix();
    this.eigenvectors = this.createIdentityMatrix();
    this.axisScales.fill(1);
    this.pC.fill(0);
    this.pSigma.fill(0);
    this.sigma = CMA_INITIAL_SIGMA;
    this.generation = 0;
    this.spareNormal = null;
  }

  private createWeights(): Float64Array {
    const weights = new Float64Array(this.mu);
    let total = 0;

    for (let i = 0; i < this.mu; i++) {
      const weight = Math.log(this.mu + 0.5) - Math.log(i + 1);
      weights[i] = weight;
      total += weight;
    }

    for (let i = 0; i < this.mu; i++) {
      weights[i] /= total;
    }

    return weights;
  }

  private computeMuEff(): number {
    let sumSquares = 0;
    for (let i = 0; i < this.weights.length; i++) {
      sumSquares += this.weights[i] * this.weights[i];
    }
    return 1 / sumSquares;
  }

  private recombineMean(ranked: Agent[]): Float64Array {
    const nextMean = new Float64Array(this.dimension);

    for (let parent = 0; parent < this.mu; parent++) {
      const policy = ranked[parent].policy;
      const weight = this.weights[parent];
      for (let gene = 0; gene < this.dimension; gene++) {
        nextMean[gene] += weight * policy[gene];
      }
    }

    return nextMean;
  }

  private computeNormalizedMeanShift(
    previousMean: Float64Array,
    nextMean: Float64Array,
  ): Float64Array {
    const shift = new Float64Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      shift[i] = (nextMean[i] - previousMean[i]) / this.sigma;
    }
    return shift;
  }

  private updateCovariance(
    ranked: Agent[],
    previousMean: Float64Array,
    hSigma: number,
  ): void {
    const rankMu = new Float64Array(this.dimension * this.dimension);

    for (let parent = 0; parent < this.mu; parent++) {
      const normalized = new Float64Array(this.dimension);
      const policy = ranked[parent].policy;

      for (let gene = 0; gene < this.dimension; gene++) {
        normalized[gene] = (policy[gene] - previousMean[gene]) / this.sigma;
      }

      this.addOuterProductSymmetric(
        rankMu,
        normalized,
        this.weights[parent] * this.cMu,
      );
    }

    const oldCovarianceScale =
      1 -
      this.c1 -
      this.cMu +
      (1 - hSigma) * this.c1 * this.cC * (2 - this.cC);

    for (let row = 0; row < this.dimension; row++) {
      const rowOffset = row * this.dimension;
      for (let col = row; col < this.dimension; col++) {
        const index = rowOffset + col;
        let nextValue = this.covariance[index] * oldCovarianceScale;
        nextValue += this.c1 * this.pC[row] * this.pC[col];
        nextValue += rankMu[index];

        if (row === col) {
          nextValue = Math.max(1e-12, nextValue);
        }

        this.covariance[index] = nextValue;
        this.covariance[col * this.dimension + row] = nextValue;
      }
    }
  }

  private addOuterProductSymmetric(
    target: Float64Array,
    vector: Float64Array,
    scale: number,
  ): void {
    for (let row = 0; row < this.dimension; row++) {
      const rowOffset = row * this.dimension;
      const left = vector[row] * scale;
      for (let col = row; col < this.dimension; col++) {
        const value = left * vector[col];
        target[rowOffset + col] += value;
        if (col !== row) {
          target[col * this.dimension + row] += value;
        }
      }
    }
  }

  private samplePopulation(): PolicyParams[] {
    const policies: PolicyParams[] = [];

    for (let sample = 0; sample < this.populationSize; sample++) {
      const z = new Float64Array(this.dimension);
      for (let gene = 0; gene < this.dimension; gene++) {
        z[gene] = this.randomNormal();
      }

      const y = this.transformStandardNormal(z);
      const policy = new Float32Array(this.dimension);
      for (let gene = 0; gene < this.dimension; gene++) {
        policy[gene] = this.mean[gene] + this.sigma * y[gene];
      }

      policies.push(policy);
    }

    return policies;
  }

  private transformStandardNormal(z: Float64Array): Float64Array {
    const transformed = new Float64Array(this.dimension);

    for (let row = 0; row < this.dimension; row++) {
      const rowOffset = row * this.dimension;
      let sum = 0;
      for (let col = 0; col < this.dimension; col++) {
        sum += this.eigenvectors[rowOffset + col] * this.axisScales[col] * z[col];
      }
      transformed[row] = sum;
    }

    return transformed;
  }

  private applyInverseSqrtCovariance(vector: Float64Array): Float64Array {
    const projected = new Float64Array(this.dimension);
    for (let col = 0; col < this.dimension; col++) {
      let sum = 0;
      for (let row = 0; row < this.dimension; row++) {
        sum += this.eigenvectors[row * this.dimension + col] * vector[row];
      }
      projected[col] = sum / Math.max(1e-12, this.axisScales[col]);
    }

    const transformed = new Float64Array(this.dimension);
    for (let row = 0; row < this.dimension; row++) {
      const rowOffset = row * this.dimension;
      let sum = 0;
      for (let col = 0; col < this.dimension; col++) {
        sum += this.eigenvectors[rowOffset + col] * projected[col];
      }
      transformed[row] = sum;
    }

    return transformed;
  }

  private decomposeCovariance(): void {
    const matrix = new Float64Array(this.covariance);
    const eigenvectors = this.createIdentityMatrix();
    const maxSweeps = 12;

    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let maxOffDiagonal = 0;

      for (let p = 0; p < this.dimension - 1; p++) {
        for (let q = p + 1; q < this.dimension; q++) {
          const index = p * this.dimension + q;
          const value = matrix[index];
          const absValue = Math.abs(value);
          if (absValue > maxOffDiagonal) {
            maxOffDiagonal = absValue;
          }

          if (absValue <= 1e-10) {
            continue;
          }

          const pp = p * this.dimension + p;
          const qq = q * this.dimension + q;
          const app = matrix[pp];
          const aqq = matrix[qq];
          const tau = (aqq - app) / (2 * value);
          const signTau = tau >= 0 ? 1 : -1;
          const t = signTau / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
          const cosine = 1 / Math.sqrt(1 + t * t);
          const sine = t * cosine;

          for (let r = 0; r < this.dimension; r++) {
            if (r === p || r === q) {
              continue;
            }

            const rp = r * this.dimension + p;
            const rq = r * this.dimension + q;
            const mrp = matrix[rp];
            const mrq = matrix[rq];
            const nextRp = cosine * mrp - sine * mrq;
            const nextRq = sine * mrp + cosine * mrq;

            matrix[rp] = nextRp;
            matrix[p * this.dimension + r] = nextRp;
            matrix[rq] = nextRq;
            matrix[q * this.dimension + r] = nextRq;
          }

          matrix[pp] =
            cosine * cosine * app -
            2 * sine * cosine * value +
            sine * sine * aqq;
          matrix[qq] =
            sine * sine * app +
            2 * sine * cosine * value +
            cosine * cosine * aqq;
          matrix[p * this.dimension + q] = 0;
          matrix[q * this.dimension + p] = 0;

          for (let row = 0; row < this.dimension; row++) {
            const vp = row * this.dimension + p;
            const vq = row * this.dimension + q;
            const evp = eigenvectors[vp];
            const evq = eigenvectors[vq];
            eigenvectors[vp] = cosine * evp - sine * evq;
            eigenvectors[vq] = sine * evp + cosine * evq;
          }
        }
      }

      if (maxOffDiagonal <= 1e-10) {
        break;
      }
    }

    const order = Array.from({ length: this.dimension }, (_, index) => index).sort(
      (a, b) =>
        matrix[b * this.dimension + b] - matrix[a * this.dimension + a],
    );

    const sortedEigenvectors = new Float64Array(this.dimension * this.dimension);
    const sortedScales = new Float64Array(this.dimension);

    for (let column = 0; column < this.dimension; column++) {
      const sourceColumn = order[column];
      const eigenvalue = Math.max(
        1e-20,
        matrix[sourceColumn * this.dimension + sourceColumn],
      );
      sortedScales[column] = Math.sqrt(eigenvalue);

      let norm = 0;
      for (let row = 0; row < this.dimension; row++) {
        const value = eigenvectors[row * this.dimension + sourceColumn];
        sortedEigenvectors[row * this.dimension + column] = value;
        norm += value * value;
      }

      const invNorm = 1 / Math.sqrt(Math.max(1e-20, norm));
      for (let row = 0; row < this.dimension; row++) {
        sortedEigenvectors[row * this.dimension + column] *= invNorm;
      }
    }

    this.eigenvectors = sortedEigenvectors;
    this.axisScales = sortedScales;
  }

  private createIdentityMatrix(): Float64Array {
    const matrix = new Float64Array(this.dimension * this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      matrix[i * this.dimension + i] = 1;
    }
    return matrix;
  }

  private vectorNorm(vector: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < vector.length; i++) {
      sum += vector[i] * vector[i];
    }
    return Math.sqrt(sum);
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = agent.terminalReason === "collision" ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
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

    const magnitude = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    this.spareNormal = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }
}
