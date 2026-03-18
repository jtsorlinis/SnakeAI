import {
  GRID_SIZE,
  POLICY_PARAM_COUNT,
} from "./config";
import type { Agent, PolicyParams } from "./types";

export const OPENAI_ES_POPULATION_SIZE = 180;
export const OPENAI_ES_SIGMA = 0.12;
export const OPENAI_ES_LEARNING_RATE = 0.03;
export const OPENAI_ES_WEIGHT_DECAY = 0;
export const OPENAI_ES_GRADIENT_CLIP = 1.5;

const EPSILON = 1e-8;
export type EvolutionResult = {
  best: Agent;
  nextPolicies: PolicyParams[];
};

function fitness(agent: Agent): number {
  const foodReward = agent.score;
  const deathPenalty = agent.terminalReason === "collision" ? 1 : 0;
  const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
  return foodReward - deathPenalty - stepPenalty;
}

class AdamOptimizer {
  private readonly m: Float32Array;
  private readonly v: Float32Array;
  private beta1Power = 1;
  private beta2Power = 1;

  constructor(
    size: number,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly epsilon = 1e-8,
  ) {
    this.m = new Float32Array(size);
    this.v = new Float32Array(size);
  }

  public reset(): void {
    this.m.fill(0);
    this.v.fill(0);
    this.beta1Power = 1;
    this.beta2Power = 1;
  }

  public step(
    params: Float32Array,
    gradient: Float32Array,
    learningRate: number,
  ): void {
    this.beta1Power *= this.beta1;
    this.beta2Power *= this.beta2;

    const invBias1 = 1 / (1 - this.beta1Power);
    const invBias2 = 1 / (1 - this.beta2Power);

    for (let i = 0; i < params.length; i++) {
      const g = gradient[i];
      const m = this.beta1 * this.m[i] + (1 - this.beta1) * g;
      const v = this.beta2 * this.v[i] + (1 - this.beta2) * g * g;
      this.m[i] = m;
      this.v[i] = v;

      const mHat = m * invBias1;
      const vHat = v * invBias2;
      params[i] -= learningRate * (mHat / (Math.sqrt(vHat) + this.epsilon));
    }
  }
}

export class OpenAIEvolutionStrategy {
  private readonly mean = new Float32Array(POLICY_PARAM_COUNT);
  private readonly optimizer = new AdamOptimizer(POLICY_PARAM_COUNT);
  private readonly gradient = new Float32Array(POLICY_PARAM_COUNT);

  private readonly sampledPolicies: PolicyParams[] = [];
  private readonly sampledNoises: Float32Array[] = [];
  private spareNormal: number | null = null;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.randomize(this.mean);
    this.optimizer.reset();
    this.sampledPolicies.length = 0;
    this.sampledNoises.length = 0;
  }

  public getPolicyParams(): PolicyParams {
    return this.mean;
  }

  public randomPolicy(): PolicyParams {
    const policy = new Float32Array(POLICY_PARAM_COUNT);
    this.randomize(policy);
    return policy;
  }

  public samplePopulation(): PolicyParams[] {
    this.sampledPolicies.length = 0;
    this.sampledNoises.length = 0;

    const pairCount = Math.floor(OPENAI_ES_POPULATION_SIZE / 2);
    for (let pair = 0; pair < pairCount; pair++) {
      const noise = this.sampleNoise();
      const plus = new Float32Array(POLICY_PARAM_COUNT);
      const minus = new Float32Array(POLICY_PARAM_COUNT);

      for (let i = 0; i < POLICY_PARAM_COUNT; i++) {
        const delta = OPENAI_ES_SIGMA * noise[i];
        plus[i] = this.mean[i] + delta;
        minus[i] = this.mean[i] - delta;
      }

      this.sampledPolicies.push(plus, minus);
      this.sampledNoises.push(noise);
    }

    if (this.sampledPolicies.length < OPENAI_ES_POPULATION_SIZE) {
      this.sampledPolicies.push(new Float32Array(this.mean));
    }

    return this.sampledPolicies;
  }

  public evolve(population: Agent[]): EvolutionResult {
    for (const agent of population) {
      agent.fitness = fitness(agent);
    }

    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const best = ranked[0];
    this.gradient.fill(0);
    let pairScaleSum = 0;
    let pairScaleSquaredSum = 0;
    const pairCount = this.sampledNoises.length;
    const pairScales = new Float32Array(pairCount);

    for (let pair = 0; pair < pairCount; pair++) {
      const plusFitness = population[pair * 2]?.fitness ?? 0;
      const minusFitness = population[pair * 2 + 1]?.fitness ?? 0;
      const pairScale = plusFitness - minusFitness;
      pairScales[pair] = pairScale;
      pairScaleSum += pairScale;
      pairScaleSquaredSum += pairScale * pairScale;
    }

    const pairMean = pairScaleSum / Math.max(1, pairCount);
    const pairVariance =
      pairScaleSquaredSum / Math.max(1, pairCount) - pairMean * pairMean;
    const pairStd = Math.sqrt(Math.max(EPSILON, pairVariance));

    for (let pair = 0; pair < pairCount; pair++) {
      const normalizedScale = (pairScales[pair] - pairMean) / pairStd;
      const noise = this.sampledNoises[pair];
      for (let p = 0; p < POLICY_PARAM_COUNT; p++) {
        this.gradient[p] -= normalizedScale * noise[p];
      }
    }

    const scale = 1 / (Math.max(1, pairCount) * OPENAI_ES_SIGMA);
    for (let p = 0; p < POLICY_PARAM_COUNT; p++) {
      this.gradient[p] = this.gradient[p] * scale + OPENAI_ES_WEIGHT_DECAY * this.mean[p];
    }
    this.clipGradient(this.gradient, OPENAI_ES_GRADIENT_CLIP);

    this.optimizer.step(this.mean, this.gradient, OPENAI_ES_LEARNING_RATE);

    return {
      best,
      nextPolicies: this.samplePopulation(),
    };
  }

  private randomize(target: Float32Array): void {
    for (let i = 0; i < target.length; i++) {
      target[i] = this.randomRange(-1, 1);
    }
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
  }

  private sampleNoise(): Float32Array {
    const noise = new Float32Array(POLICY_PARAM_COUNT);
    for (let i = 0; i < noise.length; i++) {
      noise[i] = this.randomNormal();
    }
    return noise;
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

    const magnitude = Math.sqrt(-2.0 * Math.log(u));
    const angle = 2.0 * Math.PI * v;
    this.spareNormal = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }

  private clipGradient(gradient: Float32Array, maxNorm: number): void {
    if (maxNorm <= 0) {
      return;
    }

    let normSquared = 0;
    for (let i = 0; i < gradient.length; i++) {
      normSquared += gradient[i] * gradient[i];
    }

    const norm = Math.sqrt(normSquared);
    if (norm <= maxNorm || norm < EPSILON) {
      return;
    }

    const scale = maxNorm / norm;
    for (let i = 0; i < gradient.length; i++) {
      gradient[i] *= scale;
    }
  }
}
