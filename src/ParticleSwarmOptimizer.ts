import { GRID_SIZE, POLICY_PARAM_COUNT, POP_SIZE } from "./config";
import type { Agent, PolicyParams } from "./types";

const PSO_MIN_POSITION = -2;
const PSO_MAX_POSITION = 2;
const PSO_STAGNATION_LIMIT = 10;
const PSO_RESTART_SIGMA = 0.35;
const PSO_ELITE_COUNT = 12;
const PSO_PARENT_POOL_COUNT = 24;
const PSO_BASE_MUTATION_SIGMA = 0.12;
const PSO_MUTATION_SIGMA_STEP = 0.015;
const PSO_MAX_MUTATION_SIGMA = 0.28;
const PSO_MIN_MUTATION_SIGMA = 0.05;
const PSO_RESTART_PROBABILITY = 0.08;
const PSO_GLOBAL_MIX_MIN = 0.15;
const PSO_GLOBAL_MIX_MAX = 0.35;

export type PSOEvolutionResult = {
  best: Agent;
  nextPolicies: PolicyParams[];
};

type ParticleState = {
  position: PolicyParams;
  velocity: PolicyParams;
  bestPosition: PolicyParams;
  bestFitness: number;
  stagnantGenerations: number;
};

function clonePolicy(policy: PolicyParams): PolicyParams {
  return new Float32Array(policy) as unknown as PolicyParams;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export class ParticleSwarmOptimizer {
  private readonly dimension = POLICY_PARAM_COUNT;
  private readonly swarmSize = POP_SIZE;
  private particles: ParticleState[] = [];
  private globalBestPosition: PolicyParams = new Float32Array(
    this.dimension,
  ) as unknown as PolicyParams;
  private globalBestFitness = Number.NEGATIVE_INFINITY;
  private generation = 0;
  private spareNormal: number | null = null;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.particles = [];
    this.globalBestPosition = new Float32Array(
      this.dimension,
    ) as unknown as PolicyParams;
    this.globalBestFitness = Number.NEGATIVE_INFINITY;
    this.generation = 0;
    this.spareNormal = null;

    for (let i = 0; i < this.swarmSize; i++) {
      const position = this.randomPolicy();
      const velocity = new Float32Array(this.dimension) as unknown as PolicyParams;
      for (let gene = 0; gene < this.dimension; gene++) {
        velocity[gene] = this.randomRange(-0.05, 0.05);
      }

      this.particles.push({
        position,
        velocity,
        bestPosition: clonePolicy(position),
        bestFitness: Number.NEGATIVE_INFINITY,
        stagnantGenerations: 0,
      });
    }
  }

  public randomPolicy(): PolicyParams {
    const policy = new Float32Array(this.dimension);
    for (let i = 0; i < policy.length; i++) {
      policy[i] = this.randomRange(-1, 1);
    }
    return policy;
  }

  public initializePopulation(): PolicyParams[] {
    return this.particles.map((particle) => clonePolicy(particle.position));
  }

  public evolve(population: Agent[]): PSOEvolutionResult {
    this.generation += 1;
    for (const agent of population) {
      agent.fitness = this.fitness(agent);
    }

    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const best = ranked[0];

    for (let i = 0; i < population.length && i < this.particles.length; i++) {
      const particle = this.particles[i];
      const agent = population[i];

      if (agent.fitness > particle.bestFitness) {
        particle.bestFitness = agent.fitness;
        particle.bestPosition = clonePolicy(agent.policy);
        particle.stagnantGenerations = 0;
      } else {
        particle.stagnantGenerations += 1;
      }
    }

    if (best.fitness >= this.globalBestFitness) {
      this.globalBestFitness = best.fitness;
      this.globalBestPosition = clonePolicy(best.policy);
    }

    const nextPolicies: PolicyParams[] = new Array(this.particles.length);
    const eliteCount = Math.min(PSO_ELITE_COUNT, this.particles.length);
    for (let i = 0; i < eliteCount; i++) {
      const elite = ranked[i] ?? ranked[0];
      nextPolicies[i] = clonePolicy(elite.policy);
    }

    for (let i = 0; i < this.particles.length; i++) {
      const particle = this.particles[i];

      if (i < eliteCount) {
        particle.position = clonePolicy(nextPolicies[i]);
        particle.velocity.fill(0);
        particle.stagnantGenerations = 0;
        continue;
      }

      const parentPoolCount = Math.min(PSO_PARENT_POOL_COUNT, ranked.length);
      const mate = ranked[this.pickRankedIndex(parentPoolCount)];
      const matePolicy = mate.policy;

      if (
        particle.stagnantGenerations >= PSO_STAGNATION_LIMIT ||
        Math.random() < PSO_RESTART_PROBABILITY
      ) {
        this.restartParticle(particle);
        nextPolicies[i] = clonePolicy(particle.position);
        continue;
      }

      const mutationSigma = clamp(
        PSO_BASE_MUTATION_SIGMA +
          Math.max(0, particle.stagnantGenerations) * PSO_MUTATION_SIGMA_STEP -
          Math.min(0.03, this.generation / 15000),
        PSO_MIN_MUTATION_SIGMA,
        PSO_MAX_MUTATION_SIGMA,
      );
      const globalPull = clamp(
        0.22 + this.generation / 20000,
        PSO_GLOBAL_MIX_MIN,
        PSO_GLOBAL_MIX_MAX,
      );

      for (let gene = 0; gene < this.dimension; gene++) {
        const currentBest = matePolicy[gene];
        const globalBest = this.globalBestPosition[gene];
        const blended =
          currentBest +
          globalPull * (globalBest - currentBest) +
          this.randomNormal() * mutationSigma;

        const nextValue = clamp(
          blended,
          PSO_MIN_POSITION,
          PSO_MAX_POSITION,
        );

        particle.velocity[gene] = nextValue - particle.position[gene];
        particle.position[gene] = nextValue;
      }

      nextPolicies[i] = clonePolicy(particle.position);
    }

    return {
      best,
      nextPolicies,
    };
  }

  private pickRankedIndex(poolSize: number): number {
    if (poolSize <= 1) {
      return 0;
    }

    const biased = Math.pow(Math.random(), 2);
    return Math.min(poolSize - 1, Math.floor(biased * poolSize));
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
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

  private restartParticle(particle: ParticleState): void {
    for (let gene = 0; gene < this.dimension; gene++) {
      particle.position[gene] = clamp(
        this.globalBestPosition[gene] +
          this.randomNormal() * PSO_RESTART_SIGMA,
        PSO_MIN_POSITION,
        PSO_MAX_POSITION,
      );
      particle.velocity[gene] = this.randomRange(-0.1, 0.1);
    }
    particle.stagnantGenerations = 0;
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = agent.terminalReason === "collision" ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }
}
