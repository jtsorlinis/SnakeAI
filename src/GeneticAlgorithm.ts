import {
  ELITE_COUNT,
  GRID_SIZE,
  MUTATION_RATE,
  MUTATION_SIZE,
  POLICY_PARAM_COUNT,
  POP_SIZE,
  TOURNAMENT_SIZE,
} from "./config";
import type { Agent, PolicyParams } from "./types";

const TOURNAMENT_POOL_RATIO = 0.4;

export type EvolutionResult = {
  best: Agent;
  nextPolicies: PolicyParams[];
};

export class GeneticAlgorithm {
  public randomPolicy(): PolicyParams {
    const policy = new Float32Array(POLICY_PARAM_COUNT);
    for (let i = 0; i < policy.length; i++) {
      policy[i] = this.randomRange(-1, 1);
    }
    return policy;
  }

  public evolve(population: Agent[]): EvolutionResult {
    for (const agent of population) {
      agent.fitness = this.fitness(agent);
    }

    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const nextPolicies: PolicyParams[] = [];

    for (let i = 0; i < ELITE_COUNT; i++) {
      nextPolicies.push(ranked[i].policy);
    }

    while (nextPolicies.length < POP_SIZE) {
      const parentA = this.pickParent(ranked);
      const parentB = this.pickParent(ranked);
      const child = this.crossover(parentA.policy, parentB.policy);
      this.mutate(child);
      nextPolicies.push(child);
    }

    return {
      best: ranked[0],
      nextPolicies,
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
    const deathPenalty = agent.terminalReason === "collision" ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }

  private crossover(a: PolicyParams, b: PolicyParams): PolicyParams {
    const child = new Float32Array(POLICY_PARAM_COUNT);
    for (let i = 0; i < POLICY_PARAM_COUNT; i++) {
      child[i] = Math.random() < 0.5 ? a[i] : b[i];
    }
    return child;
  }

  private mutate(policy: PolicyParams): void {
    for (let i = 0; i < policy.length; i++) {
      if (Math.random() < MUTATION_RATE) {
        policy[i] += this.randomRange(-MUTATION_SIZE, MUTATION_SIZE);
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
