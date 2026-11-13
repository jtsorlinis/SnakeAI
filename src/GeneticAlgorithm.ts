import {
  ELITE_COUNT,
  MUTATION_SIGMA,
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

  // Agents arrive with fitness already averaged over their evaluation games.
  public evolve(population: Agent[]): EvolutionResult {
    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const nextPolicies: PolicyParams[] = [];

    for (let i = 0; i < ELITE_COUNT; i++) {
      nextPolicies.push(ranked[i].policy);
    }

    while (nextPolicies.length < POP_SIZE) {
      const child = new Float32Array(this.pickParent(ranked).policy);
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

  // Box-Muller. 1 - random() keeps the log argument above zero.
  private randomNormal(): number {
    const magnitude = Math.sqrt(-2 * Math.log(1 - Math.random()));
    return magnitude * Math.cos(2 * Math.PI * Math.random());
  }

  private mutate(policy: PolicyParams): void {
    for (let i = 0; i < policy.length; i++) {
      policy[i] += this.randomNormal() * MUTATION_SIGMA;
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
