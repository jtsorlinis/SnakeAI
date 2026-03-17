import {
  ES_PARENT_COUNT,
  ES_SIGMA,
  GRID_SIZE,
  POLICY_PARAM_COUNT,
  POP_SIZE,
} from "./config";
import type { Agent, PolicyParams } from "./types";

export type EvolutionResult = {
  best: Agent;
  nextPolicies: PolicyParams[];
};

export class SimpleEvolutionStrategy {
  private spareNormal: number | null = null;

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
    const parentCount = Math.max(1, Math.min(ES_PARENT_COUNT, ranked.length));
    const parents = ranked.slice(0, parentCount);
    const nextPolicies: PolicyParams[] = [new Float32Array(ranked[0].policy)];

    while (nextPolicies.length < POP_SIZE) {
      const parent = parents[this.randomIndex(parentCount)];
      nextPolicies.push(this.sampleChild(parent.policy));
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

  private sampleChild(parent: PolicyParams): PolicyParams {
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

    const magnitude = Math.sqrt(-2.0 * Math.log(u));
    const angle = 2.0 * Math.PI * v;
    this.spareNormal = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }
}
