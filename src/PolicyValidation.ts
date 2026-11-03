import type { SnakeEnvironment } from "./SnakeEnvironment";
import type { Agent, PolicyParams } from "./types";

export type ValidationResult = {
  fitness: number;
  clearRate: number;
};

// Score only. Clearing the board takes a slow space-filling sweep, and a step
// penalty made that lose to greedy food chasing that traps itself. Hunger
// already ends games that stall.
export function scoreFitness(agent: Agent): number {
  return agent.score + (agent.terminalReason === "solved" ? 1 : 0);
}

// Plays fresh greedy games with a policy. Training fitness is the best of many
// noisy estimates, so trainers check a candidate here before showing it.
export function validatePolicy(
  environment: SnakeEnvironment,
  policy: PolicyParams,
  games: number,
): ValidationResult {
  let total = 0;
  let clears = 0;
  for (let i = 0; i < games; i++) {
    const agent = environment.createAgent(policy);
    while (agent.alive) {
      environment.step(agent);
    }
    total += scoreFitness(agent);
    if (agent.terminalReason === "solved") {
      clears += 1;
    }
  }
  return { fitness: total / games, clearRate: clears / games };
}

export function samePolicy(a: PolicyParams | null, b: PolicyParams): boolean {
  return a !== null && a.every((weight, i) => weight === b[i]);
}
