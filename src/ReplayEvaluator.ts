import { REPLAY_AGENT_COUNT } from "./config";
import type { SnakeEnvironment } from "./SnakeEnvironment";
import type { Agent, PolicyParams, ReplayStats } from "./types";

// Replays the showcased policy greedily on fresh boards. "Best score" is the
// luckiest training episode; this shows how the policy does on average.
export class ReplayEvaluator {
  private agents: Agent[] = [];
  private games = 0;
  private totalScore = 0;
  private wins = 0;

  constructor(private readonly environment: SnakeEnvironment) {}

  public setPolicy(policy: PolicyParams | null): void {
    this.agents = [];
    this.games = 0;
    this.totalScore = 0;
    this.wins = 0;

    if (!policy) {
      return;
    }

    for (let i = 0; i < REPLAY_AGENT_COUNT; i++) {
      this.agents.push(this.environment.createAgent(policy));
    }
  }

  public step(): void {
    for (let i = 0; i < this.agents.length; i++) {
      const agent = this.agents[i];
      this.environment.step(agent);
      if (agent.alive) {
        continue;
      }

      this.games += 1;
      this.totalScore += agent.score;
      if (agent.terminalReason === "solved") {
        this.wins += 1;
      }
      this.agents[i] = this.environment.createAgent(agent.policy);
    }
  }

  public getStats(): ReplayStats {
    return {
      games: this.games,
      averageScore: this.games > 0 ? this.totalScore / this.games : 0,
      wins: this.wins,
    };
  }
}
