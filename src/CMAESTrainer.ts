import {
  EVAL_GAMES,
  POP_SIZE,
  RESTART_AFTER,
  RESTART_CHECK_GAMES,
  VALIDATION_GAMES,
} from "./config";
import { CmaEvolutionStrategy } from "./CmaEvolutionStrategy";
import { NeuralNetwork } from "./NeuralNetwork";
import { samePolicy, scoreFitness, validatePolicy } from "./PolicyValidation";
import { ReplayEvaluator } from "./ReplayEvaluator";
import { SnakeEnvironment } from "./SnakeEnvironment";
import type {
  Agent,
  PolicyParams,
  PolicyPlaybackMode,
  TrainerController,
  TrainerState,
} from "./types";

export class CMAESTrainer implements TrainerController {
  private readonly es = new CmaEvolutionStrategy();
  private readonly network = new NeuralNetwork();
  private readonly environment = new SnakeEnvironment(this.network);
  private readonly replay = new ReplayEvaluator(this.environment);

  private population: Agent[] = [];
  private gamesPlayed: number[] = [];
  private fitnessTotals: number[] = [];
  private generation = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessGeneration = 1;
  // When the current search was started, and whether it has passed the
  // board-clearing check.
  private runStartGeneration = 1;
  private runClearsBoard = false;
  private fitnessHistory: number[] = [];

  private showcasePolicy: PolicyParams | null = null;
  private showcaseAgent: Agent | null = null;
  private randomBoardAgents: Agent[] = [];
  private randomBoardGeneration = -1;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.population = [];
    this.generation = 1;
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessGeneration = 1;
    this.runStartGeneration = 1;
    this.runClearsBoard = false;
    this.fitnessHistory = [];
    this.showcasePolicy = null;
    this.showcaseAgent = null;
    this.invalidateRandomBoardAgents();

    this.population = this.es
      .initializePopulation()
      .map((policy) => this.environment.createAgent(policy));
    this.resetEvaluation();

    if (this.population.length > 0) {
      this.setShowcasePolicy(this.population[0].policy);
    }
  }

  public simulate(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      let alive = 0;

      for (let a = 0; a < this.population.length; a++) {
        const agent = this.population[a];
        if (!agent.alive) {
          continue;
        }

        this.environment.step(agent);
        if (agent.alive || this.finishGame(a)) {
          alive += 1;
        }
      }

      if (alive === 0) {
        this.evolve();
      }

      if (this.showcaseAgent) {
        this.environment.step(this.showcaseAgent);
      }

      if (!this.showcaseAgent?.alive && this.showcasePolicy) {
        this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
      }

      this.replay.step();
    }
  }

  public getState(randomBoardCount = 0): TrainerState {
    let alive = 0;
    for (const agent of this.population) {
      if (agent.alive) {
        alive += 1;
      }
    }

    const boardAgent = this.showcaseAgent ?? this.population[0];
    const network = this.showcasePolicy
      ? {
          policy: this.showcasePolicy,
          activations: this.environment.computeNetworkActivations(
            this.showcasePolicy,
            this.showcaseAgent,
          ),
        }
      : { policy: null, activations: null };

    return {
      algorithm: "cmaes",
      boardAgent,
      boardAgents: this.getRandomBoardAgents(randomBoardCount),
      fitnessHistory: this.fitnessHistory,
      iteration: this.generation,
      iterationLabel: "Generation",
      alive,
      batchSize: POP_SIZE,
      batchSizeLabel: "Population",
      bestEverScore: this.bestEverScore,
      bestEverFitness: this.bestEverFitness,
      replay: this.replay.getStats(),
      staleIterations: Math.max(0, this.generation - this.bestFitnessGeneration),
      staleLabel: "Generations since best",
      historyLabel: "CMA-ES best fitness history",
      policySourceLabel: "Best policy",
      playbackMode: "greedy",
      playbackModeEnabled: false,
      network,
    };
  }

  public onGridSizeChanged(): void {
    this.population = this.population.map((agent) =>
      this.environment.createAgent(agent.policy),
    );
    this.resetEvaluation();
    this.runStartGeneration = this.generation;
    this.runClearsBoard = false;

    // Scores from the other board size don't compare, so the records start
    // again from what the shown policy does on this one.
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessGeneration = this.generation;
    this.fitnessHistory = [];

    if (this.showcasePolicy) {
      this.bestEverFitness = validatePolicy(
        this.environment,
        this.showcasePolicy,
        VALIDATION_GAMES,
      ).fitness;
      this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
    } else {
      this.showcaseAgent = null;
    }
    this.replay.setPolicy(this.showcasePolicy);

    this.invalidateRandomBoardAgents();
  }

  public setPlaybackMode(_mode: PolicyPlaybackMode): void {}

  public getPlaybackMode(): PolicyPlaybackMode {
    return "greedy";
  }

  public supportsPlaybackMode(): boolean {
    return false;
  }

  private setShowcasePolicy(policy: PolicyParams): void {
    this.showcasePolicy = new Float32Array(policy);
    this.showcaseAgent = this.environment.createAgent(this.showcasePolicy);
    this.replay.setPolicy(this.showcasePolicy);
  }

  private invalidateRandomBoardAgents(): void {
    this.randomBoardAgents = [];
    this.randomBoardGeneration = -1;
  }

  private sampleAgents(candidates: Agent[], sampleCount: number): Agent[] {
    const count = Math.min(sampleCount, candidates.length);
    const pool = candidates.slice();
    const sampled: Agent[] = [];

    for (let i = 0; i < count; i++) {
      const pick = i + Math.floor(Math.random() * (pool.length - i));
      [pool[i], pool[pick]] = [pool[pick], pool[i]];
      sampled.push(pool[i]);
    }

    return sampled;
  }

  private getRandomBoardAgents(randomBoardCount: number): readonly Agent[] {
    if (randomBoardCount <= 0) {
      return [];
    }

    const requiredCount = Math.min(randomBoardCount, this.population.length);
    const generationChanged = this.randomBoardGeneration !== this.generation;
    const sizeChanged = this.randomBoardAgents.length !== requiredCount;

    if (generationChanged || sizeChanged) {
      const aliveFirst = this.sampleAgents(
        this.population.filter((agent) => agent.alive),
        requiredCount,
      );
      const remaining = requiredCount - aliveFirst.length;
      if (remaining > 0) {
        const aliveSet = new Set(aliveFirst);
        const fallback = this.sampleAgents(
          this.population.filter((agent) => !aliveSet.has(agent)),
          remaining,
        );
        this.randomBoardAgents = aliveFirst.concat(fallback);
      } else {
        this.randomBoardAgents = aliveFirst;
      }
      this.randomBoardGeneration = this.generation;
    }

    const currentlyAlive = new Set(
      this.randomBoardAgents.filter((agent) => agent.alive),
    );
    const replacementPool = this.population.filter(
      (agent) => agent.alive && !currentlyAlive.has(agent),
    );

    for (let i = 0; i < this.randomBoardAgents.length; i++) {
      const current = this.randomBoardAgents[i];
      if (current.alive || replacementPool.length === 0) {
        continue;
      }

      const pick = Math.floor(Math.random() * replacementPool.length);
      const replacement = replacementPool[pick];
      this.randomBoardAgents[i] = replacement;
      currentlyAlive.add(replacement);
      replacementPool[pick] = replacementPool[replacementPool.length - 1];
      replacementPool.pop();
    }

    return this.randomBoardAgents;
  }

  private resetEvaluation(): void {
    this.gamesPlayed = this.population.map(() => 0);
    this.fitnessTotals = this.population.map(() => 0);
  }

  // Records a finished game. Returns true if the agent starts another one.
  private finishGame(index: number): boolean {
    const agent = this.population[index];
    this.bestEverScore = Math.max(this.bestEverScore, agent.score);
    this.fitnessTotals[index] += scoreFitness(agent);
    this.gamesPlayed[index] += 1;

    if (this.gamesPlayed[index] >= EVAL_GAMES) {
      return false;
    }

    this.population[index] = this.environment.createAgent(agent.policy);
    return true;
  }

  private evolve(): void {
    for (let a = 0; a < this.population.length; a++) {
      this.population[a].fitness = this.fitnessTotals[a] / this.gamesPlayed[a];
    }

    const { best, nextPolicies } = this.es.evolve(this.population);

    // The leader's mean is the max of many noisy estimates, so it only takes
    // over the display if it also does well on fresh games. Ties go to the
    // newer leader: fitness caps out once games are won, and a strict
    // comparison would freeze the first lucky perfect run on display.
    if (best.fitness >= this.bestEverFitness) {
      const validated = validatePolicy(
        this.environment,
        best.policy,
        VALIDATION_GAMES,
      );
      if (validated.fitness >= this.bestEverFitness) {
        this.bestEverFitness = validated.fitness;
        this.bestFitnessGeneration = this.generation;
        // Elites carry over, so the same policy can lead again; keep its game
        // going.
        if (!samePolicy(this.showcasePolicy, best.policy)) {
          this.setShowcasePolicy(best.policy);
        }
      }
    }

    this.fitnessHistory.push(best.fitness);
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory.shift();
    }

    // Restart the search if it hasn't learned to clear the board within its
    // budget, judged by one check on fresh games (a running record of noisy
    // scores keeps creeping up, so it can't tell a stuck run from a slow
    // one). The showcase keeps the best policy found so far.
    let policies = nextPolicies;
    if (
      !this.runClearsBoard &&
      this.generation - this.runStartGeneration >= RESTART_AFTER
    ) {
      const check = validatePolicy(
        this.environment,
        best.policy,
        RESTART_CHECK_GAMES,
      );
      this.runClearsBoard = check.clearRate >= 0.5;
      if (!this.runClearsBoard) {
        this.runStartGeneration = this.generation;
        policies = this.es.initializePopulation();
      }
    }

    this.population = policies.map((policy) =>
      this.environment.createAgent(policy),
    );
    this.resetEvaluation();
    this.generation += 1;
    this.invalidateRandomBoardAgents();
  }
}
