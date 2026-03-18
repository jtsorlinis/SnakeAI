import { MAX_SCORE } from "./config";
import { SnakeTrainer } from "./SnakeTrainer";
import type { TrainerAlgorithm } from "./types";

declare const process: { argv: string[] };

const BLACK_BOX_ALGORITHMS: TrainerAlgorithm[] = [
  "ga",
  "es",
  "cmaes",
  "pso",
  "openai-es",
];

type BenchmarkResult = {
  algorithm: TrainerAlgorithm;
  seed: number;
  targetGenerations: number;
  completedGenerations: number;
  bestEverScore: number;
  bestEverFitness: number;
  solveGeneration: number | null;
  wallTimeMs: number;
  simulationSteps: number;
  scoreMilestones: Record<string, number>;
};

type BenchmarkOptions = {
  algorithms: TrainerAlgorithm[];
  generations: number;
  seeds: number[];
  stepChunk: number;
};

function parseArgs(argv: readonly string[]): BenchmarkOptions {
  const values = new Map<string, string>();

  for (const arg of argv) {
    if (!arg.startsWith("--")) {
      continue;
    }

    const eq = arg.indexOf("=");
    if (eq === -1) {
      values.set(arg.slice(2), "true");
    } else {
      values.set(arg.slice(2, eq), arg.slice(eq + 1));
    }
  }

  const algorithms = parseAlgorithms(values.get("algorithms"));
  const generations = parsePositiveInt(values.get("generations"), 2000);
  const seeds = parseSeeds(values.get("seeds"));
  const stepChunk = parsePositiveInt(values.get("step-chunk"), 100);

  return { algorithms, generations, seeds, stepChunk };
}

function parseAlgorithms(value: string | undefined): TrainerAlgorithm[] {
  if (!value) {
    return BLACK_BOX_ALGORITHMS;
  }

  const parsed = value
    .split(",")
    .map((item) => item.trim())
    .filter((item): item is TrainerAlgorithm =>
      BLACK_BOX_ALGORITHMS.includes(item as TrainerAlgorithm),
    );

  return parsed.length > 0 ? parsed : BLACK_BOX_ALGORITHMS;
}

function parseSeeds(value: string | undefined): number[] {
  if (!value) {
    return [1];
  }

  const parsed = value
    .split(",")
    .map((item) => Number.parseInt(item.trim(), 10))
    .filter((item) => Number.isFinite(item));

  return parsed.length > 0 ? parsed : [1];
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }

  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  if (state === 0) {
    state = 1;
  }

  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function milestoneSchedule(generations: number): number[] {
  const defaults = [50, 100, 250, 500, 1000, 2000];
  const filtered = defaults.filter((milestone) => milestone <= generations);
  if (!filtered.includes(generations)) {
    filtered.push(generations);
  }
  return filtered;
}

function runBenchmark(
  algorithm: TrainerAlgorithm,
  seed: number,
  targetGenerations: number,
  stepChunk: number,
): BenchmarkResult {
  const originalRandom = Math.random;
  const milestones = milestoneSchedule(targetGenerations);
  const scoreMilestones: Record<string, number> = {};

  Math.random = createSeededRandom(seed);

  try {
    const trainer = new SnakeTrainer();
    trainer.setAlgorithm(algorithm);

    let state = trainer.getState();
    let solveGeneration: number | null =
      state.bestEverScore >= MAX_SCORE ? 0 : null;
    let simulationSteps = 0;
    const startedAt = Date.now();

    while (state.iteration <= targetGenerations) {
      trainer.simulate(stepChunk);
      simulationSteps += stepChunk;
      state = trainer.getState();

      const completedGenerations = Math.max(0, state.iteration - 1);
      if (solveGeneration === null && state.bestEverScore >= MAX_SCORE) {
        solveGeneration = completedGenerations;
      }

      for (const milestone of milestones) {
        if (
          completedGenerations >= milestone &&
          scoreMilestones[String(milestone)] === undefined
        ) {
          scoreMilestones[String(milestone)] = state.bestEverScore;
        }
      }
    }

    return {
      algorithm,
      seed,
      targetGenerations,
      completedGenerations: Math.max(0, state.iteration - 1),
      bestEverScore: state.bestEverScore,
      bestEverFitness: state.bestEverFitness,
      solveGeneration,
      wallTimeMs: Date.now() - startedAt,
      simulationSteps,
      scoreMilestones,
    };
  } finally {
    Math.random = originalRandom;
  }
}

function summarize(results: readonly BenchmarkResult[]): {
  perRun: readonly BenchmarkResult[];
  byAlgorithm: Record<string, unknown>;
} {
  const byAlgorithm: Record<string, BenchmarkResult[]> = {};

  for (const result of results) {
    const key = result.algorithm;
    const bucket = byAlgorithm[key] ?? [];
    bucket.push(result);
    byAlgorithm[key] = bucket;
  }

  const aggregate: Record<string, unknown> = {};
  for (const [algorithm, runs] of Object.entries(byAlgorithm)) {
    const solveGenerations = runs
      .map((run) => run.solveGeneration)
      .filter((value): value is number => value !== null);
    const avg = (values: readonly number[]): number =>
      values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);

    aggregate[algorithm] = {
      runs: runs.length,
      solvedRuns: solveGenerations.length,
      avgBestEverScore: avg(runs.map((run) => run.bestEverScore)),
      avgBestEverFitness: avg(runs.map((run) => run.bestEverFitness)),
      avgWallTimeMs: avg(runs.map((run) => run.wallTimeMs)),
      avgSimulationSteps: avg(runs.map((run) => run.simulationSteps)),
      avgSolveGeneration:
        solveGenerations.length > 0 ? avg(solveGenerations) : null,
      scoreMilestones: Object.fromEntries(
        Object.keys(runs[0]?.scoreMilestones ?? {}).map((milestone) => [
          milestone,
          avg(
            runs.map((run) => run.scoreMilestones[milestone] ?? run.bestEverScore),
          ),
        ]),
      ),
    };
  }

  return {
    perRun: results,
    byAlgorithm: aggregate,
  };
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const results: BenchmarkResult[] = [];

  for (const algorithm of options.algorithms) {
    for (const seed of options.seeds) {
      results.push(
        runBenchmark(algorithm, seed, options.generations, options.stepChunk),
      );
    }
  }

  console.log(
    JSON.stringify(
      {
        options,
        maxScore: MAX_SCORE,
        ...summarize(results),
      },
      null,
      2,
    ),
  );
}

main();
