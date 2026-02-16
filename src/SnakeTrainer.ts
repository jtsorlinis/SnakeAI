import {
  BASE_HUNGER,
  DIRS,
  ELITE_COUNT,
  GENE_COUNT,
  GRID_SIZE,
  HIDDEN_LAYERS,
  HIDDEN_LAYER_UNITS,
  INPUTS,
  MAX_SCORE,
  MUTATION_RATE,
  MUTATION_SIZE,
  OFFSET_H_BIAS,
  OFFSET_HH,
  OFFSET_HO,
  OFFSET_IH,
  OFFSET_O_BIAS,
  OUTPUTS,
  POP_SIZE,
  TOURNAMENT_SIZE,
} from "./config";
import type {
  Agent,
  Genome,
  NetworkActivations,
  Point,
  TrainerState,
} from "./types";

const TOURNAMENT_POOL_RATIO = 0.4;
export class SnakeTrainer {
  private population: Agent[] = [];
  private generation = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessGen = 1;
  private history: number[] = [];

  private showcaseGenome: Genome | null = null;
  private showcaseAgent: Agent | null = null;

  private readonly senseBuffer = new Float32Array(INPUTS);
  private readonly hiddenBuffers = HIDDEN_LAYER_UNITS.map(
    (units) => new Float32Array(units),
  );
  private readonly vizInputs = new Float32Array(INPUTS);
  private readonly vizHidden = HIDDEN_LAYER_UNITS.map(
    (units) => new Float32Array(units),
  );
  private readonly vizOutputs = new Float32Array(OUTPUTS);

  constructor() {
    for (let i = 0; i < POP_SIZE; i++) {
      this.population.push(this.createAgent());
    }
    this.setShowcaseGenome(this.population[0].genome);
  }

  public simulate(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      let alive = 0;

      for (const agent of this.population) {
        if (!agent.alive) {
          continue;
        }

        this.step(agent);
        if (agent.alive) {
          alive += 1;
        }
      }

      if (alive === 0) {
        this.evolve();
      }

      if (this.showcaseAgent) {
        this.step(this.showcaseAgent);
      }

      if (!this.showcaseAgent?.alive && this.showcaseGenome) {
        this.showcaseAgent = this.createAgent(this.showcaseGenome);
      }
    }
  }

  public getState(): TrainerState {
    let alive = 0;
    for (const agent of this.population) {
      if (agent.alive) {
        alive += 1;
      }
    }

    const boardAgent = this.showcaseAgent ?? this.population[0];
    const network = this.showcaseGenome
      ? {
          genome: this.showcaseGenome,
          activations: this.computeNetworkActivations(
            this.showcaseGenome,
            this.showcaseAgent,
          ),
        }
      : { genome: null, activations: null };

    return {
      boardAgent,
      history: this.history,
      generation: this.generation,
      alive,
      populationSize: POP_SIZE,
      bestEverScore: this.bestEverScore,
      bestEverFitness: this.bestEverFitness,
      staleGenerations: Math.max(0, this.generation - this.bestFitnessGen),
      network,
    };
  }

  public onGridSizeChanged(): void {
    this.population = this.population.map((agent) =>
      this.createAgent(agent.genome),
    );

    if (this.showcaseGenome) {
      this.showcaseAgent = this.createAgent(this.showcaseGenome);
    } else {
      this.showcaseAgent = null;
    }
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
  }

  private randomIndex(maxExclusive: number): number {
    return Math.floor(Math.random() * maxExclusive);
  }

  private randomGenome(): Genome {
    const genome = new Float32Array(GENE_COUNT);
    for (let i = 0; i < genome.length; i++) {
      genome[i] = this.randomRange(-1, 1);
    }
    return genome;
  }

  private createAgent(genome?: Genome): Agent {
    const x = Math.floor(GRID_SIZE / 2);
    const y = Math.floor(GRID_SIZE / 2);
    const body = [
      { x, y },
      { x: x - 1, y },
      { x: x - 2, y },
    ];

    return {
      genome: genome ? new Float32Array(genome) : this.randomGenome(),
      body,
      dir: 1,
      food: this.randomFood(body),
      alive: true,
      score: 0,
      steps: 0,
      hunger: BASE_HUNGER,
      fitness: 0,
    };
  }

  private setShowcaseGenome(genome: Genome): void {
    this.showcaseGenome = new Float32Array(genome);
    this.showcaseAgent = this.createAgent(this.showcaseGenome);
  }

  private randomFood(body: Point[]): Point {
    if (body.length >= GRID_SIZE * GRID_SIZE) {
      return { ...body[0] };
    }

    while (true) {
      const point = {
        x: this.randomIndex(GRID_SIZE),
        y: this.randomIndex(GRID_SIZE),
      };
      if (!this.pointInBody(body, point, body.length)) {
        return point;
      }
    }
  }

  private pointInBody(body: Point[], point: Point, len: number): boolean {
    for (let i = 0; i < len; i++) {
      const part = body[i];
      if (part.x === point.x && part.y === point.y) {
        return true;
      }
    }
    return false;
  }

  private outOfBounds(x: number, y: number): boolean {
    return x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE;
  }

  private checkForObstacle(
    head: Point,
    direction: Point,
    snake: Point[],
  ): number {
    const checkX = head.x + direction.x;
    const checkY = head.y + direction.y;

    if (this.outOfBounds(checkX, checkY)) {
      return 1;
    }

    for (let i = 1; i < snake.length; i++) {
      if (snake[i].x === checkX && snake[i].y === checkY) {
        return 1;
      }
    }

    return 0;
  }

  private findTailDistance(
    head: Point,
    direction: Point,
    snake: Point[],
  ): number {
    for (let distance = 1; distance <= GRID_SIZE; distance++) {
      const x = head.x + direction.x * distance;
      const y = head.y + direction.y * distance;

      if (this.outOfBounds(x, y)) {
        return GRID_SIZE + 1;
      }

      for (let i = 1; i < snake.length; i++) {
        if (snake[i].x === x && snake[i].y === y) {
          return distance;
        }
      }
    }

    return GRID_SIZE + 1;
  }

  private senseInto(agent: Agent, target: Float32Array): void {
    const head = agent.body[0];
    const front = DIRS[agent.dir];
    const left = DIRS[(agent.dir + 3) % 4];
    const right = DIRS[(agent.dir + 1) % 4];

    target[0] = this.checkForObstacle(head, front, agent.body);
    target[1] = this.checkForObstacle(head, left, agent.body);
    target[2] = this.checkForObstacle(head, right, agent.body);
    target[3] = 1 / this.findTailDistance(head, front, agent.body);
    target[4] = 1 / this.findTailDistance(head, left, agent.body);
    target[5] = 1 / this.findTailDistance(head, right, agent.body);

    target[6] = agent.food.x > head.x ? 1 : 0;
    target[7] = agent.food.y > head.y ? 1 : 0;
    target[8] = agent.dir === 0 ? 1 : 0;
    target[9] = agent.dir === 1 ? 1 : 0;
    target[10] = agent.dir === 2 ? 1 : 0;
  }

  private runNetwork(
    genome: Genome,
    inputs: Float32Array,
    hiddenTarget: Float32Array[],
    outputTarget?: Float32Array,
  ): number {
    if (HIDDEN_LAYERS > 0) {
      const firstLayerSize = HIDDEN_LAYER_UNITS[0];
      const firstHidden = hiddenTarget[0];
      for (let h = 0; h < firstLayerSize; h++) {
        let sum = genome[OFFSET_H_BIAS + h];
        const wOffset = OFFSET_IH + h * INPUTS;
        for (let i = 0; i < INPUTS; i++) {
          sum += genome[wOffset + i] * inputs[i];
        }
        firstHidden[h] = Math.tanh(sum);
      }

      let hhOffset = OFFSET_HH;
      let biasOffset = OFFSET_H_BIAS + firstLayerSize;
      for (let layer = 1; layer < HIDDEN_LAYERS; layer++) {
        const prev = hiddenTarget[layer - 1];
        const current = hiddenTarget[layer];
        const prevSize = HIDDEN_LAYER_UNITS[layer - 1];
        const currentSize = HIDDEN_LAYER_UNITS[layer];

        for (let h = 0; h < currentSize; h++) {
          let sum = genome[biasOffset + h];
          const wOffset = hhOffset + h * prevSize;
          for (let k = 0; k < prevSize; k++) {
            sum += genome[wOffset + k] * prev[k];
          }
          current[h] = Math.tanh(sum);
        }

        hhOffset += prevSize * currentSize;
        biasOffset += currentSize;
      }
    }

    let bestAction = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    const outputInputs =
      HIDDEN_LAYERS > 0 ? hiddenTarget[HIDDEN_LAYERS - 1] : inputs;
    const outputInputSize =
      HIDDEN_LAYERS > 0 ? HIDDEN_LAYER_UNITS[HIDDEN_LAYERS - 1] : INPUTS;

    for (let output = 0; output < OUTPUTS; output++) {
      let value = genome[OFFSET_O_BIAS + output];
      const wOffset = OFFSET_HO + output * outputInputSize;
      for (let h = 0; h < outputInputSize; h++) {
        value += genome[wOffset + h] * outputInputs[h];
      }

      if (outputTarget) {
        outputTarget[output] = value;
      }

      if (value > bestValue) {
        bestValue = value;
        bestAction = output;
      }
    }

    return bestAction;
  }

  private chooseAction(agent: Agent): number {
    this.senseInto(agent, this.senseBuffer);
    return this.runNetwork(agent.genome, this.senseBuffer, this.hiddenBuffers);
  }

  private step(agent: Agent): void {
    if (!agent.alive) {
      return;
    }

    const action = this.chooseAction(agent);
    switch (action) {
      case 1:
        agent.dir = (agent.dir + 3) % 4;
        break;
      case 2:
        agent.dir = (agent.dir + 1) % 4;
        break;
    }

    const move = DIRS[agent.dir];
    const head = agent.body[0];
    const next = { x: head.x + move.x, y: head.y + move.y };

    agent.steps += 1;
    agent.hunger -= 1;

    if (this.outOfBounds(next.x, next.y)) {
      agent.alive = false;
      return;
    }

    const ate = next.x === agent.food.x && next.y === agent.food.y;
    const len = ate ? agent.body.length : agent.body.length - 1;

    if (this.pointInBody(agent.body, next, len)) {
      agent.alive = false;
      return;
    }

    agent.body.unshift(next);

    if (ate) {
      agent.score += 1;

      if (agent.score >= MAX_SCORE) {
        agent.alive = false;
        return;
      }

      agent.hunger = BASE_HUNGER;
      agent.food = this.randomFood(agent.body);
    } else {
      agent.body.pop();
    }

    if (agent.hunger <= 0) {
      agent.alive = false;
    }
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = !agent.alive && agent.hunger > 0 ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }

  private crossover(a: Genome, b: Genome): Genome {
    const child = new Float32Array(GENE_COUNT);
    for (let i = 0; i < GENE_COUNT; i++) {
      child[i] = Math.random() < 0.5 ? a[i] : b[i];
    }
    return child;
  }

  private mutate(genome: Genome): void {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < MUTATION_RATE) {
        genome[i] += this.randomRange(-MUTATION_SIZE, MUTATION_SIZE);
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

  private evolve(): void {
    for (const agent of this.population) {
      agent.fitness = this.fitness(agent);
    }

    const ranked = [...this.population].sort((a, b) => b.fitness - a.fitness);
    const best = ranked[0];

    this.bestEverScore = Math.max(this.bestEverScore, best.score);
    if (this.generation === 1 || best.fitness > this.bestEverFitness) {
      this.bestEverFitness = best.fitness;
      this.bestFitnessGen = this.generation;
      this.setShowcaseGenome(best.genome);
    }

    this.history.push(best.score);
    if (this.history.length > 240) {
      this.history.shift();
    }

    const next: Agent[] = [];

    for (let i = 0; i < ELITE_COUNT; i++) {
      next.push(this.createAgent(ranked[i].genome));
    }

    while (next.length < POP_SIZE) {
      const parentA = this.pickParent(ranked);
      const parentB = this.pickParent(ranked);
      const child = this.crossover(parentA.genome, parentB.genome);
      this.mutate(child);
      next.push(this.createAgent(child));
    }

    this.population = next;
    this.generation += 1;
  }

  private computeNetworkActivations(
    genome: Genome,
    agent: Agent | null,
  ): NetworkActivations {
    if (agent) {
      this.senseInto(agent, this.vizInputs);
    } else {
      this.vizInputs.fill(0);
    }

    const best = this.runNetwork(
      genome,
      this.vizInputs,
      this.vizHidden,
      this.vizOutputs,
    );

    return {
      input: this.vizInputs,
      hidden: this.vizHidden,
      output: this.vizOutputs,
      best,
    };
  }
}
