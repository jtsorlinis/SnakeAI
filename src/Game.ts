type Point = { x: number; y: number };

type Genome = Float32Array;

type Agent = {
  genome: Genome;
  body: Point[];
  dir: number;
  food: Point;
  alive: boolean;
  score: number;
  steps: number;
  hunger: number;
  fitness: number;
};

const GRID_SIZE = 18;
const TILE_SIZE = 24;
const BOARD_SIZE = GRID_SIZE * TILE_SIZE;
const CHART_WIDTH = 300;
const CHART_HEIGHT = 140;

const POP_SIZE = 180;
const ELITE_COUNT = 8;
const TOURNAMENT_SIZE = 4;
const MUTATION_RATE = 0.08;
const MUTATION_SIZE = 0.35;
const BASE_HUNGER = GRID_SIZE * 2;
const NORMAL_STEPS_PER_SECOND = 30;
const TURBO_TIME_BUDGET_MS = 12;

const INPUTS = 7;
const OUTPUTS = 3;
const GENE_COUNT = INPUTS * OUTPUTS;

const DIRS: ReadonlyArray<Point> = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
];

export class Game {
  private readonly board: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly chart: HTMLCanvasElement;
  private readonly chartCtx: CanvasRenderingContext2D;
  private readonly stats: HTMLElement;
  private readonly turboToggle: HTMLInputElement;

  private population: Agent[] = [];
  private generation = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessGen = 1;
  private history: number[] = [];
  private turboMode = false;
  private stepBudget = 0;
  private lastFrameTime = 0;
  private showcaseGenome: Genome | null = null;
  private showcaseAgent: Agent | null = null;

  constructor() {
    this.board = this.getCanvas("gameCanvas");
    this.chart = this.getCanvas("chartCanvas");
    this.stats = this.getElement("stats");
    this.turboToggle = this.getInput("turboToggle");

    this.ctx = this.board.getContext("2d") as CanvasRenderingContext2D;
    this.chartCtx = this.chart.getContext("2d") as CanvasRenderingContext2D;

    this.board.width = BOARD_SIZE;
    this.board.height = BOARD_SIZE;
    this.chart.width = CHART_WIDTH;
    this.chart.height = CHART_HEIGHT;

    this.turboMode = this.turboToggle.checked;
    this.turboToggle.addEventListener("change", () => {
      this.turboMode = this.turboToggle.checked;
      this.stepBudget = 0;
    });

    for (let i = 0; i < POP_SIZE; i++) {
      this.population.push(this.createAgent());
    }

    this.setShowcaseGenome(this.population[0].genome);

    requestAnimationFrame(this.loop);
  }

  private getCanvas(id: string): HTMLCanvasElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLCanvasElement)) {
      throw new Error(`Missing canvas: ${id}`);
    }
    return element;
  }

  private getElement(id: string): HTMLElement {
    const element = document.getElementById(id);
    if (!element) {
      throw new Error(`Missing element: ${id}`);
    }
    return element;
  }

  private getInput(id: string): HTMLInputElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLInputElement)) {
      throw new Error(`Missing input: ${id}`);
    }
    return element;
  }

  private randomGenome(): Genome {
    const genome = new Float32Array(GENE_COUNT);
    for (let i = 0; i < genome.length; i++) {
      genome[i] = Math.random() * 2 - 1;
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
    while (true) {
      const point = {
        x: Math.floor(Math.random() * GRID_SIZE),
        y: Math.floor(Math.random() * GRID_SIZE),
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

  private isBlocked(agent: Agent, delta: Point): number {
    const head = agent.body[0];
    const next = { x: head.x + delta.x, y: head.y + delta.y };

    if (
      next.x < 0 ||
      next.x >= GRID_SIZE ||
      next.y < 0 ||
      next.y >= GRID_SIZE
    ) {
      return 1;
    }

    const willEat = next.x === agent.food.x && next.y === agent.food.y;
    const len = willEat ? agent.body.length : agent.body.length - 1;

    return this.pointInBody(agent.body, next, len) ? 1 : 0;
  }

  private sense(agent: Agent): number[] {
    const head = agent.body[0];
    const forward = DIRS[agent.dir];
    const left = DIRS[(agent.dir + 3) % 4];
    const right = DIRS[(agent.dir + 1) % 4];

    const dx = agent.food.x - head.x;
    const dy = agent.food.y - head.y;

    return [
      this.isBlocked(agent, forward),
      this.isBlocked(agent, left),
      this.isBlocked(agent, right),
      (dx * forward.x + dy * forward.y) / GRID_SIZE,
      (dx * left.x + dy * left.y) / GRID_SIZE,
      (dx * right.x + dy * right.y) / GRID_SIZE,
      1,
    ];
  }

  private chooseAction(genome: Genome, inputs: number[]): number {
    let bestValue = Number.NEGATIVE_INFINITY;
    let bestAction = 0;

    for (let output = 0; output < OUTPUTS; output++) {
      let value = 0;
      const offset = output * INPUTS;

      for (let i = 0; i < INPUTS; i++) {
        value += genome[offset + i] * inputs[i];
      }

      if (value > bestValue) {
        bestValue = value;
        bestAction = output;
      }
    }

    return bestAction;
  }

  private step(agent: Agent): void {
    if (!agent.alive) {
      return;
    }

    const action = this.chooseAction(agent.genome, this.sense(agent));
    if (action === 1) {
      agent.dir = (agent.dir + 3) % 4;
    } else if (action === 2) {
      agent.dir = (agent.dir + 1) % 4;
    }

    const move = DIRS[agent.dir];
    const head = agent.body[0];
    const next = { x: head.x + move.x, y: head.y + move.y };

    agent.steps += 1;
    agent.hunger -= 1;

    if (
      next.x < 0 ||
      next.x >= GRID_SIZE ||
      next.y < 0 ||
      next.y >= GRID_SIZE
    ) {
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
      agent.hunger = BASE_HUNGER + agent.score * 25;
      agent.food = this.randomFood(agent.body);
    } else {
      agent.body.pop();
    }

    if (agent.hunger <= 0) {
      agent.alive = false;
    }
  }

  private fitness(agent: Agent): number {
    const head = agent.body[0];
    const dist = Math.abs(head.x - agent.food.x) + Math.abs(head.y - agent.food.y);
    return agent.score * agent.score * 200 + agent.steps - dist;
  }

  private crossover(a: Genome, b: Genome): Genome {
    const child = new Float32Array(GENE_COUNT);
    for (let i = 0; i < GENE_COUNT; i++) {
      child[i] = Math.random() < 0.5 ? a[i] : b[i];
    }
    return child;
  }

  private mutate(genome: Genome, rate: number, amount: number): void {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < rate) {
        genome[i] += (Math.random() * 2 - 1) * amount;
      }
    }
  }

  private pickParent(ranked: Agent[]): Agent {
    const poolSize = Math.max(2, Math.floor(ranked.length * 0.4));
    let winner = ranked[Math.floor(Math.random() * poolSize)];

    for (let i = 1; i < TOURNAMENT_SIZE; i++) {
      const challenger = ranked[Math.floor(Math.random() * poolSize)];
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
      this.mutate(child, MUTATION_RATE, MUTATION_SIZE);
      next.push(this.createAgent(child));
    }

    this.population = next;
    this.generation += 1;
  }

  private simulate(stepCount: number): void {
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

  private drawBoard(agent: Agent): void {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, BOARD_SIZE, BOARD_SIZE);

    this.ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    this.ctx.lineWidth = 1.1;

    for (let p = 0; p <= BOARD_SIZE; p += TILE_SIZE) {
      this.ctx.beginPath();
      this.ctx.moveTo(p + 0.5, 0);
      this.ctx.lineTo(p + 0.5, BOARD_SIZE);
      this.ctx.stroke();

      this.ctx.beginPath();
      this.ctx.moveTo(0, p + 0.5);
      this.ctx.lineTo(BOARD_SIZE, p + 0.5);
      this.ctx.stroke();
    }

    this.ctx.fillStyle = "#f44336";
    this.ctx.fillRect(
      agent.food.x * TILE_SIZE + 3,
      agent.food.y * TILE_SIZE + 3,
      TILE_SIZE - 6,
      TILE_SIZE - 6,
    );

    for (let i = agent.body.length - 1; i >= 0; i--) {
      const part = agent.body[i];
      this.ctx.fillStyle = i === 0 ? "#4caf50" : "#43a047";
      this.ctx.fillRect(
        part.x * TILE_SIZE + 2,
        part.y * TILE_SIZE + 2,
        TILE_SIZE - 4,
        TILE_SIZE - 4,
      );
    }
  }

  private drawHistory(): void {
    this.chartCtx.fillStyle = "#000";
    this.chartCtx.fillRect(0, 0, CHART_WIDTH, CHART_HEIGHT);

    this.chartCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    this.chartCtx.lineWidth = 1;

    for (let i = 1; i <= 3; i++) {
      const y = (CHART_HEIGHT * i) / 4;
      this.chartCtx.beginPath();
      this.chartCtx.moveTo(0, y + 0.5);
      this.chartCtx.lineTo(CHART_WIDTH, y + 0.5);
      this.chartCtx.stroke();
    }

    if (this.history.length < 2) {
      return;
    }

    let maxScore = 1;
    for (const score of this.history) {
      if (score > maxScore) {
        maxScore = score;
      }
    }

    this.chartCtx.strokeStyle = "#646cff";
    this.chartCtx.lineWidth = 2;
    this.chartCtx.beginPath();

    for (let i = 0; i < this.history.length; i++) {
      const x = 10 + (i / (this.history.length - 1)) * (CHART_WIDTH - 20);
      const y = CHART_HEIGHT - 10 - (this.history[i] / maxScore) * (CHART_HEIGHT - 20);
      if (i === 0) {
        this.chartCtx.moveTo(x, y);
      } else {
        this.chartCtx.lineTo(x, y);
      }
    }

    this.chartCtx.stroke();

    this.chartCtx.fillStyle = "rgba(255, 255, 255, 0.75)";
    this.chartCtx.font = "12px IBM Plex Mono, monospace";
    this.chartCtx.fillText(`Gen best score history (max ${maxScore})`, 10, 16);
  }

  private updateStats(): void {
    let alive = 0;

    for (const agent of this.population) {
      if (agent.alive) {
        alive += 1;
      }
    }

    this.stats.innerHTML = [
      `Generation: <strong>${this.generation}</strong>`,
      `Alive: ${alive}/${POP_SIZE}`,
      `Best score: ${this.bestEverScore}`,
      `Best fitness: ${this.bestEverFitness.toFixed(1)}`,
      `Stale: ${Math.max(0, this.generation - this.bestFitnessGen)}`,
    ].join("<br>");
  }

  private render(): void {
    const view = this.showcaseAgent ?? this.population[0];
    this.drawBoard(view);
    this.drawHistory();
    this.updateStats();
  }

  private loop = (time: number): void => {
    if (this.lastFrameTime === 0) {
      this.lastFrameTime = time;
    }
    const deltaSeconds = Math.max(0, (time - this.lastFrameTime) / 1000);
    this.lastFrameTime = time;

    if (this.turboMode) {
      const start = performance.now();
      do {
        this.simulate(1);
      } while (performance.now() - start < TURBO_TIME_BUDGET_MS);
    } else {
      this.stepBudget += deltaSeconds * NORMAL_STEPS_PER_SECOND;
      const stepCount = Math.floor(this.stepBudget);
      if (stepCount > 0) {
        this.stepBudget -= stepCount;
        this.simulate(stepCount);
      }
    }
    this.render();
    requestAnimationFrame(this.loop);
  };
}
