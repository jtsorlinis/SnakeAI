type Point = { x: number; y: number };

type Genome = Float32Array;
type NetEdge = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  weight: number;
  label: string;
};

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
const NET_WIDTH = 480;
const NET_HEIGHT = 432;
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
const HIDDEN = 8;
const OUTPUTS = 3;
const IH_COUNT = INPUTS * HIDDEN;
const H_BIAS_COUNT = HIDDEN;
const HO_COUNT = HIDDEN * OUTPUTS;
const O_BIAS_COUNT = OUTPUTS;
const OFFSET_IH = 0;
const OFFSET_H_BIAS = OFFSET_IH + IH_COUNT;
const OFFSET_HO = OFFSET_H_BIAS + H_BIAS_COUNT;
const OFFSET_O_BIAS = OFFSET_HO + HO_COUNT;
const GENE_COUNT = OFFSET_O_BIAS + O_BIAS_COUNT;
const NET_HOVER_RADIUS = 8;

const DIRS: ReadonlyArray<Point> = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
];

export class Game {
  private readonly net: HTMLCanvasElement;
  private readonly netCtx: CanvasRenderingContext2D;
  private readonly board: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly chart: HTMLCanvasElement;
  private readonly chartCtx: CanvasRenderingContext2D;
  private readonly stats: HTMLElement;
  private readonly turboToggle: HTMLInputElement;
  private readonly networkToggle: HTMLInputElement;

  private population: Agent[] = [];
  private generation = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessGen = 1;
  private history: number[] = [];
  private turboMode = false;
  private showNetwork = true;
  private stepBudget = 0;
  private lastFrameTime = 0;
  private showcaseGenome: Genome | null = null;
  private showcaseAgent: Agent | null = null;
  private netMouse: Point | null = null;
  private readonly hiddenBuffer = new Float32Array(HIDDEN);
  private readonly vizInputs = new Float32Array(INPUTS);
  private readonly vizHidden = new Float32Array(HIDDEN);
  private readonly vizOutputs = new Float32Array(OUTPUTS);

  constructor() {
    this.net = this.getCanvas("netCanvas");
    this.board = this.getCanvas("gameCanvas");
    this.chart = this.getCanvas("chartCanvas");
    this.stats = this.getElement("stats");
    this.turboToggle = this.getInput("turboToggle");
    this.networkToggle = this.getInput("networkToggle");

    this.netCtx = this.net.getContext("2d") as CanvasRenderingContext2D;
    this.ctx = this.board.getContext("2d") as CanvasRenderingContext2D;
    this.chartCtx = this.chart.getContext("2d") as CanvasRenderingContext2D;

    this.net.width = NET_WIDTH;
    this.net.height = NET_HEIGHT;
    this.board.width = BOARD_SIZE;
    this.board.height = BOARD_SIZE;
    this.chart.width = CHART_WIDTH;
    this.chart.height = CHART_HEIGHT;

    this.turboMode = this.turboToggle.checked;
    this.turboToggle.addEventListener("change", () => {
      this.turboMode = this.turboToggle.checked;
      this.stepBudget = 0;
    });
    this.showNetwork = this.networkToggle.checked;
    this.networkToggle.addEventListener("change", () => {
      this.showNetwork = this.networkToggle.checked;
      this.applyNetworkVisibility();
    });

    this.net.addEventListener("mousemove", (event) => {
      const rect = this.net.getBoundingClientRect();
      const scaleX = this.net.width / rect.width;
      const scaleY = this.net.height / rect.height;
      this.netMouse = {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
      };
    });
    this.net.addEventListener("mouseleave", () => {
      this.netMouse = null;
      this.net.style.cursor = "default";
    });

    for (let i = 0; i < POP_SIZE; i++) {
      this.population.push(this.createAgent());
    }

    this.setShowcaseGenome(this.population[0].genome);
    this.applyNetworkVisibility();

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

  private applyNetworkVisibility(): void {
    this.net.style.display = this.showNetwork ? "block" : "none";
    if (!this.showNetwork) {
      this.netMouse = null;
      this.net.style.cursor = "default";
    }
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
    for (let h = 0; h < HIDDEN; h++) {
      let sum = genome[OFFSET_H_BIAS + h];
      const wOffset = OFFSET_IH + h * INPUTS;
      for (let i = 0; i < INPUTS; i++) {
        sum += genome[wOffset + i] * inputs[i];
      }
      this.hiddenBuffer[h] = Math.tanh(sum);
    }

    let bestValue = Number.NEGATIVE_INFINITY;
    let bestAction = 0;

    for (let output = 0; output < OUTPUTS; output++) {
      let value = genome[OFFSET_O_BIAS + output];
      const wOffset = OFFSET_HO + output * HIDDEN;
      for (let h = 0; h < HIDDEN; h++) {
        value += genome[wOffset + h] * this.hiddenBuffer[h];
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
    const dist =
      Math.abs(head.x - agent.food.x) + Math.abs(head.y - agent.food.y);
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

  private computeVisualActivations(
    genome: Genome,
    agent: Agent | null,
  ): {
    input: Float32Array;
    hidden: Float32Array;
    output: Float32Array;
    best: number;
  } {
    if (agent) {
      const sensed = this.sense(agent);
      for (let i = 0; i < INPUTS; i++) {
        this.vizInputs[i] = sensed[i];
      }
    } else {
      for (let i = 0; i < INPUTS; i++) {
        this.vizInputs[i] = 0;
      }
      this.vizInputs[INPUTS - 1] = 1;
    }

    for (let h = 0; h < HIDDEN; h++) {
      let sum = genome[OFFSET_H_BIAS + h];
      const wOffset = OFFSET_IH + h * INPUTS;
      for (let i = 0; i < INPUTS; i++) {
        sum += genome[wOffset + i] * this.vizInputs[i];
      }
      this.vizHidden[h] = Math.tanh(sum);
    }

    let best = 0;
    let bestValue = Number.NEGATIVE_INFINITY;

    for (let o = 0; o < OUTPUTS; o++) {
      let value = genome[OFFSET_O_BIAS + o];
      const wOffset = OFFSET_HO + o * HIDDEN;
      for (let h = 0; h < HIDDEN; h++) {
        value += genome[wOffset + h] * this.vizHidden[h];
      }
      this.vizOutputs[o] = value;
      if (value > bestValue) {
        bestValue = value;
        best = o;
      }
    }

    return {
      input: this.vizInputs,
      hidden: this.vizHidden,
      output: this.vizOutputs,
      best,
    };
  }

  private segmentDistanceSquared(
    px: number,
    py: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
  ): number {
    const vx = x2 - x1;
    const vy = y2 - y1;
    const wx = px - x1;
    const wy = py - y1;
    const lengthSq = vx * vx + vy * vy;

    if (lengthSq <= 0.000001) {
      const dx = px - x1;
      const dy = py - y1;
      return dx * dx + dy * dy;
    }

    let t = (wx * vx + wy * vy) / lengthSq;
    t = Math.max(0, Math.min(1, t));
    const cx = x1 + t * vx;
    const cy = y1 + t * vy;
    const dx = px - cx;
    const dy = py - cy;
    return dx * dx + dy * dy;
  }

  private drawNetwork(genome: Genome | null, agent: Agent | null): void {
    this.netCtx.fillStyle = "#000";
    this.netCtx.fillRect(0, 0, NET_WIDTH, NET_HEIGHT);

    if (!genome) {
      return;
    }

    const inputLabels = [
      "Front blocked",
      "Left blocked",
      "Right blocked",
      "Food forward",
      "Food left",
      "Food right",
      "Bias",
    ];
    const outputLabels = ["Straight", "Turn left", "Turn right"];
    const activations = this.computeVisualActivations(genome, agent);

    let outputAbsMax = 0.001;
    for (let o = 0; o < OUTPUTS; o++) {
      outputAbsMax = Math.max(outputAbsMax, Math.abs(activations.output[o]));
    }

    const top = 24;
    const bottom = NET_HEIGHT - 24;
    const inputX = 120;
    const hiddenX = NET_WIDTH / 2;
    const outputX = NET_WIDTH - 120;

    const inputY: number[] = [];
    const hiddenY: number[] = [];
    const outputY: number[] = [];

    for (let i = 0; i < INPUTS; i++) {
      inputY.push(top + (i * (bottom - top)) / Math.max(1, INPUTS - 1));
    }
    for (let h = 0; h < HIDDEN; h++) {
      hiddenY.push(top + (h * (bottom - top)) / Math.max(1, HIDDEN - 1));
    }
    for (let o = 0; o < OUTPUTS; o++) {
      outputY.push(top + (o * (bottom - top)) / Math.max(1, OUTPUTS - 1));
    }

    const edges: NetEdge[] = [];

    for (let h = 0; h < HIDDEN; h++) {
      const wOffset = OFFSET_IH + h * INPUTS;
      for (let i = 0; i < INPUTS; i++) {
        edges.push({
          x1: inputX,
          y1: inputY[i],
          x2: hiddenX,
          y2: hiddenY[h],
          weight: genome[wOffset + i],
          label: `${inputLabels[i]} -> H${h + 1}`,
        });
      }
    }

    for (let o = 0; o < OUTPUTS; o++) {
      const wOffset = OFFSET_HO + o * HIDDEN;
      for (let h = 0; h < HIDDEN; h++) {
        edges.push({
          x1: hiddenX,
          y1: hiddenY[h],
          x2: outputX,
          y2: outputY[o],
          weight: genome[wOffset + h],
          label: `H${h + 1} -> ${outputLabels[o]}`,
        });
      }
    }

    let maxAbs = 0.001;
    for (const edge of edges) {
      maxAbs = Math.max(maxAbs, Math.abs(edge.weight));
    }

    let hoveredEdge: NetEdge | null = null;
    if (this.netMouse) {
      let bestDistSq = NET_HOVER_RADIUS * NET_HOVER_RADIUS;
      for (const edge of edges) {
        const distSq = this.segmentDistanceSquared(
          this.netMouse.x,
          this.netMouse.y,
          edge.x1,
          edge.y1,
          edge.x2,
          edge.y2,
        );
        if (distSq <= bestDistSq) {
          bestDistSq = distSq;
          hoveredEdge = edge;
        }
      }
    }
    this.net.style.cursor = hoveredEdge ? "pointer" : "default";

    const drawConnection = (edge: NetEdge): void => {
      const strength = Math.min(1, Math.abs(edge.weight) / maxAbs);
      this.netCtx.strokeStyle =
        edge.weight >= 0
          ? `rgba(76, 175, 80, ${0.15 + strength * 0.7})`
          : `rgba(244, 67, 54, ${0.15 + strength * 0.7})`;
      this.netCtx.lineWidth = 0.5 + strength * 2;
      this.netCtx.beginPath();
      this.netCtx.moveTo(edge.x1, edge.y1);
      this.netCtx.lineTo(edge.x2, edge.y2);
      this.netCtx.stroke();
    };
    for (const edge of edges) {
      drawConnection(edge);
    }

    if (hoveredEdge) {
      this.netCtx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      this.netCtx.lineWidth = 3;
      this.netCtx.beginPath();
      this.netCtx.moveTo(hoveredEdge.x1, hoveredEdge.y1);
      this.netCtx.lineTo(hoveredEdge.x2, hoveredEdge.y2);
      this.netCtx.stroke();
    }

    this.netCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
    this.netCtx.font = "12px JetBrains Mono, monospace";
    this.netCtx.fillText("Input", inputX - 28, 14);
    this.netCtx.fillText("Hidden", hiddenX - 22, 14);
    this.netCtx.fillText("Output", outputX - 24, 14);

    for (let i = 0; i < INPUTS; i++) {
      const value = activations.input[i];
      const intensity = Math.min(1, Math.abs(value));
      if (intensity > 0.02) {
        this.netCtx.fillStyle =
          value >= 0
            ? `rgba(41, 182, 246, ${0.1 + intensity * 0.55})`
            : `rgba(255, 183, 77, ${0.1 + intensity * 0.55})`;
        this.netCtx.beginPath();
        this.netCtx.arc(inputX, inputY[i], 7 + intensity * 4, 0, Math.PI * 2);
        this.netCtx.fill();
      }

      this.netCtx.fillStyle = value >= 0 ? "#29b6f6" : "#ffb74d";
      this.netCtx.beginPath();
      this.netCtx.arc(inputX, inputY[i], 5, 0, Math.PI * 2);
      this.netCtx.fill();

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.88)";
      this.netCtx.font = "10px JetBrains Mono, monospace";
      this.netCtx.fillText(inputLabels[i], 6, inputY[i] + 3);
    }

    for (let h = 0; h < HIDDEN; h++) {
      const value = activations.hidden[h];
      const intensity = Math.min(1, Math.abs(value));
      if (intensity > 0.02) {
        this.netCtx.fillStyle =
          value >= 0
            ? `rgba(76, 175, 80, ${0.1 + intensity * 0.6})`
            : `rgba(244, 67, 54, ${0.1 + intensity * 0.6})`;
        this.netCtx.beginPath();
        this.netCtx.arc(hiddenX, hiddenY[h], 7 + intensity * 4, 0, Math.PI * 2);
        this.netCtx.fill();
      }

      this.netCtx.fillStyle = value >= 0 ? "#26a69a" : "#ef5350";
      this.netCtx.beginPath();
      this.netCtx.arc(hiddenX, hiddenY[h], 5, 0, Math.PI * 2);
      this.netCtx.fill();
    }

    for (let o = 0; o < OUTPUTS; o++) {
      const value = activations.output[o];
      const intensity = Math.min(1, Math.abs(value) / outputAbsMax);
      if (intensity > 0.02) {
        this.netCtx.fillStyle =
          value >= 0
            ? `rgba(76, 175, 80, ${0.12 + intensity * 0.6})`
            : `rgba(244, 67, 54, ${0.12 + intensity * 0.6})`;
        this.netCtx.beginPath();
        this.netCtx.arc(outputX, outputY[o], 8 + intensity * 5, 0, Math.PI * 2);
        this.netCtx.fill();
      }

      this.netCtx.fillStyle = o === activations.best ? "#ffd54f" : "#ffb74d";
      this.netCtx.beginPath();
      this.netCtx.arc(outputX, outputY[o], 6, 0, Math.PI * 2);
      this.netCtx.fill();

      if (o === activations.best) {
        this.netCtx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        this.netCtx.lineWidth = 1.5;
        this.netCtx.beginPath();
        this.netCtx.arc(outputX, outputY[o], 9, 0, Math.PI * 2);
        this.netCtx.stroke();
      }

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
      this.netCtx.font = "11px JetBrains Mono, monospace";
      this.netCtx.fillText(outputLabels[o], outputX + 12, outputY[o] + 3);
      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.65)";
      this.netCtx.font = "9px JetBrains Mono, monospace";
      this.netCtx.fillText(
        `b=${genome[OFFSET_O_BIAS + o].toFixed(2)}`,
        outputX + 12,
        outputY[o] + 14,
      );
      this.netCtx.fillText(
        `a=${value.toFixed(2)}`,
        outputX + 12,
        outputY[o] + 24,
      );
    }

    if (hoveredEdge && this.netMouse) {
      const line1 = hoveredEdge.label;
      const line2 = `w = ${hoveredEdge.weight.toFixed(4)}`;
      this.netCtx.font = "10px JetBrains Mono, monospace";
      const width = Math.max(
        this.netCtx.measureText(line1).width,
        this.netCtx.measureText(line2).width,
      );
      const boxWidth = width + 12;
      const boxHeight = 34;
      const x = Math.min(NET_WIDTH - boxWidth - 4, this.netMouse.x + 10);
      const y = Math.max(4, this.netMouse.y - boxHeight - 8);

      this.netCtx.fillStyle = "rgba(12, 12, 12, 0.95)";
      this.netCtx.fillRect(x, y, boxWidth, boxHeight);
      this.netCtx.strokeStyle = "rgba(255, 255, 255, 0.25)";
      this.netCtx.lineWidth = 1;
      this.netCtx.strokeRect(x, y, boxWidth, boxHeight);

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.92)";
      this.netCtx.fillText(line1, x + 6, y + 13);
      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.75)";
      this.netCtx.fillText(line2, x + 6, y + 26);
    }

    this.netCtx.fillStyle = "rgba(255, 255, 255, 0.55)";
    this.netCtx.font = "10px JetBrains Mono, monospace";
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
      const y =
        CHART_HEIGHT - 10 - (this.history[i] / maxScore) * (CHART_HEIGHT - 20);
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
    if (this.showNetwork) {
      this.drawNetwork(this.showcaseGenome, view);
    }
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
