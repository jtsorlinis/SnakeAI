import {
  BASE_HUNGER,
  CRITIC_INPUTS,
  DIRS,
  GRID_SIZE,
  INPUTS,
  MAX_SCORE,
  OUTPUTS,
} from "./config";
import { NeuralNetwork, createNetworkHiddenBuffers } from "./NeuralNetwork";
import type {
  Agent,
  PolicyParams,
  NetworkActivations,
  Point,
  PolicyPlaybackMode,
} from "./types";

export class SnakeEnvironment {
  private readonly actionInputs = new Float32Array(INPUTS);
  private readonly actionHidden = createNetworkHiddenBuffers();
  private readonly actionOutputs = new Float32Array(OUTPUTS);
  private readonly vizInputs = new Float32Array(INPUTS);
  private readonly vizHidden = createNetworkHiddenBuffers();
  private readonly vizOutputs = new Float32Array(OUTPUTS);

  constructor(private readonly network: NeuralNetwork) {}

  public createAgent(policy: PolicyParams): Agent {
    const x = Math.floor(GRID_SIZE / 2);
    const y = Math.floor(GRID_SIZE / 2);
    const body = [
      { x, y },
      { x: x - 1, y },
      { x: x - 2, y },
    ];

    const cells = new Uint8Array(GRID_SIZE * GRID_SIZE);
    for (const part of body) {
      cells[part.y * GRID_SIZE + part.x] = 1;
    }

    return {
      policy: new Float32Array(policy),
      body,
      cells,
      dir: 1,
      food: this.randomFood(cells),
      alive: true,
      score: 0,
      steps: 0,
      hunger: BASE_HUNGER,
      fitness: 0,
      terminalReason: null,
    };
  }

  // Policy inputs followed by critic-only context (length, hunger).
  public observe(agent: Agent): Float32Array {
    const observation = new Float32Array(CRITIC_INPUTS);
    this.senseInto(agent, observation);
    observation[INPUTS] = agent.body.length / (GRID_SIZE * GRID_SIZE);
    observation[INPUTS + 1] = agent.hunger / BASE_HUNGER;
    return observation;
  }

  public selectAction(
    agent: Agent,
    playbackMode: PolicyPlaybackMode = "greedy",
  ): number {
    this.senseInto(agent, this.actionInputs);

    if (playbackMode === "stochastic") {
      this.network.run(
        agent.policy,
        this.actionInputs,
        this.actionHidden,
        this.actionOutputs,
      );
      return this.sampleFromLogits(this.actionOutputs);
    }

    return this.network.chooseAction(agent.policy, this.actionInputs);
  }

  public step(agent: Agent, actionOverride?: number): void {
    if (!agent.alive) {
      return;
    }

    const action = actionOverride ?? this.selectAction(agent);
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
      agent.terminalReason = "collision";
      return;
    }

    const ate = next.x === agent.food.x && next.y === agent.food.y;
    const nextCell = next.y * GRID_SIZE + next.x;

    if (agent.cells[nextCell] === 1) {
      // The tail tip moves out of the way, unless the snake is eating.
      const tail = agent.body[agent.body.length - 1];
      if (ate || tail.x !== next.x || tail.y !== next.y) {
        agent.alive = false;
        agent.terminalReason = "collision";
        return;
      }
    }

    if (!ate) {
      const tail = agent.body.pop()!;
      agent.cells[tail.y * GRID_SIZE + tail.x] = 0;
    }
    agent.body.unshift(next);
    agent.cells[nextCell] = 1;

    if (ate) {
      agent.score += 1;

      if (agent.score >= MAX_SCORE) {
        agent.alive = false;
        agent.terminalReason = "solved";
        return;
      }

      agent.hunger = BASE_HUNGER;
      agent.food = this.randomFood(agent.cells);
    }

    if (agent.hunger <= 0) {
      agent.alive = false;
      agent.terminalReason = "hunger";
    }
  }

  public computeNetworkActivations(
    policy: PolicyParams,
    agent: Agent | null,
  ): NetworkActivations {
    if (agent) {
      this.senseInto(agent, this.vizInputs);
    } else {
      this.vizInputs.fill(0);
    }

    const best = this.network.run(
      policy,
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

  private randomIndex(maxExclusive: number): number {
    return Math.floor(Math.random() * maxExclusive);
  }

  // There is always a free cell to find: a game ends as solved before the
  // body covers the whole board.
  private randomFood(cells: Uint8Array): Point {
    while (true) {
      const x = this.randomIndex(GRID_SIZE);
      const y = this.randomIndex(GRID_SIZE);
      if (cells[y * GRID_SIZE + x] === 0) {
        return { x, y };
      }
    }
  }

  private outOfBounds(x: number, y: number): boolean {
    return x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE;
  }

  private checkForObstacle(
    head: Point,
    direction: Point,
    cells: Uint8Array,
  ): number {
    const checkX = head.x + direction.x;
    const checkY = head.y + direction.y;

    if (this.outOfBounds(checkX, checkY)) {
      return 1;
    }

    return cells[checkY * GRID_SIZE + checkX];
  }

  private findTailDistance(
    head: Point,
    direction: Point,
    cells: Uint8Array,
  ): number {
    for (let distance = 1; distance <= GRID_SIZE; distance++) {
      const x = head.x + direction.x * distance;
      const y = head.y + direction.y * distance;

      if (this.outOfBounds(x, y)) {
        return 0;
      }

      if (cells[y * GRID_SIZE + x] === 1) {
        return 1 / distance;
      }
    }

    return 0;
  }

  private checkForFood(head: Point, direction: Point, food: Point): number {
    const foodDeltaX = food.x - head.x;
    const foodDeltaY = food.y - head.y;
    const dot = foodDeltaX * direction.x + foodDeltaY * direction.y;
    return Math.sign(dot);
  }

  private senseInto(agent: Agent, target: Float32Array): void {
    const head = agent.body[0];
    const front = DIRS[agent.dir];
    const left = DIRS[(agent.dir + 3) % 4];
    const right = DIRS[(agent.dir + 1) % 4];

    target[0] = this.checkForObstacle(head, front, agent.cells);
    target[1] = this.checkForObstacle(head, left, agent.cells);
    target[2] = this.checkForObstacle(head, right, agent.cells);
    target[3] = this.findTailDistance(head, front, agent.cells);
    target[4] = this.findTailDistance(head, left, agent.cells);
    target[5] = this.findTailDistance(head, right, agent.cells);
    target[6] = this.checkForFood(head, front, agent.food);
    target[7] = this.checkForFood(head, left, agent.food);
    target[8] = front.x;
    target[9] = front.y;
  }

  private sampleFromLogits(logits: Float32Array): number {
    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) {
        maxLogit = logits[i];
      }
    }

    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      const expValue = Math.exp(logits[i] - maxLogit);
      this.actionOutputs[i] = expValue;
      sum += expValue;
    }

    const sample = Math.random() * sum;
    let cumulative = 0;
    for (let i = 0; i < this.actionOutputs.length; i++) {
      cumulative += this.actionOutputs[i];
      if (sample <= cumulative) {
        return i;
      }
    }

    return this.actionOutputs.length - 1;
  }
}
