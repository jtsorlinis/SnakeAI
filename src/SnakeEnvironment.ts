import {
  BASE_HUNGER,
  DIRS,
  GRID_SIZE,
  INPUT_CHANNEL_CELLS,
  INPUT_GRID_SIZE,
  INPUTS,
  MAX_SCORE,
  OUTPUTS,
} from "./config";
import { NeuralNetwork, createNetworkHiddenBuffers } from "./NeuralNetwork";
import type { Agent, Genome, NetworkActivations, Point } from "./types";

export class SnakeEnvironment {
  private readonly actionInputs = new Float32Array(INPUTS);
  private readonly vizInputs = new Float32Array(INPUTS);
  private readonly vizHidden = createNetworkHiddenBuffers();
  private readonly vizOutputs = new Float32Array(OUTPUTS);

  constructor(private readonly network: NeuralNetwork) {}

  public createAgent(genome: Genome): Agent {
    const x = Math.floor(GRID_SIZE / 2);
    const y = Math.floor(GRID_SIZE / 2);
    const body = [
      { x, y },
      { x: x - 1, y },
      { x: x - 2, y },
    ];

    return {
      genome: new Float32Array(genome),
      body,
      dir: 1,
      food: this.randomFood(body),
      alive: true,
      score: 0,
      steps: 0,
      stepsSinceFood: 0,
      hunger: BASE_HUNGER,
      fitness: 0,
    };
  }

  public step(agent: Agent): void {
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
    agent.stepsSinceFood += 1;
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
      agent.stepsSinceFood = 0;
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

  public computeNetworkActivations(
    genome: Genome,
    agent: Agent | null,
  ): NetworkActivations {
    if (agent) {
      this.senseInto(agent, this.vizInputs);
    } else {
      this.vizInputs.fill(0);
    }

    const best = this.network.run(
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

  private randomIndex(maxExclusive: number): number {
    return Math.floor(Math.random() * maxExclusive);
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

  private scaleCoordinate(value: number): number {
    if (GRID_SIZE <= 1) {
      return 0;
    }

    const ratio = value / (GRID_SIZE - 1);
    return Math.max(
      0,
      Math.min(INPUT_GRID_SIZE - 1, Math.round(ratio * (INPUT_GRID_SIZE - 1))),
    );
  }

  private toInputCellIndex(point: Point): number {
    const x = this.scaleCoordinate(point.x);
    const y = this.scaleCoordinate(point.y);
    return y * INPUT_GRID_SIZE + x;
  }

  private senseInto(agent: Agent, target: Float32Array): void {
    target.fill(0);

    const head = agent.body[0]!;
    const front = DIRS[agent.dir];

    const headOffset = 0;
    const bodyOffset = INPUT_CHANNEL_CELLS;
    const foodOffset = INPUT_CHANNEL_CELLS * 2;
    const directionOffset = INPUT_CHANNEL_CELLS * 3;

    target[headOffset + this.toInputCellIndex(head)] = 1;

    const bodyAgeDenominator = Math.max(1, agent.body.length - 1);
    for (let i = 1; i < agent.body.length; i++) {
      const ageNormalized = i / bodyAgeDenominator;
      const bodyIndex = bodyOffset + this.toInputCellIndex(agent.body[i]!);
      target[bodyIndex] = Math.max(target[bodyIndex], ageNormalized);
    }

    target[foodOffset + this.toInputCellIndex(agent.food)] = 1;
    target[directionOffset] = front.x;
    target[directionOffset + 1] = front.y;
  }

  private chooseAction(agent: Agent): number {
    this.senseInto(agent, this.actionInputs);
    return this.network.chooseAction(agent.genome, this.actionInputs);
  }
}
