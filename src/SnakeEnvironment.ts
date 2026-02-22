import {
  BASE_HUNGER,
  DIRS,
  GRID_SIZE,
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
      hunger: BASE_HUNGER,
      fitness: 0,
    };
  }

  public observe(agent: Agent): Float32Array {
    const observation = new Float32Array(INPUTS);
    this.senseInto(agent, observation);
    return observation;
  }

  public step(agent: Agent, actionOverride?: number): void {
    if (!agent.alive) {
      return;
    }

    const action = actionOverride ?? this.chooseAction(agent);
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
        return 0;
      }

      for (let i = 1; i < snake.length; i++) {
        if (snake[i].x === x && snake[i].y === y) {
          return 1 / distance;
        }
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

    target[0] = this.checkForObstacle(head, front, agent.body);
    target[1] = this.checkForObstacle(head, left, agent.body);
    target[2] = this.checkForObstacle(head, right, agent.body);
    target[3] = this.findTailDistance(head, front, agent.body);
    target[4] = this.findTailDistance(head, left, agent.body);
    target[5] = this.findTailDistance(head, right, agent.body);
    target[6] = this.checkForFood(head, front, agent.food);
    target[7] = this.checkForFood(head, left, agent.food);
    target[8] = front.x;
    target[9] = front.y;
  }

  private chooseAction(agent: Agent): number {
    this.senseInto(agent, this.actionInputs);
    return this.network.chooseAction(agent.genome, this.actionInputs);
  }
}
