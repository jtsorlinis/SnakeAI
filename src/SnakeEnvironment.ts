import {
  BASE_HUNGER,
  DIRS,
  GRID_SIZE,
  MAX_SCORE,
  OBS_CHANNELS,
  REWARD_COLLISION_DEATH,
  REWARD_EAT,
  REWARD_STARVATION_DEATH,
  rewardStepPenalty,
} from "./config";
import type { Agent, Point } from "./types";

export type StepResult = {
  reward: number;
  done: boolean;
};

const CHANNEL_FOOD = 0;
const CHANNEL_HEAD = 1;
const CHANNEL_BODY_AGE = 2;
const CHANNEL_DIR_X = 3;
const CHANNEL_DIR_Y = 4;

export class SnakeEnvironment {
  public createAgent(): Agent {
    const x = Math.floor(GRID_SIZE / 2);
    const y = Math.floor(GRID_SIZE / 2);
    const body = [
      { x, y },
      { x: x - 1, y },
      { x: x - 2, y },
    ];

    return {
      body,
      dir: 1,
      food: this.randomFood(body),
      alive: true,
      score: 0,
      steps: 0,
      hunger: BASE_HUNGER,
      episodeReturn: 0,
    };
  }

  public resetAgent(agent?: Agent): Agent {
    if (!agent) {
      return this.createAgent();
    }

    const x = Math.floor(GRID_SIZE / 2);
    const y = Math.floor(GRID_SIZE / 2);
    const body = [
      { x, y },
      { x: x - 1, y },
      { x: x - 2, y },
    ];

    agent.body = body;
    agent.dir = 1;
    agent.food = this.randomFood(body);
    agent.alive = true;
    agent.score = 0;
    agent.steps = 0;
    agent.hunger = BASE_HUNGER;
    agent.episodeReturn = 0;

    return agent;
  }

  public step(agent: Agent, action: number): StepResult {
    if (!agent.alive) {
      return { reward: 0, done: true };
    }

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
      return { reward: REWARD_COLLISION_DEATH, done: true };
    }

    const ate = next.x === agent.food.x && next.y === agent.food.y;
    const bodyCheckLen = ate ? agent.body.length : agent.body.length - 1;

    if (this.pointInBody(agent.body, next, bodyCheckLen)) {
      agent.alive = false;
      return { reward: REWARD_COLLISION_DEATH, done: true };
    }

    agent.body.unshift(next);

    if (ate) {
      agent.score += 1;
      agent.hunger = BASE_HUNGER;

      if (agent.score >= MAX_SCORE) {
        agent.alive = false;
        return { reward: REWARD_EAT, done: true };
      }

      agent.food = this.randomFood(agent.body);
      return { reward: REWARD_EAT, done: false };
    }

    agent.body.pop();

    if (agent.hunger <= 0) {
      agent.alive = false;
      return { reward: REWARD_STARVATION_DEATH, done: true };
    }

    return {
      reward: rewardStepPenalty(),
      done: false,
    };
  }

  public observe(agent: Agent, target?: Float32Array): Float32Array {
    const size = GRID_SIZE * GRID_SIZE;
    const obs = target ?? new Float32Array(OBS_CHANNELS * size);
    obs.fill(0);

    if (!agent.alive || agent.body.length === 0) {
      return obs;
    }

    obs[this.obsIndex(CHANNEL_FOOD, agent.food.x, agent.food.y)] = 1;

    const head = agent.body[0];
    obs[this.obsIndex(CHANNEL_HEAD, head.x, head.y)] = 1;

    const maxClearTime = GRID_SIZE * GRID_SIZE;
    for (let i = 0; i < agent.body.length; i++) {
      const part = agent.body[i];
      const stepsUntilClear = agent.body.length - i;
      obs[this.obsIndex(CHANNEL_BODY_AGE, part.x, part.y)] =
        stepsUntilClear / maxClearTime;
    }

    const direction = DIRS[agent.dir];
    const dirXBase = CHANNEL_DIR_X * size;
    const dirYBase = CHANNEL_DIR_Y * size;
    for (let i = 0; i < size; i++) {
      obs[dirXBase + i] = direction.x;
      obs[dirYBase + i] = direction.y;
    }

    return obs;
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

  private obsIndex(channel: number, x: number, y: number): number {
    const area = GRID_SIZE * GRID_SIZE;
    return channel * area + y * GRID_SIZE + x;
  }
}
