import { Direction, GRID_HEIGHT, GRID_WIDTH, Point } from "./types";
import { NeuralNetwork } from "./NeuralNetwork";
import { Food } from "./Food";

export const BRAIN_CONFIG = {
  inputNodes: 11,
  hiddenNodes: 10,
  outputNodes: 3,
};

export const WIN_REWARD = 1000;

const MAX_STEPS_WITHOUT_FOOD = GRID_WIDTH * GRID_HEIGHT;

export class Snake {
  body: Point[];
  direction: Direction;
  newDirection: Direction;
  growPending: boolean;
  dead: boolean;
  won: boolean;

  brain: NeuralNetwork;
  fitness: number = 0;
  maxStepsWithoutFood: number = MAX_STEPS_WITHOUT_FOOD;
  moves: number = 0;

  bodySet: Set<number>;
  visionInputs: Float32Array;

  constructor(brain?: NeuralNetwork) {
    const startX = Math.floor(GRID_WIDTH / 2);
    const startY = Math.floor(GRID_HEIGHT / 2);
    this.body = [
      { x: startX, y: startY },
      { x: startX - 1, y: startY },
      { x: startX - 2, y: startY },
    ];

    this.bodySet = new Set();
    this.body.forEach((p) => this.bodySet.add(p.y * GRID_WIDTH + p.x));

    this.visionInputs = new Float32Array(BRAIN_CONFIG.inputNodes);
    this.direction = Direction.Right;
    this.newDirection = Direction.Right;
    this.growPending = false;
    this.dead = false;
    this.won = false;

    if (brain) {
      this.brain = brain.clone();
    } else {
      this.brain = new NeuralNetwork(
        BRAIN_CONFIG.inputNodes,
        BRAIN_CONFIG.hiddenNodes,
        BRAIN_CONFIG.outputNodes,
      );
    }
  }

  get score(): number {
    return this.body.length - 3;
  }

  setDirection(dir: Direction) {
    if (this.dead) return;
    if (
      (this.direction === Direction.Up && dir === Direction.Down) ||
      (this.direction === Direction.Down && dir === Direction.Up) ||
      (this.direction === Direction.Left && dir === Direction.Right) ||
      (this.direction === Direction.Right && dir === Direction.Left)
    ) {
      return;
    }
    this.newDirection = dir;
  }

  look(food: Food) {
    const head = this.body[0];

    const { front, left, right } = this.getRelativeDirections();

    this.visionInputs[0] = this.isBlocked(this.nextHeadForDirection(front))
      ? 1
      : 0;
    this.visionInputs[1] = this.isBlocked(this.nextHeadForDirection(left))
      ? 1
      : 0;
    this.visionInputs[2] = this.isBlocked(this.nextHeadForDirection(right))
      ? 1
      : 0;

    this.visionInputs[3] = this.tailDistanceInput(front);
    this.visionInputs[4] = this.tailDistanceInput(left);
    this.visionInputs[5] = this.tailDistanceInput(right);

    this.visionInputs[6] = food.position.x > head.x ? 1 : 0;
    this.visionInputs[7] = food.position.y > head.y ? 1 : 0;

    this.visionInputs[8] = this.direction === Direction.Up ? 1 : 0;
    this.visionInputs[9] = this.direction === Direction.Right ? 1 : 0;
    this.visionInputs[10] = this.direction === Direction.Down ? 1 : 0;

    return this.visionInputs;
  }

  isOnBody(pt: Point): boolean {
    return this.bodySet.has(pt.y * GRID_WIDTH + pt.x);
  }

  private isBlocked(pt: Point): boolean {
    if (pt.x < 0 || pt.x >= GRID_WIDTH || pt.y < 0 || pt.y >= GRID_HEIGHT) {
      return true;
    }

    if (!this.isOnBody(pt)) return false;

    if (!this.growPending) {
      const tail = this.body[this.body.length - 1];
      if (pt.x === tail.x && pt.y === tail.y) return false;
    }

    return true;
  }

  private tailDistanceInput(dir: Direction): number {
    const step = this.deltaForDirection(dir);
    const head = this.body[0];
    let pos = { x: head.x + step.x, y: head.y + step.y };
    let distance = 1;

    while (
      pos.x >= 0 &&
      pos.x < GRID_WIDTH &&
      pos.y >= 0 &&
      pos.y < GRID_HEIGHT
    ) {
      if (this.isOnBody(pos)) {
        if (!this.growPending) {
          const tail = this.body[this.body.length - 1];
          if (pos.x === tail.x && pos.y === tail.y) {
            pos.x += step.x;
            pos.y += step.y;
            distance++;
            continue;
          }
        }
        return 1 / distance;
      }
      pos.x += step.x;
      pos.y += step.y;
      distance++;
    }

    return 0;
  }

  private deltaForDirection(dir: Direction): Point {
    switch (dir) {
      case Direction.Up:
        return { x: 0, y: -1 };
      case Direction.Down:
        return { x: 0, y: 1 };
      case Direction.Left:
        return { x: -1, y: 0 };
      case Direction.Right:
        return { x: 1, y: 0 };
      default:
        return { x: 0, y: 0 };
    }
  }

  private getRelativeDirections(): {
    front: Direction;
    left: Direction;
    right: Direction;
  } {
    switch (this.direction) {
      case Direction.Up:
        return {
          front: Direction.Up,
          left: Direction.Left,
          right: Direction.Right,
        };
      case Direction.Right:
        return {
          front: Direction.Right,
          left: Direction.Up,
          right: Direction.Down,
        };
      case Direction.Down:
        return {
          front: Direction.Down,
          left: Direction.Right,
          right: Direction.Left,
        };
      case Direction.Left:
        return {
          front: Direction.Left,
          left: Direction.Down,
          right: Direction.Up,
        };
      default:
        return {
          front: Direction.Up,
          left: Direction.Left,
          right: Direction.Right,
        };
    }
  }

  private dirAfterAction(actionIndex: number): Direction {
    if (actionIndex === 0) return this.direction;

    if (actionIndex === 1) {
      switch (this.direction) {
        case Direction.Up:
          return Direction.Left;
        case Direction.Down:
          return Direction.Right;
        case Direction.Left:
          return Direction.Down;
        case Direction.Right:
          return Direction.Up;
      }
    }

    switch (this.direction) {
      case Direction.Up:
        return Direction.Right;
      case Direction.Down:
        return Direction.Left;
      case Direction.Left:
        return Direction.Up;
      case Direction.Right:
        return Direction.Down;
    }
  }

  private nextHeadForDirection(dir: Direction): Point {
    const head = this.body[0];
    switch (dir) {
      case Direction.Up:
        return { x: head.x, y: head.y - 1 };
      case Direction.Down:
        return { x: head.x, y: head.y + 1 };
      case Direction.Left:
        return { x: head.x - 1, y: head.y };
      case Direction.Right:
        return { x: head.x + 1, y: head.y };
    }
  }

  think(food: Food) {
    const inputs = this.look(food);
    const outputs = this.brain.predict(inputs);

    let maxVal = -Infinity;
    let maxIndex = 0;
    for (let i = 0; i < outputs.length; i++) {
      if (outputs[i] > maxVal) {
        maxVal = outputs[i];
        maxIndex = i;
      }
    }

    this.newDirection = this.dirAfterAction(maxIndex);
  }

  move(food?: Food) {
    if (this.dead) return;

    this.moves++;
    this.maxStepsWithoutFood--;

    if (this.maxStepsWithoutFood < 0) {
      this.dead = true;
      return;
    }

    this.direction = this.newDirection;
    const newHead = this.nextHeadForDirection(this.direction);

    if (
      newHead.x < 0 ||
      newHead.x >= GRID_WIDTH ||
      newHead.y < 0 ||
      newHead.y >= GRID_HEIGHT
    ) {
      this.dead = true;
      return;
    }

    for (let i = 0; i < this.body.length; i++) {
      if (!this.growPending && i === this.body.length - 1) continue;
      if (newHead.x === this.body[i].x && newHead.y === this.body[i].y) {
        this.dead = true;
        return;
      }
    }

    this.body.unshift(newHead);
    this.bodySet.add(newHead.y * GRID_WIDTH + newHead.x);

    if (!this.growPending) {
      const tail = this.body.pop();
      if (tail) {
        this.bodySet.delete(tail.y * GRID_WIDTH + tail.x);
      }
    } else {
      this.growPending = false;
      this.maxStepsWithoutFood = MAX_STEPS_WITHOUT_FOOD;
    }
  }

  grow() {
    this.growPending = true;
  }

  draw(ctx: CanvasRenderingContext2D, tileSize: number, color: string) {
    if (this.dead) return;

    ctx.fillStyle = color;
    for (const segment of this.body) {
      ctx.fillRect(
        segment.x * tileSize,
        segment.y * tileSize,
        tileSize,
        tileSize,
      );
    }
  }

  calculateFitness() {
    let fitness = this.score * 10;
    fitness += this.moves / 100;
    if (this.won) fitness += WIN_REWARD;
    if (this.score === 0) {
      fitness = this.moves / 200;
    }
    this.fitness = fitness;
  }

  mutate(rate: number) {
    this.brain.mutate(rate);
  }
}
