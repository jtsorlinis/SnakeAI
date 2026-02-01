import { Direction, GRID_HEIGHT, GRID_WIDTH, Point } from "./types";
import { NeuralNetwork } from "./NeuralNetwork";
import { Food } from "./Food";

export const BRAIN_CONFIG = {
  inputNodes: 27,
  hiddenNodes: 64,
  outputNodes: 3,
};

export class Snake {
  private static readonly GLOBAL_DIRS: Point[] = [
    { x: 0, y: -1 },
    { x: 1, y: -1 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
    { x: -1, y: 1 },
    { x: -1, y: 0 },
    { x: -1, y: -1 },
  ];

  body: Point[];
  direction: Direction;
  newDirection: Direction;
  growPending: boolean;
  dead: boolean;

  brain: NeuralNetwork;
  lifetime: number = 0;
  fitness: number = 0;
  movesLeft: number = 500;
  foodProgress: number = 0;

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

    let startIndex = 0;
    switch (this.direction) {
      case Direction.Up:
        startIndex = 0;
        break;
      case Direction.Right:
        startIndex = 2;
        break;
      case Direction.Down:
        startIndex = 4;
        break;
      case Direction.Left:
        startIndex = 6;
        break;
    }

    for (let i = 0; i < 8; i++) {
      const dirIndex = (startIndex + i) % 8;
      this.lookInDirection(Snake.GLOBAL_DIRS[dirIndex], food, i * 3);
    }

    const dx = food.position.x - head.x;
    const dy = food.position.y - head.y;

    let forwardNorm = 0;
    let rightNorm = 0;

    switch (this.direction) {
      case Direction.Up:
        forwardNorm = -dy / GRID_HEIGHT;
        rightNorm = dx / GRID_WIDTH;
        break;
      case Direction.Right:
        forwardNorm = dx / GRID_WIDTH;
        rightNorm = dy / GRID_HEIGHT;
        break;
      case Direction.Down:
        forwardNorm = dy / GRID_HEIGHT;
        rightNorm = -dx / GRID_WIDTH;
        break;
      case Direction.Left:
        forwardNorm = -dx / GRID_WIDTH;
        rightNorm = -dy / GRID_HEIGHT;
        break;
    }

    this.visionInputs[24] = forwardNorm;
    this.visionInputs[25] = rightNorm;

    const maxCells = GRID_WIDTH * GRID_HEIGHT;
    this.visionInputs[26] = this.body.length / maxCells;

    return this.visionInputs;
  }

  lookInDirection(dir: Point, food: Food, offset: number) {
    const head = this.body[0];
    let pos = { x: head.x, y: head.y };
    let foundFood = false;
    let foundBody = false;

    this.visionInputs[offset] = 0;
    this.visionInputs[offset + 1] = 0;
    this.visionInputs[offset + 2] = 0;

    pos.x += dir.x;
    pos.y += dir.y;

    while (
      pos.x >= 0 &&
      pos.x < GRID_WIDTH &&
      pos.y >= 0 &&
      pos.y < GRID_HEIGHT
    ) {
      const distance = Math.abs(pos.x - head.x) + Math.abs(pos.y - head.y);

      if (!foundFood && pos.x === food.position.x && pos.y === food.position.y) {
        this.visionInputs[offset + 1] = 1 / distance;
        foundFood = true;
      }

      if (!foundBody && this.isOnBody(pos)) {
        this.visionInputs[offset + 2] = 1 / distance;
        foundBody = true;
      }

      pos.x += dir.x;
      pos.y += dir.y;
    }

    const wallDistance = Math.abs(pos.x - head.x) + Math.abs(pos.y - head.y);
    this.visionInputs[offset] = 1 / wallDistance;
  }

  isOnBody(pt: Point): boolean {
    return this.bodySet.has(pt.y * GRID_WIDTH + pt.x);
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

    this.lifetime++;
    this.movesLeft--;

    let prevFoodDist = 0;
    if (food) {
      const head = this.body[0];
      prevFoodDist =
        Math.abs(head.x - food.position.x) +
        Math.abs(head.y - food.position.y);
    }

    if (this.movesLeft < 0) {
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

    if (food && !this.dead) {
      const newFoodDist =
        Math.abs(newHead.x - food.position.x) +
        Math.abs(newHead.y - food.position.y);
      if (newFoodDist < prevFoodDist) {
        this.foodProgress += 2;
      } else if (newFoodDist > prevFoodDist) {
        this.foodProgress = Math.max(0, this.foodProgress - 2);
      } else {
        this.foodProgress = Math.max(0, this.foodProgress - 0.5);
      }
      if (this.foodProgress > 200) this.foodProgress = 200;
    }

    if (!this.growPending) {
      const tail = this.body.pop();
      if (tail) {
        this.bodySet.delete(tail.y * GRID_WIDTH + tail.x);
      }
    } else {
      this.growPending = false;
      this.movesLeft += 1000;
      if (this.movesLeft > 5000) this.movesLeft = 5000;
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
    const foodEaten = this.score;
    const progressBonus = Math.floor(this.foodProgress);

    if (foodEaten < 1) {
      this.fitness = Math.floor(this.lifetime / 10) + progressBonus;
    } else {
      this.fitness =
        foodEaten * foodEaten * 100 +
        Math.floor(this.lifetime / 10) +
        progressBonus;
    }
  }

  mutate(rate: number) {
    this.brain.mutate(rate);
  }
}
