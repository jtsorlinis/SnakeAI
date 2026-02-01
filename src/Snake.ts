import { Direction, GRID_HEIGHT, GRID_WIDTH, Point } from "./types";
import { NeuralNetwork } from "./NeuralNetwork";
import { Food } from "./Food";

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

  // AI properties
  brain: NeuralNetwork;
  lifetime: number = 0;
  fitness: number = 0;
  movesLeft: number = 500; // Limit moves to prevent infinite loops without eating
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

    // Pre-allocate input array to avoid GC
    this.visionInputs = new Float32Array(27);

    this.direction = Direction.Right;
    this.newDirection = Direction.Right;
    this.growPending = false;
    this.dead = false;

    // Inputs:
    // 24 inputs (8 directions * 3 types: wall, food, tail)
    // + 2 inputs (Relative food vector: forward, right)
    // + 1 input (Normalized length: current/maxCells)
    // Total = 27 inputs
    if (brain) {
      this.brain = brain.clone();
    } else {
      this.brain = new NeuralNetwork(27, 64, 3);
    }
  }

  setDirection(dir: Direction) {
    if (this.dead) return;

    // Prevent reversing direction
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

  // AI Logic: Look around
  look(food: Food) {
    const head = this.body[0];

    // Determine rotation index based on current heading
    // We want index 0 to be "Forward", index 2 to be "Right", etc.
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

    // 1. Raycasting Vision (8 Relative Directions)
    for (let i = 0; i < 8; i++) {
      const dirIndex = (startIndex + i) % 8;
      this.lookInDirection(Snake.GLOBAL_DIRS[dirIndex], food, i * 3);
    }

    // 2. Relative Food Vector (2 Inputs)
    const dx = food.position.x - head.x;
    const dy = food.position.y - head.y;

    // Rotate vector based on heading and normalize
    let forwardNorm = 0;
    let rightNorm = 0;

    switch (this.direction) {
      case Direction.Up:
        forwardNorm = -dy / GRID_HEIGHT; // Forward is -y
        rightNorm = dx / GRID_WIDTH; // Right is +x
        break;
      case Direction.Right:
        forwardNorm = dx / GRID_WIDTH; // Forward is +x
        rightNorm = dy / GRID_HEIGHT; // Right is +y
        break;
      case Direction.Down:
        forwardNorm = dy / GRID_HEIGHT; // Forward is +y
        rightNorm = -dx / GRID_WIDTH; // Right is -x
        break;
      case Direction.Left:
        forwardNorm = -dx / GRID_WIDTH; // Forward is -x
        rightNorm = -dy / GRID_HEIGHT; // Right is -y
        break;
    }

    this.visionInputs[24] = forwardNorm;
    this.visionInputs[25] = rightNorm;

    // 3. Normalized Length (1 Input)
    const maxCells = GRID_WIDTH * GRID_HEIGHT;
    this.visionInputs[26] = this.body.length / maxCells;

    return this.visionInputs;
  }

  lookInDirection(dir: Point, food: Food, offset: number) {
    // [Wall, Food, Self] stored at offset, offset+1, offset+2
    const head = this.body[0];
    let pos = { x: head.x, y: head.y };
    let foundFood = false;
    let foundBody = false;

    // Reset values
    this.visionInputs[offset] = 0;
    this.visionInputs[offset + 1] = 0;
    this.visionInputs[offset + 2] = 0;

    pos.x += dir.x;
    pos.y += dir.y;

    // Move in direction until hit wall
    while (
      pos.x >= 0 &&
      pos.x < GRID_WIDTH &&
      pos.y >= 0 &&
      pos.y < GRID_HEIGHT
    ) {
      // Manhattan distance: |dx| + |dy|
      const distance = Math.abs(pos.x - head.x) + Math.abs(pos.y - head.y);

      // Check for food
      if (!foundFood && pos.x === food.position.x && pos.y === food.position.y) {
        this.visionInputs[offset + 1] = 1 / distance;
        foundFood = true;
      }

      // Check for body
      if (!foundBody && this.isOnBody(pos)) {
        this.visionInputs[offset + 2] = 1 / distance;
        foundBody = true;
      }

      pos.x += dir.x;
      pos.y += dir.y;
    }

    // Distance to wall (using Manhattan distance of the out-of-bounds step)
    const wallDistance = Math.abs(pos.x - head.x) + Math.abs(pos.y - head.y);
    this.visionInputs[offset] = 1 / wallDistance;
  }

  isOnBody(pt: Point): boolean {
    return this.bodySet.has(pt.y * GRID_WIDTH + pt.x);
  }

  private dirAfterAction(actionIndex: number): Direction {
    // Actions: 0 = Straight, 1 = Left, 2 = Right
    if (actionIndex === 0) return this.direction;

    if (actionIndex === 1) {
      // Turn Left
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

    // Turn Right
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

  private wouldCollide(nextDir: Direction): boolean {
    const newHead = this.nextHeadForDirection(nextDir);

    // Wall collision
    if (
      newHead.x < 0 ||
      newHead.x >= GRID_WIDTH ||
      newHead.y < 0 ||
      newHead.y >= GRID_HEIGHT
    ) {
      return true;
    }

    // Self collision (same logic as move: ignore tail if not growing)
    for (let i = 0; i < this.body.length; i++) {
      if (!this.growPending && i === this.body.length - 1) continue;
      if (newHead.x === this.body[i].x && newHead.y === this.body[i].y) {
        return true;
      }
    }

    return false;
  }

  think(food: Food) {
    const inputs = this.look(food);
    const outputs = this.brain.predict(inputs);

    // Outputs: 0 = Straight, 1 = Left, 2 = Right
    let maxVal = -Infinity;
    let maxIndex = 0;
    for (let i = 0; i < outputs.length; i++) {
      if (outputs[i] > maxVal) {
        maxVal = outputs[i];
        maxIndex = i;
      }
    }

    // Action masking: if the preferred action would immediately die, pick the best safe action instead.
    let chosenIndex = maxIndex;
    const preferredDir = this.dirAfterAction(maxIndex);
    if (this.wouldCollide(preferredDir)) {
      let bestSafeIndex = -1;
      let bestSafeVal = -Infinity;
      for (let i = 0; i < outputs.length; i++) {
        const dir = this.dirAfterAction(i);
        if (this.wouldCollide(dir)) continue;
        if (outputs[i] > bestSafeVal) {
          bestSafeVal = outputs[i];
          bestSafeIndex = i;
        }
      }
      if (bestSafeIndex !== -1) chosenIndex = bestSafeIndex;
    }

    this.newDirection = this.dirAfterAction(chosenIndex);
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

    // Wall Collision
    if (
      newHead.x < 0 ||
      newHead.x >= GRID_WIDTH ||
      newHead.y < 0 ||
      newHead.y >= GRID_HEIGHT
    ) {
      this.dead = true;
      return;
    }

    // Self Collision
    for (let i = 0; i < this.body.length; i++) {
      // If we are not growing, the tail (last segment) will move away, so ignore it.
      if (!this.growPending && i === this.body.length - 1) {
        continue;
      }
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
      // Cap movesLeft to prevent infinite stalling
      if (this.movesLeft > 5000) this.movesLeft = 5000;
    }
  }

  grow() {
    this.growPending = true;
  }

  draw(
    ctx: CanvasRenderingContext2D,
    tileSize: number,
    color: string = "#4caf50",
  ) {
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
    // 1. Reward food eaten heavily (exponential)
    // 2. Reward lifetime slightly (linear)
    const startLength = 3;
    const foodEaten = this.body.length - startLength;
    const progressBonus = Math.floor(this.foodProgress);

    if (foodEaten < 1) {
      // Just lifetime, but capped so it doesn't overshadow eating.
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
