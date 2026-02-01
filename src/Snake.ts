import { Point, Direction, GRID_WIDTH, GRID_HEIGHT } from "./types";
import { NeuralNetwork } from "./NeuralNetwork";
import { Food } from "./Food";

export class Snake {
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
    this.body.forEach(p => this.bodySet.add(p.y * GRID_WIDTH + p.x));
    
    // Pre-allocate input array to avoid GC
    this.visionInputs = new Float32Array(27);

    this.direction = Direction.Right;
    this.newDirection = Direction.Right;
    this.growPending = false;
    this.dead = false;

    // Inputs:
    // 24 inputs (8 directions * 3 types(wall, food, tail))
    // + 2 inputs (Relative Food Vector: Forward, Right)
    // + 1 input (Normalized Length: current/maxCells)
    // Total = 27 inputs
    if (brain) {
      this.brain = brain.clone();
    } else {
      this.brain = new NeuralNetwork(27, 24, 3);
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

    // Standard 8 directions (Clockwise from Up)
    const globalDirs = [
      { x: 0, y: -1 },  // 0: Up
      { x: 1, y: -1 },  // 1: Up-Right
      { x: 1, y: 0 },   // 2: Right
      { x: 1, y: 1 },   // 3: Down-Right
      { x: 0, y: 1 },   // 4: Down
      { x: -1, y: 1 },  // 5: Down-Left
      { x: -1, y: 0 },  // 6: Left
      { x: -1, y: -1 }  // 7: Up-Left
    ];

    // Determine rotation index based on current heading
    // We want index 0 to be "Forward", index 2 to be "Right", etc.
    let startIndex = 0;
    switch (this.direction) {
      case Direction.Up:    startIndex = 0; break;
      case Direction.Right: startIndex = 2; break;
      case Direction.Down:  startIndex = 4; break;
      case Direction.Left:  startIndex = 6; break;
    }

    // 1. Raycasting Vision (8 Relative Directions)
    for (let i = 0; i < 8; i++) {
      // Wrap around the globalDirs array
      const dirIndex = (startIndex + i) % 8;
      this.lookInDirection(globalDirs[dirIndex], food, i * 3);
    }

    // 2. Relative Food Vector (2 Inputs)
    // Calculate food position relative to head
    const dx = food.position.x - head.x;
    const dy = food.position.y - head.y;

    // Rotate vector based on heading and normalize
    let forwardNorm = 0;
    let rightNorm = 0;

    switch (this.direction) {
      case Direction.Up:
        forwardNorm = -dy / GRID_HEIGHT; // Forward is -y
        rightNorm = dx / GRID_WIDTH;     // Right is +x
        break;
      case Direction.Right:
        forwardNorm = dx / GRID_WIDTH;   // Forward is +x
        rightNorm = dy / GRID_HEIGHT;    // Right is +y
        break;
      case Direction.Down:
        forwardNorm = dy / GRID_HEIGHT;  // Forward is +y
        rightNorm = -dx / GRID_WIDTH;    // Right is -x
        break;
      case Direction.Left:
        forwardNorm = -dx / GRID_WIDTH;  // Forward is -x
        rightNorm = -dy / GRID_HEIGHT;   // Right is -y
        break;
    }

    const offsetFood = 24;
    this.visionInputs[offsetFood] = forwardNorm;
    this.visionInputs[offsetFood + 1] = rightNorm;

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
      pos.x >= 0 && pos.x < GRID_WIDTH &&
      pos.y >= 0 && pos.y < GRID_HEIGHT
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

    if (maxIndex === 0) {
      // Straight: Keep current direction
      this.newDirection = this.direction;
    } else if (maxIndex === 1) {
      // Turn Left
      switch (this.direction) {
        case Direction.Up: this.newDirection = Direction.Left; break;
        case Direction.Down: this.newDirection = Direction.Right; break;
        case Direction.Left: this.newDirection = Direction.Down; break;
        case Direction.Right: this.newDirection = Direction.Up; break;
      }
    } else if (maxIndex === 2) {
      // Turn Right
      switch (this.direction) {
        case Direction.Up: this.newDirection = Direction.Right; break;
        case Direction.Down: this.newDirection = Direction.Left; break;
        case Direction.Left: this.newDirection = Direction.Up; break;
        case Direction.Right: this.newDirection = Direction.Down; break;
      }
    }
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
    const head = this.body[0];
    let newHead: Point;

    switch (this.direction) {
      case Direction.Up:
        newHead = { x: head.x, y: head.y - 1 };
        break;
      case Direction.Down:
        newHead = { x: head.x, y: head.y + 1 };
        break;
      case Direction.Left:
        newHead = { x: head.x - 1, y: head.y };
        break;
      case Direction.Right:
        newHead = { x: head.x + 1, y: head.y };
        break;
    }

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
        this.foodProgress += 1;
      } else if (newFoodDist > prevFoodDist) {
        this.foodProgress = Math.max(0, this.foodProgress - 1);
      } else {
        this.foodProgress = Math.max(0, this.foodProgress - 0.25);
      }
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
    // New Fitness Formula
    // 1. Reward Food Eaten heavily (exponential)
    // 2. Reward Lifetime slightly (linear)

    const startLength = 3;
    const foodEaten = this.body.length - startLength;

    // If they haven't eaten, fitness is low.
    const progressBonus = Math.floor(this.foodProgress);

    if (foodEaten < 1) {
      // Just lifetime, but capped so it doesn't overshadow eating.
      this.fitness = Math.floor(this.lifetime / 10) + progressBonus;
    } else {
      // Eaten at least once.
      // foodEaten^2 * 100 (Exponential reward) + lifetime
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
