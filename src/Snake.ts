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
    this.visionInputs = new Float32Array(28);

    this.direction = Direction.Right;
    this.newDirection = Direction.Right;
    this.growPending = false;
    this.dead = false;

    // Inputs:
    // 24 inputs (8 directions * 3 types(wall, food, tail))
    // + 4 inputs (Food direction: Up, Down, Left, Right)
    // Total = 28 inputs
    if (brain) {
      this.brain = brain.clone();
    } else {
      this.brain = new NeuralNetwork(28, 48, 4);
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
    // 1. Raycasting Vision (8 Directions)
    const dirs = [
      { x: 0, y: -1 },  // Up
      { x: 1, y: -1 },  // Up-Right
      { x: 1, y: 0 },   // Right
      { x: 1, y: 1 },   // Down-Right
      { x: 0, y: 1 },   // Down
      { x: -1, y: 1 },  // Down-Left
      { x: -1, y: 0 },  // Left
      { x: -1, y: -1 }  // Up-Left
    ];

    for (let i = 0; i < dirs.length; i++) {
      this.lookInDirection(dirs[i], food, i * 3);
    }

    // 2. Global Food Direction (4 Inputs)
    // This gives a "smell" of where the food is, helping when rays miss
    const head = this.body[0];
    const offset = 24;
    this.visionInputs[offset] = food.position.y < head.y ? 1 : 0; // Food is Up
    this.visionInputs[offset + 1] = food.position.y > head.y ? 1 : 0; // Food is Down
    this.visionInputs[offset + 2] = food.position.x < head.x ? 1 : 0; // Food is Left
    this.visionInputs[offset + 3] = food.position.x > head.x ? 1 : 0; // Food is Right

    return this.visionInputs;
  }

  lookInDirection(dir: Point, food: Food, offset: number) {
    // [Wall, Food, Self] stored at offset, offset+1, offset+2
    let pos = { x: this.body[0].x, y: this.body[0].y };
    let distance = 0;
    let foundFood = false;
    let foundBody = false;
    
    // Reset values
    this.visionInputs[offset] = 0;
    this.visionInputs[offset + 1] = 0;
    this.visionInputs[offset + 2] = 0;

    pos.x += dir.x;
    pos.y += dir.y;
    distance += 1;

    // Move in direction until hit wall
    while (
      pos.x >= 0 && pos.x < GRID_WIDTH &&
      pos.y >= 0 && pos.y < GRID_HEIGHT
    ) {
      
      // Check for food
      if (!foundFood && pos.x === food.position.x && pos.y === food.position.y) {
        this.visionInputs[offset + 1] = 1; 
        foundFood = true;
      }

      // Check for body
      if (!foundBody && this.isOnBody(pos)) {
        this.visionInputs[offset + 2] = 1 / distance;
        foundBody = true;
      }

      pos.x += dir.x;
      pos.y += dir.y;
      distance += 1;
    }

    // Distance to wall (closer is higher number)
    this.visionInputs[offset] = 1 / distance;
  }

  isOnBody(pt: Point): boolean {
    return this.bodySet.has(pt.y * GRID_WIDTH + pt.x);
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

    this.setDirection(maxIndex);
  }

  move() {
    if (this.dead) return;

    this.lifetime++;
    this.movesLeft--;

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
    // 1. Reward Length heavily (exponential)
    // 2. Reward Lifetime slightly (linear)
    // 3. To differentiate young snakes: Reward getting closer to food?
    //    Actually, keep it simple first.

    const length = this.body.length - 1;

    // If they haven't eaten, fitness is low.
    if (length < 1) {
      // Just lifetime, but capped so it doesn't overshadow eating.
      this.fitness = Math.floor(this.lifetime / 10);
    } else {
      // Eaten at least once.
      // Length^2 * 100 (Exponential reward) + lifetime
      this.fitness = length * length * 100 + Math.floor(this.lifetime / 2);
    }

    // Bonus for finding food recently? No, simplicity is key.
  }

  mutate(rate: number) {
    this.brain.mutate(rate);
  }
}
