import { DIRS, GRID_SIZE } from "./config";
import type { Agent, Point } from "./types";

// Layers of the view, each one on/off value per cell.
const PLANE_INSIDE = 0;
const PLANE_BODY = 1;
const PLANE_TAIL = 2;
const PLANE_FOOD = 3;
const PLANE_COUNT = 4;

// Cells per side of the view: wide enough to hold the whole board wherever
// the head is.
export function visionViewSize(): number {
  return 2 * GRID_SIZE - 1;
}

export function visionInputCount(): number {
  const size = visionViewSize();
  return size * size * PLANE_COUNT;
}

// Roughly how many inputs are on at once, for scaling a network's starting
// weights.
export function visionTypicalInputs(): number {
  return GRID_SIZE * GRID_SIZE * 2;
}

// What Board PPO's policy sees. Only things drawn on the board go in: which
// cells are on the board, and where the body, the tail tip and the food are.
// The board is re-drawn with the head in the middle and turned so the snake
// faces up, so "a wall two cells ahead" looks the same wherever the snake is
// and whichever way it is heading. Fed the board as drawn, the same network
// learned several times more slowly.
//
// Inputs are produced as a list of the ones that are switched on, for
// VisionNetwork. One spare slot is left after them for a caller to append an
// extra input.
export class BoardVision {
  public readonly indices: Int32Array;
  public readonly values: Float32Array;
  public count = 0;

  private readonly grid = GRID_SIZE;
  private readonly size = visionViewSize();
  private readonly area = this.size * this.size;

  constructor() {
    // Every board cell, plus one more entry per body cell, the food and the
    // spare slot.
    const capacity = this.grid * this.grid * 2 + 2;
    this.indices = new Int32Array(capacity);
    this.values = new Float32Array(capacity);
  }

  public observe(agent: Agent): void {
    const { indices, area, grid } = this;
    const head = agent.body[0];
    const tail = agent.body.length - 1;
    let n = 0;

    // Marking the board's cells shows where all four walls are. Two wall
    // distances would say the same with far fewer inputs, and train about
    // twice as fast per update, but cleared a few percent fewer games.
    for (let y = 0; y < grid; y++) {
      for (let x = 0; x < grid; x++) {
        indices[n++] =
          PLANE_INSIDE * area + this.cellIndex(x, y, head, agent.dir);
      }
    }

    // The head is always the middle cell, so it needs no entry. The tail tip
    // moves away before the head can reach it, so it gets its own layer
    // instead of counting as body.
    for (let i = 1; i <= tail; i++) {
      const part = agent.body[i];
      indices[n++] =
        (i === tail ? PLANE_TAIL : PLANE_BODY) * area +
        this.cellIndex(part.x, part.y, head, agent.dir);
    }

    indices[n++] =
      PLANE_FOOD * area +
      this.cellIndex(agent.food.x, agent.food.y, head, agent.dir);

    // Every input here is simply on. A caller may have used the spare slot
    // for something else last time, so the values are rewritten each time.
    this.values.fill(1, 0, n);
    this.count = n;
  }

  // Where a board cell lands in the view: head in the middle, the snake's
  // heading pointing up.
  public cellIndex(x: number, y: number, head: Point, dir: number): number {
    const forward = DIRS[dir];
    const right = DIRS[(dir + 1) % 4];
    const dx = x - head.x;
    const dy = y - head.y;
    const across = dx * right.x + dy * right.y;
    const ahead = dx * forward.x + dy * forward.y;
    return (this.grid - 1 - ahead) * this.size + (across + this.grid - 1);
  }
}
