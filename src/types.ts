export interface Point {
  x: number;
  y: number;
}

export enum Direction {
  Up,
  Down,
  Left,
  Right,
}

export const TILE_SIZE = 20;
export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 600;
export const GRID_WIDTH = CANVAS_WIDTH / TILE_SIZE;
export const GRID_HEIGHT = CANVAS_HEIGHT / TILE_SIZE;
