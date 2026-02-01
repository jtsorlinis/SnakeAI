import { Point, GRID_WIDTH, GRID_HEIGHT } from './types';

export class Food {
  position: Point;

  constructor() {
    this.position = this.getRandomPosition();
  }

  getRandomPosition(): Point {
    return {
      x: Math.floor(Math.random() * GRID_WIDTH),
      y: Math.floor(Math.random() * GRID_HEIGHT),
    };
  }

  respawn(snakeBodySet: Set<number>) {
    let validPosition = false;
    while (!validPosition) {
      this.position = this.getRandomPosition();
      const key = this.position.y * GRID_WIDTH + this.position.x;
      validPosition = !snakeBodySet.has(key);
    }
  }

  draw(ctx: CanvasRenderingContext2D, tileSize: number) {
    ctx.fillStyle = '#f44336';
    ctx.fillRect(
      this.position.x * tileSize,
      this.position.y * tileSize,
      tileSize,
      tileSize
    );
  }
}
