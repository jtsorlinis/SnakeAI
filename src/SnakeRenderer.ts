import {
  BOARD_VIEW_SIZE,
  GRID_SIZE,
  HIDDEN_LAYER_UNITS,
  INPUTS,
  INPUT_LABELS,
  NET_HEIGHT,
  NET_HOVER_RADIUS,
  NET_WIDTH,
  OFFSET_HH,
  OFFSET_HO,
  OFFSET_IH,
  OFFSET_O_BIAS,
  OUTPUTS,
  OUTPUT_LABELS,
  MULTI_VIEW_COLUMNS,
} from "./config";
import type { Agent, Point, TrainerState } from "./types";

type RendererElements = {
  netCanvas: HTMLCanvasElement;
  boardCanvas: HTMLCanvasElement;
  chartCanvas: HTMLCanvasElement;
  // Holds the [data-stat] text fields and [data-meter] bars.
  statsRoot: HTMLElement;
};

type NetEdge = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  weight: number;
  label: string;
};

type Rgb = readonly [number, number, number];

const CANVAS_BG = "#0b0e13";
const TILE_ALT = "#0f131a";
const TEXT = "rgba(231, 235, 240, 0.92)";
const TEXT_MUTED = "rgba(139, 149, 165, 0.95)";
const TEXT_FAINT = "rgba(93, 103, 120, 0.95)";
const GRID_LINE = "rgba(255, 255, 255, 0.05)";
const NODE_FILL = "#151a22";
const NODE_RING = "#2c3442";

const POSITIVE: Rgb = [52, 211, 153];
const NEGATIVE: Rgb = [251, 113, 133];
const CHOSEN: Rgb = [251, 191, 36];
const SNAKE_HEAD: Rgb = [110, 231, 183];
const SNAKE_TAIL: Rgb = [4, 120, 87];
const FOOD: Rgb = [251, 113, 133];

const UI_FONT = "Inter, system-ui, sans-serif";
const MONO_FONT = '"JetBrains Mono", ui-monospace, monospace';

function rgba(color: Rgb, alpha = 1): string {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

function mixRgb(from: Rgb, to: Rgb, t: number): string {
  const r = Math.round(from[0] + (to[0] - from[0]) * t);
  const g = Math.round(from[1] + (to[1] - from[1]) * t);
  const b = Math.round(from[2] + (to[2] - from[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function signed(value: number, digits: number): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

// Gap around each snake and food block, for the retro tiled look.
function blockInset(tile: number): number {
  return Math.max(0.5, tile * 0.08);
}

export class SnakeRenderer {
  private readonly net: HTMLCanvasElement;
  private readonly netCtx: CanvasRenderingContext2D;
  private readonly board: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly chart: HTMLCanvasElement;
  private readonly chartCtx: CanvasRenderingContext2D;

  private readonly statFields = new Map<string, HTMLElement[]>();
  private readonly aliveMeter: HTMLElement | null;

  // CSS box sizes reported by the ResizeObserver; applied at the start of the
  // next render so a resize never leaves a cleared canvas on screen.
  private readonly cssSizes = new Map<
    HTMLCanvasElement,
    { width: number; height: number }
  >();

  private showMiniBoardsInLeft = false;
  private netMouse: Point | null = null;

  constructor(elements: RendererElements) {
    this.net = elements.netCanvas;
    this.board = elements.boardCanvas;
    this.chart = elements.chartCanvas;

    this.netCtx = this.net.getContext("2d") as CanvasRenderingContext2D;
    this.ctx = this.board.getContext("2d") as CanvasRenderingContext2D;
    this.chartCtx = this.chart.getContext("2d") as CanvasRenderingContext2D;

    elements.statsRoot
      .querySelectorAll<HTMLElement>("[data-stat]")
      .forEach((element) => {
        const key = element.dataset.stat as string;
        const fields = this.statFields.get(key) ?? [];
        fields.push(element);
        this.statFields.set(key, fields);
      });
    this.aliveMeter = elements.statsRoot.querySelector<HTMLElement>(
      '[data-meter="alive"]',
    );

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        this.cssSizes.set(entry.target as HTMLCanvasElement, {
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    for (const canvas of [this.net, this.board, this.chart]) {
      this.cssSizes.set(canvas, {
        width: canvas.clientWidth,
        height: canvas.clientHeight,
      });
      resizeObserver.observe(canvas);
    }

    this.net.addEventListener("mousemove", (event) => {
      this.netMouse = {
        x: (event.offsetX * NET_WIDTH) / this.net.clientWidth,
        y: (event.offsetY * NET_HEIGHT) / this.net.clientHeight,
      };
    });

    this.net.addEventListener("mouseleave", () => {
      this.netMouse = null;
      this.net.style.cursor = "default";
    });
  }

  public setShowMiniBoardsInLeft(show: boolean): void {
    this.showMiniBoardsInLeft = show;
    if (show) {
      this.net.style.cursor = "default";
    }
  }

  public render(state: TrainerState): void {
    this.beginFrame(this.netCtx, NET_WIDTH, NET_HEIGHT);
    if (this.showMiniBoardsInLeft && state.boardAgents.length > 0) {
      this.drawBoardGridInNet(state.boardAgents);
    } else {
      this.drawNetwork(state);
    }

    this.beginFrame(this.ctx, BOARD_VIEW_SIZE, BOARD_VIEW_SIZE);
    this.drawSingleBoard(state.boardAgent);

    const chartSize = this.cssSizes.get(this.chart);
    const chartWidth = Math.max(1, chartSize?.width ?? 0);
    const chartHeight = Math.max(1, chartSize?.height ?? 0);
    this.beginFrame(this.chartCtx, chartWidth, chartHeight);
    this.drawHistory(state.fitnessHistory, chartWidth, chartHeight);

    this.updateStats(state);
  }

  // Sizes the backing store to the canvas's on-screen pixels and maps the
  // given drawing-space size onto it.
  private beginFrame(
    context: CanvasRenderingContext2D,
    width: number,
    height: number,
  ): void {
    const canvas = context.canvas;
    const cssSize = this.cssSizes.get(canvas);
    if (cssSize && cssSize.width > 0 && cssSize.height > 0) {
      const ratio = window.devicePixelRatio || 1;
      const pixelWidth = Math.round(cssSize.width * ratio);
      const pixelHeight = Math.round(cssSize.height * ratio);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
    }

    context.setTransform(
      canvas.width / width,
      0,
      0,
      canvas.height / height,
      0,
      0,
    );
  }

  private segmentDistanceSquared(
    px: number,
    py: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
  ): number {
    const vx = x2 - x1;
    const vy = y2 - y1;
    const wx = px - x1;
    const wy = py - y1;
    const lengthSq = vx * vx + vy * vy;

    if (lengthSq <= 0.000001) {
      const dx = px - x1;
      const dy = py - y1;
      return dx * dx + dy * dy;
    }

    let t = (wx * vx + wy * vy) / lengthSq;
    t = Math.max(0, Math.min(1, t));
    const cx = x1 + t * vx;
    const cy = y1 + t * vy;
    const dx = px - cx;
    const dy = py - cy;
    return dx * dx + dy * dy;
  }

  private drawNetwork(state: TrainerState): void {
    const { policy, activations } = state.network;
    const ctx = this.netCtx;

    ctx.fillStyle = CANVAS_BG;
    ctx.fillRect(0, 0, NET_WIDTH, NET_HEIGHT);

    if (!policy || !activations) {
      return;
    }

    let outputAbsMax = 0.001;
    for (let o = 0; o < OUTPUTS; o++) {
      outputAbsMax = Math.max(outputAbsMax, Math.abs(activations.output[o]));
    }

    const hiddenLayerCount = HIDDEN_LAYER_UNITS.length;
    const top = 50;
    const bottom = NET_HEIGHT - 34;
    const inputX = 104;
    const outputX = NET_WIDTH - 108;

    const inputY: number[] = [];
    const hiddenX: number[] = [];
    const hiddenY: number[][] = [];
    const outputY: number[] = [];

    for (let i = 0; i < INPUTS; i++) {
      inputY.push(top + (i * (bottom - top)) / Math.max(1, INPUTS - 1));
    }

    for (let layer = 0; layer < hiddenLayerCount; layer++) {
      hiddenX.push(
        inputX + ((layer + 1) * (outputX - inputX)) / (hiddenLayerCount + 1),
      );
    }

    for (let layer = 0; layer < hiddenLayerCount; layer++) {
      const units = HIDDEN_LAYER_UNITS[layer];
      const positions: number[] = [];
      for (let h = 0; h < units; h++) {
        positions.push(top + (h * (bottom - top)) / Math.max(1, units - 1));
      }
      hiddenY.push(positions);
    }

    for (let o = 0; o < OUTPUTS; o++) {
      outputY.push(top + (o * (bottom - top)) / Math.max(1, OUTPUTS - 1));
    }

    const edges: NetEdge[] = [];

    if (hiddenLayerCount > 0) {
      const firstHiddenX = hiddenX[0];
      const firstHiddenY = hiddenY[0];
      const firstSize = HIDDEN_LAYER_UNITS[0];
      for (let h = 0; h < firstSize; h++) {
        const wOffset = OFFSET_IH + h * INPUTS;
        for (let i = 0; i < INPUTS; i++) {
          edges.push({
            x1: inputX,
            y1: inputY[i],
            x2: firstHiddenX,
            y2: firstHiddenY[h],
            weight: policy[wOffset + i],
            label: `${INPUT_LABELS[i]} → H1:${h + 1}`,
          });
        }
      }

      let hhOffset = OFFSET_HH;
      for (let layer = 1; layer < hiddenLayerCount; layer++) {
        const prevSize = HIDDEN_LAYER_UNITS[layer - 1];
        const currSize = HIDDEN_LAYER_UNITS[layer];
        const fromX = hiddenX[layer - 1];
        const toX = hiddenX[layer];
        const fromY = hiddenY[layer - 1];
        const toY = hiddenY[layer];

        for (let to = 0; to < currSize; to++) {
          const wOffset = hhOffset + to * prevSize;
          for (let from = 0; from < prevSize; from++) {
            edges.push({
              x1: fromX,
              y1: fromY[from],
              x2: toX,
              y2: toY[to],
              weight: policy[wOffset + from],
              label: `H${layer}:${from + 1} → H${layer + 1}:${to + 1}`,
            });
          }
        }

        hhOffset += prevSize * currSize;
      }

      const lastHiddenX = hiddenX[hiddenLayerCount - 1];
      const lastHiddenY = hiddenY[hiddenLayerCount - 1];
      const lastSize = HIDDEN_LAYER_UNITS[hiddenLayerCount - 1];
      for (let o = 0; o < OUTPUTS; o++) {
        const wOffset = OFFSET_HO + o * lastSize;
        for (let h = 0; h < lastSize; h++) {
          edges.push({
            x1: lastHiddenX,
            y1: lastHiddenY[h],
            x2: outputX,
            y2: outputY[o],
            weight: policy[wOffset + h],
            label: `H${hiddenLayerCount}:${h + 1} → ${OUTPUT_LABELS[o]}`,
          });
        }
      }
    } else {
      for (let o = 0; o < OUTPUTS; o++) {
        const wOffset = OFFSET_HO + o * INPUTS;
        for (let i = 0; i < INPUTS; i++) {
          edges.push({
            x1: inputX,
            y1: inputY[i],
            x2: outputX,
            y2: outputY[o],
            weight: policy[wOffset + i],
            label: `${INPUT_LABELS[i]} → ${OUTPUT_LABELS[o]}`,
          });
        }
      }
    }

    let maxAbs = 0.001;
    for (const edge of edges) {
      maxAbs = Math.max(maxAbs, Math.abs(edge.weight));
    }

    let hoveredEdge: NetEdge | null = null;
    if (this.netMouse) {
      let bestDistSq = NET_HOVER_RADIUS * NET_HOVER_RADIUS;
      for (const edge of edges) {
        const distSq = this.segmentDistanceSquared(
          this.netMouse.x,
          this.netMouse.y,
          edge.x1,
          edge.y1,
          edge.x2,
          edge.y2,
        );

        if (distSq <= bestDistSq) {
          bestDistSq = distSq;
          hoveredEdge = edge;
        }
      }
    }

    this.net.style.cursor = hoveredEdge ? "pointer" : "default";

    // Strong weights last so they sit on top; everything fades while hovering.
    edges.sort((a, b) => Math.abs(a.weight) - Math.abs(b.weight));
    const edgeAlpha = hoveredEdge ? 0.35 : 1;
    ctx.lineCap = "round";
    for (const edge of edges) {
      const strength = Math.min(1, Math.abs(edge.weight) / maxAbs);
      ctx.strokeStyle = rgba(
        edge.weight >= 0 ? POSITIVE : NEGATIVE,
        (0.06 + strength * 0.62) * edgeAlpha,
      );
      ctx.lineWidth = 0.6 + strength * 2.2;
      ctx.beginPath();
      ctx.moveTo(edge.x1, edge.y1);
      ctx.lineTo(edge.x2, edge.y2);
      ctx.stroke();
    }

    if (hoveredEdge) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(hoveredEdge.x1, hoveredEdge.y1);
      ctx.lineTo(hoveredEdge.x2, hoveredEdge.y2);
      ctx.stroke();
    }

    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillStyle = TEXT_FAINT;
    ctx.font = `600 10px ${UI_FONT}`;
    ctx.fillText("INPUT", inputX, 20);
    for (let layer = 0; layer < hiddenLayerCount; layer++) {
      ctx.fillText(
        hiddenLayerCount > 1 ? `HIDDEN ${layer + 1}` : "HIDDEN",
        hiddenX[layer],
        20,
      );
    }
    if (hiddenLayerCount === 0) {
      ctx.fillText("DIRECT", NET_WIDTH / 2, 20);
    }
    ctx.fillText("OUTPUT", outputX, 20);

    ctx.textAlign = "right";
    ctx.font = `500 11px ${UI_FONT}`;
    for (let i = 0; i < INPUTS; i++) {
      const value = activations.input[i];
      const intensity = Math.min(1, Math.abs(value));
      this.drawNode(inputX, inputY[i], 5.5, value, intensity);

      ctx.fillStyle = intensity > 0.02 ? TEXT : TEXT_MUTED;
      ctx.fillText(INPUT_LABELS[i], inputX - 14, inputY[i]);
    }

    for (let layer = 0; layer < hiddenLayerCount; layer++) {
      const layerActivation = activations.hidden[layer];
      const layerY = hiddenY[layer];
      for (let h = 0; h < layerY.length; h++) {
        const value = layerActivation[h];
        this.drawNode(
          hiddenX[layer],
          layerY[h],
          5.5,
          value,
          Math.min(1, Math.abs(value)),
        );
      }
    }

    ctx.textAlign = "left";
    for (let o = 0; o < OUTPUTS; o++) {
      const value = activations.output[o];
      const intensity = Math.min(1, Math.abs(value) / outputAbsMax);
      const chosen = o === activations.best;
      this.drawNode(outputX, outputY[o], 7, value, intensity);

      if (chosen) {
        ctx.strokeStyle = rgba(CHOSEN, 0.95);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(outputX, outputY[o], 11.5, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.fillStyle = chosen ? rgba(CHOSEN) : TEXT;
      ctx.font = `${chosen ? 600 : 500} 12px ${UI_FONT}`;
      ctx.fillText(OUTPUT_LABELS[o], outputX + 18, outputY[o] - 6);
      ctx.fillStyle = TEXT_FAINT;
      ctx.font = `10px ${MONO_FONT}`;
      ctx.fillText(
        `a${signed(value, 2)} b${signed(policy[OFFSET_O_BIAS + o], 2)}`,
        outputX + 18,
        outputY[o] + 9,
      );
    }

    if (hoveredEdge && this.netMouse) {
      this.drawEdgeTooltip(hoveredEdge, this.netMouse);
    }
  }

  private drawNode(
    x: number,
    y: number,
    radius: number,
    value: number,
    intensity: number,
  ): void {
    const ctx = this.netCtx;
    const color = value >= 0 ? POSITIVE : NEGATIVE;
    const active = intensity > 0.02;

    if (active) {
      ctx.fillStyle = rgba(color, 0.06 + intensity * 0.2);
      ctx.beginPath();
      ctx.arc(x, y, radius + 3 + intensity * 5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = NODE_FILL;
    ctx.fill();
    if (active) {
      ctx.fillStyle = rgba(color, 0.2 + intensity * 0.8);
      ctx.fill();
    }
    ctx.strokeStyle = active ? rgba(color, 0.9) : NODE_RING;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  private drawEdgeTooltip(edge: NetEdge, mouse: Point): void {
    const ctx = this.netCtx;
    const line1 = edge.label;
    const line2 = `weight ${signed(edge.weight, 4)}`;

    ctx.font = `500 11px ${UI_FONT}`;
    const width1 = ctx.measureText(line1).width;
    ctx.font = `11px ${MONO_FONT}`;
    const width2 = ctx.measureText(line2).width;

    const boxWidth = Math.max(width1, width2) + 20;
    const boxHeight = 44;
    const x = Math.max(6, Math.min(NET_WIDTH - boxWidth - 6, mouse.x + 12));
    let y = mouse.y - boxHeight - 12;
    if (y < 6) {
      y = mouse.y + 16;
    }

    ctx.beginPath();
    ctx.roundRect(x, y, boxWidth, boxHeight, 8);
    ctx.fillStyle = "rgba(23, 28, 37, 0.97)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = `500 11px ${UI_FONT}`;
    ctx.fillStyle = TEXT;
    ctx.fillText(line1, x + 10, y + 15);
    ctx.font = `11px ${MONO_FONT}`;
    ctx.fillStyle = rgba(edge.weight >= 0 ? POSITIVE : NEGATIVE);
    ctx.fillText(line2, x + 10, y + 30);
  }

  private drawSingleBoard(agent: Agent): void {
    this.drawBoardAt(this.ctx, agent, 0, 0, BOARD_VIEW_SIZE);
  }

  private drawBoardGridInNet(agents: readonly Agent[]): void {
    this.drawBoardGrid(this.netCtx, NET_WIDTH, NET_HEIGHT, agents);
    this.net.style.cursor = "default";
  }

  private drawBoardGrid(
    context: CanvasRenderingContext2D,
    canvasWidth: number,
    canvasHeight: number,
    agents: readonly Agent[],
  ): void {
    context.fillStyle = CANVAS_BG;
    context.fillRect(0, 0, canvasWidth, canvasHeight);

    const columns = Math.max(1, MULTI_VIEW_COLUMNS);
    const rows = Math.max(1, Math.ceil(agents.length / columns));
    const padding = 12;
    const gap = 8;

    const boardSizeByWidth =
      (canvasWidth - padding * 2 - gap * (columns - 1)) / columns;
    const boardSizeByHeight =
      (canvasHeight - padding * 2 - gap * (rows - 1)) / rows;
    const cellSize = Math.max(8, Math.min(boardSizeByWidth, boardSizeByHeight));

    const gridWidth = cellSize * columns + gap * (columns - 1);
    const gridHeight = cellSize * rows + gap * (rows - 1);
    const startX = (canvasWidth - gridWidth) / 2;
    const startY = (canvasHeight - gridHeight) / 2;

    for (let i = 0; i < agents.length; i++) {
      const row = Math.floor(i / columns);
      const col = i % columns;
      const x = startX + col * (cellSize + gap);
      const y = startY + row * (cellSize + gap);
      const agent = agents[i];

      context.save();
      context.beginPath();
      context.roundRect(x, y, cellSize, cellSize, 6);
      context.clip();

      this.drawBoardAt(context, agent, x, y, cellSize);

      if (!agent.alive) {
        context.fillStyle = "rgba(11, 14, 19, 0.65)";
        context.fillRect(x, y, cellSize, cellSize);
      }
      context.restore();

      context.beginPath();
      context.roundRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1, 6);
      context.strokeStyle = agent.alive
        ? rgba(POSITIVE, 0.3)
        : rgba(NEGATIVE, 0.3);
      context.lineWidth = 1;
      context.stroke();
    }
  }

  private drawBoardAt(
    context: CanvasRenderingContext2D,
    agent: Agent,
    x: number,
    y: number,
    size: number,
  ): void {
    const tile = size / GRID_SIZE;

    context.save();
    context.translate(x, y);

    context.fillStyle = CANVAS_BG;
    context.fillRect(0, 0, size, size);

    // Checkerboard only when tiles are big enough not to shimmer.
    if (tile >= 8) {
      context.fillStyle = TILE_ALT;
      context.beginPath();
      for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 1 - (row % 2); col < GRID_SIZE; col += 2) {
          context.rect(col * tile, row * tile, tile, tile);
        }
      }
      context.fill();
    }

    this.drawFood(context, agent.food, tile);
    this.drawSnake(context, agent, tile);

    context.restore();
  }

  private drawFood(
    context: CanvasRenderingContext2D,
    food: Point,
    tile: number,
  ): void {
    const inset = blockInset(tile);

    context.fillStyle = rgba(FOOD, 0.14);
    context.fillRect(food.x * tile, food.y * tile, tile, tile);

    context.fillStyle = rgba(FOOD);
    context.fillRect(
      food.x * tile + inset,
      food.y * tile + inset,
      tile - inset * 2,
      tile - inset * 2,
    );
  }

  private drawSnake(
    context: CanvasRenderingContext2D,
    agent: Agent,
    tile: number,
  ): void {
    const { body } = agent;
    if (body.length === 0) {
      return;
    }

    // Retro blocks: one square per cell with a small gap, fading from head
    // to tail.
    const tailIndex = Math.max(1, body.length - 1);
    const inset = blockInset(tile);
    const blockSize = tile - inset * 2;
    for (let i = body.length - 1; i >= 0; i--) {
      context.fillStyle = mixRgb(SNAKE_HEAD, SNAKE_TAIL, i / tailIndex);
      context.fillRect(
        body[i].x * tile + inset,
        body[i].y * tile + inset,
        blockSize,
        blockSize,
      );
    }
  }

  private drawHistory(
    history: readonly number[],
    width: number,
    height: number,
  ): void {
    const ctx = this.chartCtx;
    const padX = 10;
    const padTop = 12;
    const padBottom = 10;
    const plotHeight = height - padTop - padBottom;
    const baseline = padTop + plotHeight;

    ctx.fillStyle = CANVAS_BG;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = GRID_LINE;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = Math.round(padTop + (plotHeight * i) / 3) + 0.5;
      ctx.beginPath();
      ctx.moveTo(padX, y);
      ctx.lineTo(width - padX, y);
      ctx.stroke();
    }

    if (history.length < 2) {
      ctx.fillStyle = TEXT_FAINT;
      ctx.font = `12px ${UI_FONT}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Collecting data…", width / 2, height / 2);
      this.setStat("historyMax", "–");
      return;
    }

    let maxFitness = 1;
    for (const fitness of history) {
      if (fitness > maxFitness) {
        maxFitness = fitness;
      }
    }

    const pointX = (i: number) =>
      padX + (i / (history.length - 1)) * (width - padX * 2);
    const pointY = (fitness: number) =>
      baseline - (fitness / maxFitness) * plotHeight;

    const line = new Path2D();
    for (let i = 0; i < history.length; i++) {
      if (i === 0) {
        line.moveTo(pointX(i), pointY(history[i]));
      } else {
        line.lineTo(pointX(i), pointY(history[i]));
      }
    }

    const area = new Path2D(line);
    area.lineTo(pointX(history.length - 1), baseline);
    area.lineTo(pointX(0), baseline);
    area.closePath();
    const fill = ctx.createLinearGradient(0, padTop, 0, baseline);
    fill.addColorStop(0, rgba(POSITIVE, 0.24));
    fill.addColorStop(1, rgba(POSITIVE, 0));
    ctx.fillStyle = fill;
    ctx.fill(area);

    ctx.lineJoin = "round";
    ctx.strokeStyle = rgba(POSITIVE);
    ctx.lineWidth = 2;
    ctx.stroke(line);

    const lastX = pointX(history.length - 1);
    const lastY = pointY(history[history.length - 1]);
    ctx.fillStyle = rgba(POSITIVE, 0.25);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = rgba(POSITIVE);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    ctx.fill();

    this.setStat("historyMax", maxFitness.toFixed(2));
  }

  private updateStats(state: TrainerState): void {
    const { replay } = state;
    const hasReplay = replay.games > 0;

    this.setStat("iterationLabel", state.iterationLabel);
    this.setStat("iteration", state.iteration.toLocaleString());
    this.setStat("bestScore", String(state.bestEverScore));
    this.setStat("alive", String(state.alive));
    this.setStat("batchSize", String(state.batchSize));
    this.setStat("batchSizeLabel", state.batchSizeLabel);
    this.setStat(
      "replayAverage",
      hasReplay ? replay.averageScore.toFixed(1) : "–",
    );
    this.setStat(
      "replayWins",
      hasReplay ? `${replay.wins}/${replay.games} wins` : "",
    );
    this.setStat("bestFitness", state.bestEverFitness.toFixed(2));
    this.setStat("staleLabel", state.staleLabel);
    this.setStat("stale", String(state.staleIterations));
    this.setStat("historyLabel", state.historyLabel);
    this.setStat("boardScore", String(state.boardAgent.score));

    if (this.aliveMeter) {
      const ratio = state.batchSize > 0 ? state.alive / state.batchSize : 0;
      this.aliveMeter.style.transform = `scaleX(${ratio})`;
    }
  }

  private setStat(key: string, value: string): void {
    for (const element of this.statFields.get(key) ?? []) {
      if (element.textContent !== value) {
        element.textContent = value;
      }
    }
  }
}
