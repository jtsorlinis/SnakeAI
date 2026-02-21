import {
  BOARD_SIZE,
  CHART_HEIGHT,
  CHART_WIDTH,
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
  TILE_SIZE,
  MULTI_VIEW_COLUMNS,
} from "./config";
import type { Agent, Point, TrainerState } from "./types";

type RendererElements = {
  netCanvas: HTMLCanvasElement;
  boardCanvas: HTMLCanvasElement;
  chartCanvas: HTMLCanvasElement;
  statsElement: HTMLElement;
};

type NetEdge = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  weight: number;
  label: string;
};

export class SnakeRenderer {
  private readonly net: HTMLCanvasElement;
  private readonly netCtx: CanvasRenderingContext2D;
  private readonly board: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly chart: HTMLCanvasElement;
  private readonly chartCtx: CanvasRenderingContext2D;
  private readonly stats: HTMLElement;

  private showMiniBoardsInLeft = false;
  private netMouse: Point | null = null;

  constructor(elements: RendererElements) {
    this.net = elements.netCanvas;
    this.board = elements.boardCanvas;
    this.chart = elements.chartCanvas;
    this.stats = elements.statsElement;

    this.netCtx = this.net.getContext("2d") as CanvasRenderingContext2D;
    this.ctx = this.board.getContext("2d") as CanvasRenderingContext2D;
    this.chartCtx = this.chart.getContext("2d") as CanvasRenderingContext2D;

    this.net.width = NET_WIDTH;
    this.net.height = NET_HEIGHT;
    this.board.width = BOARD_SIZE;
    this.board.height = BOARD_SIZE;
    this.chart.width = CHART_WIDTH;
    this.chart.height = CHART_HEIGHT;

    this.net.addEventListener("mousemove", (event) => {
      const rect = this.net.getBoundingClientRect();
      const scaleX = this.net.width / rect.width;
      const scaleY = this.net.height / rect.height;
      this.netMouse = {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
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
    this.syncBoardSize();

    if (this.showMiniBoardsInLeft && state.boardAgents.length > 0) {
      this.drawBoardGridInNet(state.boardAgents);
    } else {
      this.drawNetwork(state);
    }

    this.drawSingleBoard(state.boardAgent);
    this.drawHistory(state.fitnessHistory);
    this.updateStats(state);
  }

  private syncBoardSize(): void {
    if (this.board.width !== BOARD_SIZE || this.board.height !== BOARD_SIZE) {
      this.board.width = BOARD_SIZE;
      this.board.height = BOARD_SIZE;
    }
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
    const { genome, activations } = state.network;

    this.netCtx.fillStyle = "#000";
    this.netCtx.fillRect(0, 0, NET_WIDTH, NET_HEIGHT);

    if (!genome || !activations) {
      return;
    }

    let outputAbsMax = 0.001;
    for (let o = 0; o < OUTPUTS; o++) {
      outputAbsMax = Math.max(outputAbsMax, Math.abs(activations.output[o]));
    }

    const hiddenLayerCount = HIDDEN_LAYER_UNITS.length;
    const top = 24;
    const bottom = NET_HEIGHT - 24;
    const inputX = 84;
    const outputX = NET_WIDTH - 84;

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
            weight: genome[wOffset + i],
            label: `${INPUT_LABELS[i]} -> H1:${h + 1}`,
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
              weight: genome[wOffset + from],
              label: `H${layer}:${from + 1} -> H${layer + 1}:${to + 1}`,
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
            weight: genome[wOffset + h],
            label: `H${hiddenLayerCount}:${h + 1} -> ${OUTPUT_LABELS[o]}`,
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
            weight: genome[wOffset + i],
            label: `${INPUT_LABELS[i]} -> ${OUTPUT_LABELS[o]}`,
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

    for (const edge of edges) {
      const strength = Math.min(1, Math.abs(edge.weight) / maxAbs);
      this.netCtx.strokeStyle =
        edge.weight >= 0
          ? `rgba(76, 175, 80, ${0.15 + strength * 0.7})`
          : `rgba(244, 67, 54, ${0.15 + strength * 0.7})`;
      this.netCtx.lineWidth = 0.5 + strength * 2;
      this.netCtx.beginPath();
      this.netCtx.moveTo(edge.x1, edge.y1);
      this.netCtx.lineTo(edge.x2, edge.y2);
      this.netCtx.stroke();
    }

    if (hoveredEdge) {
      this.netCtx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      this.netCtx.lineWidth = 3;
      this.netCtx.beginPath();
      this.netCtx.moveTo(hoveredEdge.x1, hoveredEdge.y1);
      this.netCtx.lineTo(hoveredEdge.x2, hoveredEdge.y2);
      this.netCtx.stroke();
    }

    this.netCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
    this.netCtx.font = "12px JetBrains Mono, monospace";
    this.netCtx.fillText("Input", inputX - 28, 14);
    for (let layer = 0; layer < hiddenLayerCount; layer++) {
      this.netCtx.fillText(`H${layer + 1}`, hiddenX[layer] - 10, 14);
    }
    if (hiddenLayerCount === 0) {
      this.netCtx.fillText("Direct", NET_WIDTH / 2 - 20, 14);
    }
    this.netCtx.fillText("Output", outputX - 24, 14);

    for (let i = 0; i < INPUTS; i++) {
      const value = activations.input[i];
      const intensity = Math.min(1, Math.abs(value));

      if (intensity > 0.02) {
        this.netCtx.fillStyle =
          value >= 0
            ? `rgba(41, 182, 246, ${0.1 + intensity * 0.55})`
            : `rgba(255, 183, 77, ${0.1 + intensity * 0.55})`;
        this.netCtx.beginPath();
        this.netCtx.arc(inputX, inputY[i], 7 + intensity * 4, 0, Math.PI * 2);
        this.netCtx.fill();
      }

      this.netCtx.fillStyle = value >= 0 ? "#29b6f6" : "#ffb74d";
      this.netCtx.beginPath();
      this.netCtx.arc(inputX, inputY[i], 5, 0, Math.PI * 2);
      this.netCtx.fill();

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.88)";
      this.netCtx.font = "10px JetBrains Mono, monospace";
      this.netCtx.fillText(INPUT_LABELS[i], 6, inputY[i] + 3);
    }

    for (let layer = 0; layer < hiddenLayerCount; layer++) {
      const layerActivation = activations.hidden[layer];
      const layerY = hiddenY[layer];
      for (let h = 0; h < layerY.length; h++) {
        const value = layerActivation[h];
        const intensity = Math.min(1, Math.abs(value));

        if (intensity > 0.02) {
          this.netCtx.fillStyle =
            value >= 0
              ? `rgba(76, 175, 80, ${0.1 + intensity * 0.6})`
              : `rgba(244, 67, 54, ${0.1 + intensity * 0.6})`;
          this.netCtx.beginPath();
          this.netCtx.arc(
            hiddenX[layer],
            layerY[h],
            7 + intensity * 4,
            0,
            Math.PI * 2,
          );
          this.netCtx.fill();
        }

        this.netCtx.fillStyle = value >= 0 ? "#26a69a" : "#ef5350";
        this.netCtx.beginPath();
        this.netCtx.arc(hiddenX[layer], layerY[h], 5, 0, Math.PI * 2);
        this.netCtx.fill();
      }
    }

    for (let o = 0; o < OUTPUTS; o++) {
      const value = activations.output[o];
      const intensity = Math.min(1, Math.abs(value) / outputAbsMax);

      if (intensity > 0.02) {
        this.netCtx.fillStyle =
          value >= 0
            ? `rgba(76, 175, 80, ${0.12 + intensity * 0.6})`
            : `rgba(244, 67, 54, ${0.12 + intensity * 0.6})`;
        this.netCtx.beginPath();
        this.netCtx.arc(outputX, outputY[o], 8 + intensity * 5, 0, Math.PI * 2);
        this.netCtx.fill();
      }

      this.netCtx.fillStyle = o === activations.best ? "#ffd54f" : "#ffb74d";
      this.netCtx.beginPath();
      this.netCtx.arc(outputX, outputY[o], 6, 0, Math.PI * 2);
      this.netCtx.fill();

      if (o === activations.best) {
        this.netCtx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        this.netCtx.lineWidth = 1.5;
        this.netCtx.beginPath();
        this.netCtx.arc(outputX, outputY[o], 9, 0, Math.PI * 2);
        this.netCtx.stroke();
      }

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
      this.netCtx.font = "11px JetBrains Mono, monospace";
      this.netCtx.fillText(OUTPUT_LABELS[o], outputX + 12, outputY[o] + 3);
      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.65)";
      this.netCtx.font = "9px JetBrains Mono, monospace";
      this.netCtx.fillText(
        `b=${genome[OFFSET_O_BIAS + o].toFixed(2)}`,
        outputX + 12,
        outputY[o] + 14,
      );
      this.netCtx.fillText(
        `a=${value.toFixed(2)}`,
        outputX + 12,
        outputY[o] + 24,
      );
    }

    if (hoveredEdge && this.netMouse) {
      const line1 = hoveredEdge.label;
      const line2 = `w = ${hoveredEdge.weight.toFixed(4)}`;
      this.netCtx.font = "10px JetBrains Mono, monospace";
      const width = Math.max(
        this.netCtx.measureText(line1).width,
        this.netCtx.measureText(line2).width,
      );
      const boxWidth = width + 12;
      const boxHeight = 34;
      const x = Math.min(NET_WIDTH - boxWidth - 4, this.netMouse.x + 10);
      const y = Math.max(4, this.netMouse.y - boxHeight - 8);

      this.netCtx.fillStyle = "rgba(12, 12, 12, 0.95)";
      this.netCtx.fillRect(x, y, boxWidth, boxHeight);
      this.netCtx.strokeStyle = "rgba(255, 255, 255, 0.25)";
      this.netCtx.lineWidth = 1;
      this.netCtx.strokeRect(x, y, boxWidth, boxHeight);

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.92)";
      this.netCtx.fillText(line1, x + 6, y + 13);
      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.75)";
      this.netCtx.fillText(line2, x + 6, y + 26);
    }
  }

  private drawSingleBoard(agent: Agent): void {
    this.drawBoardAt(this.ctx, agent, 0, 0, BOARD_SIZE);
  }

  private drawBoardGridInNet(agents: readonly Agent[]): void {
    this.drawBoardGrid(this.netCtx, this.net.width, this.net.height, agents);
    this.net.style.cursor = "default";
  }

  private drawBoardGrid(
    context: CanvasRenderingContext2D,
    canvasWidth: number,
    canvasHeight: number,
    agents: readonly Agent[],
  ): void {
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvasWidth, canvasHeight);

    const columns = Math.max(1, MULTI_VIEW_COLUMNS);
    const rows = Math.max(1, Math.ceil(agents.length / columns));
    const gap = Math.max(2, Math.floor(Math.min(canvasWidth, canvasHeight) * 0.01));

    const boardSizeByWidth = (canvasWidth - gap * (columns + 1)) / columns;
    const boardSizeByHeight = (canvasHeight - gap * (rows + 1)) / rows;
    const cellSize = Math.max(8, Math.floor(Math.min(boardSizeByWidth, boardSizeByHeight)));

    const gridWidth = cellSize * columns + gap * (columns - 1);
    const gridHeight = cellSize * rows + gap * (rows - 1);
    const startX = Math.floor((canvasWidth - gridWidth) / 2);
    const startY = Math.floor((canvasHeight - gridHeight) / 2);

    for (let i = 0; i < agents.length; i++) {
      const row = Math.floor(i / columns);
      const col = i % columns;
      const x = startX + col * (cellSize + gap);
      const y = startY + row * (cellSize + gap);
      const agent = agents[i];

      context.strokeStyle = agent.alive
        ? "rgba(100, 108, 255, 0.45)"
        : "rgba(244, 67, 54, 0.45)";
      context.lineWidth = 1;
      context.strokeRect(x - 1.5, y - 1.5, cellSize + 3, cellSize + 3);

      this.drawBoardAt(context, agent, x, y, cellSize);

      if (!agent.alive) {
        context.fillStyle = "rgba(0, 0, 0, 0.45)";
        context.fillRect(x, y, cellSize, cellSize);
      }
    }
  }

  private drawBoardAt(
    context: CanvasRenderingContext2D,
    agent: Agent,
    x: number,
    y: number,
    size: number,
  ): void {
    const scale = size / BOARD_SIZE;

    context.save();
    context.translate(x, y);
    context.scale(scale, scale);

    context.fillStyle = "#000";
    context.fillRect(0, 0, BOARD_SIZE, BOARD_SIZE);

    context.strokeStyle = "rgba(255, 255, 255, 0.12)";
    context.lineWidth = 1.1;

    for (let p = 0; p <= BOARD_SIZE; p += TILE_SIZE) {
      context.beginPath();
      context.moveTo(p + 0.5, 0);
      context.lineTo(p + 0.5, BOARD_SIZE);
      context.stroke();

      context.beginPath();
      context.moveTo(0, p + 0.5);
      context.lineTo(BOARD_SIZE, p + 0.5);
      context.stroke();
    }

    context.fillStyle = "#f44336";
    context.fillRect(
      agent.food.x * TILE_SIZE + 3,
      agent.food.y * TILE_SIZE + 3,
      TILE_SIZE - 6,
      TILE_SIZE - 6,
    );

    for (let i = agent.body.length - 1; i >= 0; i--) {
      const part = agent.body[i];
      context.fillStyle = i === 0 ? "#4caf50" : "#43a047";
      context.fillRect(
        part.x * TILE_SIZE + 2,
        part.y * TILE_SIZE + 2,
        TILE_SIZE - 4,
        TILE_SIZE - 4,
      );
    }

    context.restore();
  }

  private drawHistory(history: readonly number[]): void {
    this.chartCtx.fillStyle = "#000";
    this.chartCtx.fillRect(0, 0, CHART_WIDTH, CHART_HEIGHT);

    this.chartCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    this.chartCtx.lineWidth = 1;

    for (let i = 1; i <= 3; i++) {
      const y = (CHART_HEIGHT * i) / 4;
      this.chartCtx.beginPath();
      this.chartCtx.moveTo(0, y + 0.5);
      this.chartCtx.lineTo(CHART_WIDTH, y + 0.5);
      this.chartCtx.stroke();
    }

    if (history.length < 2) {
      return;
    }

    let maxFitness = 1;
    for (const fitness of history) {
      if (fitness > maxFitness) {
        maxFitness = fitness;
      }
    }

    this.chartCtx.strokeStyle = "#646cff";
    this.chartCtx.lineWidth = 2;
    this.chartCtx.beginPath();

    for (let i = 0; i < history.length; i++) {
      const x = 10 + (i / (history.length - 1)) * (CHART_WIDTH - 20);
      const y =
        CHART_HEIGHT - 10 - (history[i] / maxFitness) * (CHART_HEIGHT - 20);

      if (i === 0) {
        this.chartCtx.moveTo(x, y);
      } else {
        this.chartCtx.lineTo(x, y);
      }
    }

    this.chartCtx.stroke();

    this.chartCtx.fillStyle = "rgba(255, 255, 255, 0.75)";
    this.chartCtx.font = "12px IBM Plex Mono, monospace";
    this.chartCtx.fillText(
      `Gen best fitness history (max ${maxFitness.toFixed(2)})`,
      10,
      16,
    );
  }

  private updateStats(state: TrainerState): void {
    this.stats.innerHTML = [
      `Generation: <strong>${state.generation}</strong>`,
      `Alive: ${state.alive}/${state.populationSize}`,
      `Grid: ${GRID_SIZE}x${GRID_SIZE}`,
      `Best score: ${state.bestEverScore}`,
      `Best fitness: ${state.bestEverFitness.toFixed(2)}`,
      `Stale: ${state.staleGenerations}`,
    ].join("<br>");
  }
}
