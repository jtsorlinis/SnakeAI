import {
  BOARD_SIZE,
  CHART_HEIGHT,
  CHART_WIDTH,
  HIDDEN,
  INPUTS,
  INPUT_LABELS,
  NET_HEIGHT,
  NET_HOVER_RADIUS,
  NET_WIDTH,
  OFFSET_HO,
  OFFSET_IH,
  OFFSET_O_BIAS,
  OUTPUTS,
  OUTPUT_LABELS,
  TILE_SIZE,
} from "./config";
import type { NetEdge, Point, TrainerState } from "./types";

type RendererElements = {
  netCanvas: HTMLCanvasElement;
  boardCanvas: HTMLCanvasElement;
  chartCanvas: HTMLCanvasElement;
  statsElement: HTMLElement;
};

export class SnakeRenderer {
  private readonly net: HTMLCanvasElement;
  private readonly netCtx: CanvasRenderingContext2D;
  private readonly board: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly chart: HTMLCanvasElement;
  private readonly chartCtx: CanvasRenderingContext2D;
  private readonly stats: HTMLElement;

  private showNetwork = true;
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

  public setShowNetwork(show: boolean): void {
    this.showNetwork = show;
    this.net.style.display = show ? "block" : "none";

    if (!show) {
      this.netMouse = null;
      this.net.style.cursor = "default";
    }
  }

  public render(state: TrainerState): void {
    if (this.showNetwork) {
      this.drawNetwork(state);
    }

    this.drawBoard(state);
    this.drawHistory(state.history);
    this.updateStats(state);
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

    const top = 24;
    const bottom = NET_HEIGHT - 24;
    const inputX = 120;
    const hiddenX = NET_WIDTH / 2;
    const outputX = NET_WIDTH - 120;

    const inputY: number[] = [];
    const hiddenY: number[] = [];
    const outputY: number[] = [];

    for (let i = 0; i < INPUTS; i++) {
      inputY.push(top + (i * (bottom - top)) / Math.max(1, INPUTS - 1));
    }

    for (let h = 0; h < HIDDEN; h++) {
      hiddenY.push(top + (h * (bottom - top)) / Math.max(1, HIDDEN - 1));
    }

    for (let o = 0; o < OUTPUTS; o++) {
      outputY.push(top + (o * (bottom - top)) / Math.max(1, OUTPUTS - 1));
    }

    const edges: NetEdge[] = [];

    for (let h = 0; h < HIDDEN; h++) {
      const wOffset = OFFSET_IH + h * INPUTS;
      for (let i = 0; i < INPUTS; i++) {
        edges.push({
          x1: inputX,
          y1: inputY[i],
          x2: hiddenX,
          y2: hiddenY[h],
          weight: genome[wOffset + i],
          label: `${INPUT_LABELS[i]} -> H${h + 1}`,
        });
      }
    }

    for (let o = 0; o < OUTPUTS; o++) {
      const wOffset = OFFSET_HO + o * HIDDEN;
      for (let h = 0; h < HIDDEN; h++) {
        edges.push({
          x1: hiddenX,
          y1: hiddenY[h],
          x2: outputX,
          y2: outputY[o],
          weight: genome[wOffset + h],
          label: `H${h + 1} -> ${OUTPUT_LABELS[o]}`,
        });
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
    this.netCtx.fillText("Hidden", hiddenX - 22, 14);
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

    for (let h = 0; h < HIDDEN; h++) {
      const value = activations.hidden[h];
      const intensity = Math.min(1, Math.abs(value));

      if (intensity > 0.02) {
        this.netCtx.fillStyle =
          value >= 0
            ? `rgba(76, 175, 80, ${0.1 + intensity * 0.6})`
            : `rgba(244, 67, 54, ${0.1 + intensity * 0.6})`;
        this.netCtx.beginPath();
        this.netCtx.arc(hiddenX, hiddenY[h], 7 + intensity * 4, 0, Math.PI * 2);
        this.netCtx.fill();
      }

      this.netCtx.fillStyle = value >= 0 ? "#26a69a" : "#ef5350";
      this.netCtx.beginPath();
      this.netCtx.arc(hiddenX, hiddenY[h], 5, 0, Math.PI * 2);
      this.netCtx.fill();
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
      this.netCtx.fillText(`b=${genome[OFFSET_O_BIAS + o].toFixed(2)}`, outputX + 12, outputY[o] + 14);
      this.netCtx.fillText(`a=${value.toFixed(2)}`, outputX + 12, outputY[o] + 24);
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

  private drawBoard(state: TrainerState): void {
    const agent = state.boardAgent;

    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, BOARD_SIZE, BOARD_SIZE);

    this.ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    this.ctx.lineWidth = 1.1;

    for (let p = 0; p <= BOARD_SIZE; p += TILE_SIZE) {
      this.ctx.beginPath();
      this.ctx.moveTo(p + 0.5, 0);
      this.ctx.lineTo(p + 0.5, BOARD_SIZE);
      this.ctx.stroke();

      this.ctx.beginPath();
      this.ctx.moveTo(0, p + 0.5);
      this.ctx.lineTo(BOARD_SIZE, p + 0.5);
      this.ctx.stroke();
    }

    this.ctx.fillStyle = "#f44336";
    this.ctx.fillRect(
      agent.food.x * TILE_SIZE + 3,
      agent.food.y * TILE_SIZE + 3,
      TILE_SIZE - 6,
      TILE_SIZE - 6,
    );

    for (let i = agent.body.length - 1; i >= 0; i--) {
      const part = agent.body[i];
      this.ctx.fillStyle = i === 0 ? "#4caf50" : "#43a047";
      this.ctx.fillRect(
        part.x * TILE_SIZE + 2,
        part.y * TILE_SIZE + 2,
        TILE_SIZE - 4,
        TILE_SIZE - 4,
      );
    }
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

    let maxScore = 1;
    for (const score of history) {
      if (score > maxScore) {
        maxScore = score;
      }
    }

    this.chartCtx.strokeStyle = "#646cff";
    this.chartCtx.lineWidth = 2;
    this.chartCtx.beginPath();

    for (let i = 0; i < history.length; i++) {
      const x = 10 + (i / (history.length - 1)) * (CHART_WIDTH - 20);
      const y = CHART_HEIGHT - 10 - (history[i] / maxScore) * (CHART_HEIGHT - 20);

      if (i === 0) {
        this.chartCtx.moveTo(x, y);
      } else {
        this.chartCtx.lineTo(x, y);
      }
    }

    this.chartCtx.stroke();

    this.chartCtx.fillStyle = "rgba(255, 255, 255, 0.75)";
    this.chartCtx.font = "12px IBM Plex Mono, monospace";
    this.chartCtx.fillText(`Gen best score history (max ${maxScore})`, 10, 16);
  }

  private updateStats(state: TrainerState): void {
    this.stats.innerHTML = [
      `Generation: <strong>${state.generation}</strong>`,
      `Alive: ${state.alive}/${state.populationSize}`,
      `Best score: ${state.bestEverScore}`,
      `Best fitness: ${state.bestEverFitness.toFixed(1)}`,
      `Stale: ${state.staleGenerations}`,
    ].join("<br>");
  }
}
