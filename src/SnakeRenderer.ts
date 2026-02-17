import {
  BOARD_SIZE,
  CHART_HEIGHT,
  CHART_WIDTH,
  GRID_SIZE,
  NET_HEIGHT,
  NET_HOVER_RADIUS,
  NET_WIDTH,
  TILE_SIZE,
} from "./config";
import type { NetworkActivationNode, Point, TrainerState } from "./types";

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
    this.syncBoardSize();

    if (this.showNetwork) {
      this.drawNetwork(state);
    }

    this.drawBoard(state);
    this.drawHistory(state.history);
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
    const { activations } = state.network;

    this.netCtx.fillStyle = "#000";
    this.netCtx.fillRect(0, 0, NET_WIDTH, NET_HEIGHT);

    if (!activations) {
      return;
    }

    const top = 24;
    const bottom = NET_HEIGHT - 24;
    const inputX = 84;
    const outputX = NET_WIDTH - 84;

    const inputNodes = activations.nodes
      .filter((node) => node.type === "input")
      .sort((left, right) => {
        const leftIndex = left.ioIndex ?? Number.MAX_SAFE_INTEGER;
        const rightIndex = right.ioIndex ?? Number.MAX_SAFE_INTEGER;
        if (leftIndex !== rightIndex) {
          return leftIndex - rightIndex;
        }
        return left.id - right.id;
      });

    const hiddenNodes = activations.nodes
      .filter((node) => node.type === "hidden")
      .sort((left, right) => {
        if (left.layer !== right.layer) {
          return left.layer - right.layer;
        }
        return left.id - right.id;
      });

    const outputNodes = activations.nodes
      .filter((node) => node.type === "output")
      .sort((left, right) => {
        const leftIndex = left.ioIndex ?? Number.MAX_SAFE_INTEGER;
        const rightIndex = right.ioIndex ?? Number.MAX_SAFE_INTEGER;
        if (leftIndex !== rightIndex) {
          return leftIndex - rightIndex;
        }
        return left.id - right.id;
      });

    const hiddenLayerValues = [
      ...new Set(hiddenNodes.map((node) => node.layer)),
    ].sort((left, right) => left - right);

    const hiddenLayers = hiddenLayerValues.map((layerValue) =>
      hiddenNodes.filter((node) => node.layer === layerValue),
    );

    const nodePositions = new Map<number, Point>();
    const inputY = new Map<number, number>();
    const hiddenLayout: Array<{ x: number; nodes: NetworkActivationNode[] }> =
      [];
    const outputY = new Map<number, number>();

    for (let i = 0; i < inputNodes.length; i++) {
      const y = top + (i * (bottom - top)) / Math.max(1, inputNodes.length - 1);
      nodePositions.set(inputNodes[i].id, { x: inputX, y });
      inputY.set(inputNodes[i].id, y);
    }

    for (let layer = 0; layer < hiddenLayers.length; layer++) {
      const x =
        inputX + ((layer + 1) * (outputX - inputX)) / (hiddenLayers.length + 1);
      const nodes = hiddenLayers[layer];
      hiddenLayout.push({ x, nodes });

      for (let i = 0; i < nodes.length; i++) {
        const y = top + ((i + 1) * (bottom - top)) / (nodes.length + 1);
        nodePositions.set(nodes[i].id, { x, y });
      }
    }

    for (let i = 0; i < outputNodes.length; i++) {
      const y =
        top + (i * (bottom - top)) / Math.max(1, outputNodes.length - 1);
      nodePositions.set(outputNodes[i].id, { x: outputX, y });
      outputY.set(outputNodes[i].id, y);
    }

    const edges: NetEdge[] = [];
    for (const edge of activations.edges) {
      const from = nodePositions.get(edge.from);
      const to = nodePositions.get(edge.to);
      if (!from || !to) {
        continue;
      }

      edges.push({
        x1: from.x,
        y1: from.y,
        x2: to.x,
        y2: to.y,
        weight: edge.weight,
        label: edge.label,
      });
    }

    let outputAbsMax = 0.001;
    for (let i = 0; i < activations.output.length; i++) {
      outputAbsMax = Math.max(outputAbsMax, Math.abs(activations.output[i]));
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
    for (let layer = 0; layer < hiddenLayout.length; layer++) {
      this.netCtx.fillText(`H${layer + 1}`, hiddenLayout[layer].x - 10, 14);
    }
    if (hiddenLayout.length === 0) {
      this.netCtx.fillText("Direct", NET_WIDTH / 2 - 20, 14);
    }
    this.netCtx.fillText("Output", outputX - 24, 14);

    for (const node of inputNodes) {
      const y = inputY.get(node.id);
      if (y === undefined) {
        continue;
      }

      const value = node.value;
      const intensity = Math.min(1, Math.abs(value));

      if (intensity > 0.02) {
        this.netCtx.fillStyle =
          value >= 0
            ? `rgba(41, 182, 246, ${0.1 + intensity * 0.55})`
            : `rgba(255, 183, 77, ${0.1 + intensity * 0.55})`;
        this.netCtx.beginPath();
        this.netCtx.arc(inputX, y, 7 + intensity * 4, 0, Math.PI * 2);
        this.netCtx.fill();
      }

      this.netCtx.fillStyle = value >= 0 ? "#29b6f6" : "#ffb74d";
      this.netCtx.beginPath();
      this.netCtx.arc(inputX, y, 5, 0, Math.PI * 2);
      this.netCtx.fill();

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.88)";
      this.netCtx.font = "10px JetBrains Mono, monospace";
      this.netCtx.fillText(node.label, 6, y + 3);
    }

    for (const layer of hiddenLayout) {
      for (const node of layer.nodes) {
        const point = nodePositions.get(node.id);
        if (!point) {
          continue;
        }

        const value = node.value;
        const intensity = Math.min(1, Math.abs(value));

        if (intensity > 0.02) {
          this.netCtx.fillStyle =
            value >= 0
              ? `rgba(76, 175, 80, ${0.1 + intensity * 0.6})`
              : `rgba(244, 67, 54, ${0.1 + intensity * 0.6})`;
          this.netCtx.beginPath();
          this.netCtx.arc(point.x, point.y, 7 + intensity * 4, 0, Math.PI * 2);
          this.netCtx.fill();
        }

        this.netCtx.fillStyle = value >= 0 ? "#26a69a" : "#ef5350";
        this.netCtx.beginPath();
        this.netCtx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        this.netCtx.fill();
      }
    }

    for (const node of outputNodes) {
      const y = outputY.get(node.id);
      if (y === undefined) {
        continue;
      }

      const outputIndex = node.ioIndex ?? -1;
      const value =
        outputIndex >= 0 && outputIndex < activations.output.length
          ? activations.output[outputIndex]
          : node.value;
      const intensity = Math.min(1, Math.abs(value) / outputAbsMax);

      if (intensity > 0.02) {
        this.netCtx.fillStyle =
          value >= 0
            ? `rgba(76, 175, 80, ${0.12 + intensity * 0.6})`
            : `rgba(244, 67, 54, ${0.12 + intensity * 0.6})`;
        this.netCtx.beginPath();
        this.netCtx.arc(outputX, y, 8 + intensity * 5, 0, Math.PI * 2);
        this.netCtx.fill();
      }

      const isBest = outputIndex === activations.best;
      this.netCtx.fillStyle = isBest ? "#ffd54f" : "#ffb74d";
      this.netCtx.beginPath();
      this.netCtx.arc(outputX, y, 6, 0, Math.PI * 2);
      this.netCtx.fill();

      if (isBest) {
        this.netCtx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        this.netCtx.lineWidth = 1.5;
        this.netCtx.beginPath();
        this.netCtx.arc(outputX, y, 9, 0, Math.PI * 2);
        this.netCtx.stroke();
      }

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
      this.netCtx.font = "11px JetBrains Mono, monospace";
      this.netCtx.fillText(node.label, outputX + 12, y + 3);
      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.65)";
      this.netCtx.font = "9px JetBrains Mono, monospace";
      this.netCtx.fillText(`b=${node.bias.toFixed(2)}`, outputX + 12, y + 14);
      this.netCtx.fillText(`a=${value.toFixed(2)}`, outputX + 12, y + 24);
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
      const y =
        CHART_HEIGHT - 10 - (history[i] / maxScore) * (CHART_HEIGHT - 20);

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
      `Grid: ${GRID_SIZE}x${GRID_SIZE}`,
      `Best score: ${state.bestEverScore}`,
      `Best fitness: ${state.bestEverFitness.toFixed(1)}`,
      `Stale: ${state.staleGenerations}`,
    ].join("<br>");
  }
}
