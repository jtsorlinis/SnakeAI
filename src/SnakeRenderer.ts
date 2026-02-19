import {
  BOARD_SIZE,
  CHART_HEIGHT,
  CHART_WIDTH,
  GRID_SIZE,
  NET_HEIGHT,
  NET_WIDTH,
  OBS_CHANNELS,
  OBS_LABELS,
  OUTPUT_LABELS,
  TILE_SIZE,
} from "./config";
import type { TrainerState } from "./types";

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
  }

  public setShowNetwork(show: boolean): void {
    this.showNetwork = show;
    this.net.style.display = show ? "block" : "none";
  }

  public render(state: TrainerState): void {
    this.syncBoardSize();

    if (this.showNetwork) {
      this.drawNetwork(state);
    }

    this.drawBoard(state);
    this.drawHistory(state.rewardHistory);
    this.updateStats(state);
  }

  private syncBoardSize(): void {
    if (this.board.width !== BOARD_SIZE || this.board.height !== BOARD_SIZE) {
      this.board.width = BOARD_SIZE;
      this.board.height = BOARD_SIZE;
    }
  }

  private drawNetwork(state: TrainerState): void {
    this.netCtx.fillStyle = "#000";
    this.netCtx.fillRect(0, 0, NET_WIDTH, NET_HEIGHT);

    const obs = state.network.observation;
    const policy = state.network.policy;

    if (!obs || !policy) {
      this.netCtx.fillStyle = "rgba(255,255,255,0.8)";
      this.netCtx.font = "14px JetBrains Mono, monospace";
      this.netCtx.fillText("Waiting for observation", 16, 26);
      return;
    }

    const area = GRID_SIZE * GRID_SIZE;
    const marginX = 14;
    const top = 36;
    const gap = 10;
    const panelWidth = Math.floor(
      (NET_WIDTH - marginX * 2 - gap * (OBS_CHANNELS - 1)) / OBS_CHANNELS,
    );
    const cellSize = Math.max(2, Math.floor((panelWidth - 12) / GRID_SIZE));
    const panelGridSize = cellSize * GRID_SIZE;

    this.netCtx.fillStyle = "rgba(255, 255, 255, 0.88)";
    this.netCtx.font = "13px JetBrains Mono, monospace";
    this.netCtx.fillText("Observation channels", marginX, 18);

    for (let channel = 0; channel < OBS_CHANNELS; channel++) {
      const panelX = marginX + channel * (panelWidth + gap);
      const gridX = panelX + Math.floor((panelWidth - panelGridSize) / 2);
      const gridY = top;

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
      this.netCtx.fillRect(panelX, top - 18, panelWidth, panelGridSize + 20);

      this.netCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
      this.netCtx.font = "11px JetBrains Mono, monospace";
      this.netCtx.fillText(OBS_LABELS[channel], panelX + 6, top - 4);

      const base = channel * area;
      for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          const value = obs[base + y * GRID_SIZE + x];
          const px = gridX + x * cellSize;
          const py = gridY + y * cellSize;

          this.netCtx.fillStyle = this.observationCellColor(channel, value);
          this.netCtx.fillRect(px, py, cellSize - 1, cellSize - 1);
        }
      }
    }

    const qTop = top + panelGridSize + 44;
    const barLeft = 142;
    const barMaxWidth = NET_WIDTH - barLeft - 24;

    this.netCtx.fillStyle = "rgba(255, 255, 255, 0.88)";
    this.netCtx.font = "13px JetBrains Mono, monospace";
    this.netCtx.fillText("Policy probabilities", marginX, qTop - 16);

    for (let i = 0; i < policy.length; i++) {
      const y = qTop + i * 34;
      const value = policy[i];
      const width = Math.max(0, Math.min(1, value)) * barMaxWidth;

      this.netCtx.fillStyle = "rgba(255,255,255,0.85)";
      this.netCtx.font = "11px JetBrains Mono, monospace";
      this.netCtx.fillText(OUTPUT_LABELS[i], marginX, y + 10);

      this.netCtx.fillStyle = "rgba(255,255,255,0.08)";
      this.netCtx.fillRect(barLeft, y, barMaxWidth, 14);

      this.netCtx.fillStyle = "rgba(102, 187, 106, 0.9)";
      this.netCtx.fillRect(barLeft, y, width, 14);

      if (i === state.network.action) {
        this.netCtx.strokeStyle = "rgba(255, 213, 79, 0.95)";
        this.netCtx.lineWidth = 2;
        this.netCtx.strokeRect(barLeft - 1, y - 1, barMaxWidth + 2, 16);
      }

      this.netCtx.fillStyle = "rgba(255,255,255,0.85)";
      this.netCtx.fillText(
        `${(value * 100).toFixed(1)}%`,
        barLeft + barMaxWidth - 54,
        y + 11,
      );
    }

    const valueY = qTop + policy.length * 34 + 6;
    this.netCtx.fillStyle = "rgba(255,255,255,0.86)";
    this.netCtx.font = "12px JetBrains Mono, monospace";
    this.netCtx.fillText(
      `Value estimate: ${state.network.value.toFixed(3)}`,
      marginX,
      valueY,
    );

    const footerY = valueY + 18;
    this.netCtx.fillStyle = "rgba(255,255,255,0.78)";
    this.netCtx.font = "11px JetBrains Mono, monospace";
    this.netCtx.fillText(
      `loss=${state.totalLoss.toFixed(4)}   policy=${state.policyLoss.toFixed(4)}   value=${state.valueLoss.toFixed(4)}`,
      marginX,
      footerY,
    );

    this.netCtx.fillText(
      `entropy=${state.entropy.toFixed(4)}   kl=${state.approxKl.toFixed(4)}   clip=${state.clipFraction.toFixed(3)}`,
      marginX,
      footerY + 14,
    );
  }

  private observationCellColor(channel: number, value: number): string {
    if (Math.abs(value) < 1e-6) {
      return "rgba(255,255,255,0.04)";
    }

    const alpha = 0.15 + 0.75 * Math.min(1, Math.abs(value));

    if (channel === 0) {
      return `rgba(239, 83, 80, ${alpha})`;
    }

    if (channel === 1) {
      return `rgba(102, 187, 106, ${alpha})`;
    }

    if (channel === 2) {
      return `rgba(38, 166, 154, ${alpha})`;
    }

    return value > 0
      ? `rgba(102, 187, 106, ${alpha})`
      : `rgba(66, 165, 245, ${alpha})`;
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

    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;

    for (const value of history) {
      if (value < minValue) {
        minValue = value;
      }
      if (value > maxValue) {
        maxValue = value;
      }
    }

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
      return;
    }

    if (Math.abs(maxValue - minValue) < 1e-6) {
      minValue -= 1;
      maxValue += 1;
    }

    this.chartCtx.strokeStyle = "#64b5f6";
    this.chartCtx.lineWidth = 2;
    this.chartCtx.beginPath();

    for (let i = 0; i < history.length; i++) {
      const x = 10 + (i / (history.length - 1)) * (CHART_WIDTH - 20);
      const normalized = (history[i] - minValue) / (maxValue - minValue);
      const y = CHART_HEIGHT - 10 - normalized * (CHART_HEIGHT - 20);

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
      `Episode return (min ${minValue.toFixed(2)}, max ${maxValue.toFixed(2)})`,
      10,
      16,
    );
  }

  private updateStats(state: TrainerState): void {
    this.stats.innerHTML = [
      `Episodes: <strong>${state.episodeCount}</strong>`,
      `Steps: ${state.totalSteps.toLocaleString()}`,
      `Env steps/s: ${state.stepsPerSecond.toFixed(0)}`,
      `Grid: ${GRID_SIZE}x${GRID_SIZE}`,
      `PPO updates: ${state.updates.toLocaleString()}`,
      `Rollout progress: ${(state.rolloutProgress * 100).toFixed(0)}%`,
      `Avg return (last 100): ${state.avgReturn.toFixed(3)}`,
      `Best return: ${state.bestReturn.toFixed(3)}`,
      `Loss (EMA): ${state.totalLoss.toFixed(4)}`,
      `Policy loss: ${state.policyLoss.toFixed(4)}`,
      `Value loss: ${state.valueLoss.toFixed(4)}`,
      `Entropy: ${state.entropy.toFixed(4)}`,
      `Approx KL: ${state.approxKl.toFixed(4)}`,
      `Clip fraction: ${state.clipFraction.toFixed(3)}`,
    ].join("<br>");
  }
}
