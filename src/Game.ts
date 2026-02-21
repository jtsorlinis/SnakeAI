import {
  GRID_SIZE,
  GRID_SIZE_STEP,
  MAX_GRID_SIZE,
  MIN_GRID_SIZE,
  NORMAL_STEPS_PER_SECOND,
  TURBO_TIME_BUDGET_MS,
  setGridSize,
} from "./config";
import { SnakeRenderer } from "./SnakeRenderer";
import { SnakeTrainer } from "./SnakeTrainer";

const PAUSED_FPS = 30;
const PAUSED_FRAME_MS = 1000 / PAUSED_FPS;

export class Game {
  private trainer = new SnakeTrainer();
  private readonly renderer: SnakeRenderer;
  private readonly toast: HTMLElement;
  private readonly pauseButton: HTMLButtonElement;
  private readonly gridDownButton: HTMLButtonElement;
  private readonly gridUpButton: HTMLButtonElement;
  private readonly resetButton: HTMLButtonElement;
  private readonly gridValue: HTMLElement;

  private paused = true;
  private stepBudget = 0;
  private lastFrameTime = 0;
  private lastPausedRenderTime = 0;
  private envStepsPerSecond = 0;
  private throughputStepAccumulator = 0;
  private throughputTimeAccumulator = 0;
  private toastHideTimer: number | null = null;
  private checkpointSaveInFlight = false;
  private queuedBestReturnForSave = Number.NEGATIVE_INFINITY;
  private persistedBestReturn = Number.NEGATIVE_INFINITY;

  constructor() {
    const netCanvas = this.getCanvas("netCanvas");
    const boardCanvas = this.getCanvas("gameCanvas");
    const chartCanvas = this.getCanvas("chartCanvas");
    const statsElement = this.getElement("stats");
    this.toast = this.getElement("toast");
    const networkToggle = this.getInput("networkToggle");
    this.pauseButton = this.getButton("pauseTraining");
    this.gridDownButton = this.getButton("gridDown");
    this.gridUpButton = this.getButton("gridUp");
    this.resetButton = this.getButton("resetTraining");
    this.gridValue = this.getElement("gridValue");

    this.renderer = new SnakeRenderer({
      netCanvas,
      boardCanvas,
      chartCanvas,
      statsElement,
    });

    this.renderer.setShowNetwork(networkToggle.checked);
    networkToggle.addEventListener("change", () => {
      this.renderer.setShowNetwork(networkToggle.checked);
    });

    this.pauseButton.addEventListener("click", () => {
      this.togglePause();
    });
    this.gridDownButton.addEventListener("click", () => {
      this.updateGridSize(-GRID_SIZE_STEP);
    });
    this.gridUpButton.addEventListener("click", () => {
      this.updateGridSize(GRID_SIZE_STEP);
    });
    this.resetButton.addEventListener("click", () => {
      this.resetTraining();
    });
    this.refreshGridControls();
    this.updatePauseButton();
  }

  public async initialize(): Promise<void> {
    await this.restoreCheckpointOnStartup();
    requestAnimationFrame(this.loop);
  }

  private getCanvas(id: string): HTMLCanvasElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLCanvasElement)) {
      throw new Error(`Missing canvas: ${id}`);
    }
    return element;
  }

  private getElement(id: string): HTMLElement {
    const element = document.getElementById(id);
    if (!element) {
      throw new Error(`Missing element: ${id}`);
    }
    return element;
  }

  private getInput(id: string): HTMLInputElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLInputElement)) {
      throw new Error(`Missing input: ${id}`);
    }
    return element;
  }

  private getButton(id: string): HTMLButtonElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLButtonElement)) {
      throw new Error(`Missing button: ${id}`);
    }
    return element;
  }

  private updateGridSize(delta: number): void {
    const previous = GRID_SIZE;
    const next = setGridSize(previous + delta);
    this.refreshGridControls();
    if (next === previous) {
      return;
    }

    this.trainer.onGridSizeChanged();
    this.envStepsPerSecond = 0;
    this.throughputStepAccumulator = 0;
    this.throughputTimeAccumulator = 0;
    this.trainer.setStepsPerSecond(this.envStepsPerSecond);
    this.stepBudget = 0;
    this.lastFrameTime = 0;
    this.lastPausedRenderTime = 0;
    this.persistedBestReturn = Number.NEGATIVE_INFINITY;
    this.queuedBestReturnForSave = Number.NEGATIVE_INFINITY;
  }

  private refreshGridControls(): void {
    this.gridValue.textContent = `${GRID_SIZE}x${GRID_SIZE}`;
    this.gridDownButton.disabled = GRID_SIZE <= MIN_GRID_SIZE;
    this.gridUpButton.disabled = GRID_SIZE >= MAX_GRID_SIZE;
  }

  private resetTraining(): void {
    this.trainer.reset();
    this.envStepsPerSecond = 0;
    this.throughputStepAccumulator = 0;
    this.throughputTimeAccumulator = 0;
    this.trainer.setStepsPerSecond(this.envStepsPerSecond);
    this.stepBudget = 0;
    this.lastFrameTime = 0;
    this.lastPausedRenderTime = 0;
    this.persistedBestReturn = Number.NEGATIVE_INFINITY;
    this.queuedBestReturnForSave = Number.NEGATIVE_INFINITY;
  }

  private togglePause(): void {
    this.paused = !this.paused;
    this.envStepsPerSecond = 0;
    this.throughputStepAccumulator = 0;
    this.throughputTimeAccumulator = 0;
    this.trainer.setStepsPerSecond(0);
    this.stepBudget = 0;
    this.lastFrameTime = 0;
    this.lastPausedRenderTime = 0;
    this.updatePauseButton();
  }

  private updatePauseButton(): void {
    this.pauseButton.textContent = this.paused
      ? "Resume Training"
      : "Pause Training";
  }

  private showToast(message: string, variant: "info" | "success" | "error"): void {
    this.toast.textContent = message;
    this.toast.className = `toast show ${variant}`;

    if (this.toastHideTimer !== null) {
      window.clearTimeout(this.toastHideTimer);
    }

    this.toastHideTimer = window.setTimeout(() => {
      this.toast.classList.remove("show");
    }, 2600);
  }

  private async restoreCheckpointOnStartup(): Promise<void> {
    try {
      const snapshot = await this.trainer.loadCheckpoint();
      if (!snapshot) {
        return;
      }

      this.persistedBestReturn = snapshot.bestReturn;
      this.queuedBestReturnForSave = snapshot.bestReturn;
      this.showToast(
        `Loaded checkpoint (best return ${snapshot.bestReturn.toFixed(3)}).`,
        "info",
      );
    } catch (error) {
      console.error("Failed to restore checkpoint on startup.", error);
      this.showToast("Failed to load checkpoint from IndexedDB.", "error");
    }
  }

  private maybeQueueCheckpointSave(bestReturn: number, episodeCount: number): void {
    if (episodeCount <= 0) {
      return;
    }

    if (!Number.isFinite(bestReturn)) {
      return;
    }

    if (bestReturn <= this.persistedBestReturn + 1e-6) {
      return;
    }

    if (bestReturn > this.queuedBestReturnForSave) {
      this.queuedBestReturnForSave = bestReturn;
    }

    if (!this.checkpointSaveInFlight) {
      void this.flushCheckpointSaves();
    }
  }

  private async flushCheckpointSaves(): Promise<void> {
    if (this.checkpointSaveInFlight) {
      return;
    }

    this.checkpointSaveInFlight = true;
    try {
      while (this.queuedBestReturnForSave > this.persistedBestReturn + 1e-6) {
        const snapshot = await this.trainer.saveCheckpoint();
        this.persistedBestReturn = snapshot.bestReturn;
        this.showToast(
          `Saved checkpoint (best return ${snapshot.bestReturn.toFixed(3)}).`,
          "success",
        );
      }
    } catch (error) {
      console.error("Failed to save checkpoint to IndexedDB.", error);
      this.showToast("Failed to save checkpoint to IndexedDB.", "error");
    } finally {
      this.checkpointSaveInFlight = false;
    }
  }

  private loop = (time: number): void => {
    if (this.paused) {
      if (
        this.lastPausedRenderTime !== 0 &&
        time - this.lastPausedRenderTime < PAUSED_FRAME_MS
      ) {
        requestAnimationFrame(this.loop);
        return;
      }
      this.lastPausedRenderTime = time;
    } else {
      this.lastPausedRenderTime = 0;
    }

    if (this.lastFrameTime === 0) {
      this.lastFrameTime = time;
    }

    const deltaSeconds = Math.max(0, (time - this.lastFrameTime) / 1000);
    this.lastFrameTime = time;

    let simulatedEnvSteps = 0;
    if (this.paused) {
      this.stepBudget += deltaSeconds * NORMAL_STEPS_PER_SECOND;
      const showcaseSteps = Math.floor(this.stepBudget);
      if (showcaseSteps > 0) {
        this.stepBudget -= showcaseSteps;
        this.trainer.simulateShowcase(showcaseSteps);
      }
    } else {
      const start = performance.now();
      do {
        simulatedEnvSteps += this.trainer.simulate(1);
      } while (performance.now() - start < TURBO_TIME_BUDGET_MS);
    }

    if (!this.paused) {
      this.throughputStepAccumulator += simulatedEnvSteps;
      this.throughputTimeAccumulator += deltaSeconds;

      if (this.throughputTimeAccumulator >= 0.5) {
        this.envStepsPerSecond =
          this.throughputStepAccumulator / this.throughputTimeAccumulator;
        this.throughputStepAccumulator = 0;
        this.throughputTimeAccumulator = 0;
      }
    } else if (this.envStepsPerSecond !== 0) {
      this.envStepsPerSecond = 0;
    }

    this.trainer.setStepsPerSecond(this.envStepsPerSecond);

    const state = this.trainer.getState();
    this.maybeQueueCheckpointSave(state.bestReturn, state.episodeCount);
    this.renderer.render(state);
    requestAnimationFrame(this.loop);
  };
}
