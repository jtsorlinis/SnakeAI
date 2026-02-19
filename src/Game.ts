import {
  GRID_SIZE,
  GRID_SIZE_STEP,
  MAX_GRID_SIZE,
  MIN_GRID_SIZE,
  NORMAL_STEPS_PER_SECOND,
  TRAIN_ENVS,
  TURBO_TIME_BUDGET_MS,
  setGridSize,
} from "./config";
import { SnakeRenderer } from "./SnakeRenderer";
import { SnakeTrainer } from "./SnakeTrainer";

export class Game {
  private trainer = new SnakeTrainer();
  private readonly renderer: SnakeRenderer;
  private readonly turboToggle: HTMLInputElement;
  private readonly gridDownButton: HTMLButtonElement;
  private readonly gridUpButton: HTMLButtonElement;
  private readonly resetButton: HTMLButtonElement;
  private readonly gridValue: HTMLElement;

  private turboMode = false;
  private stepBudget = 0;
  private lastFrameTime = 0;
  private envStepsPerSecond = 0;
  private throughputStepAccumulator = 0;
  private throughputTimeAccumulator = 0;

  constructor() {
    const netCanvas = this.getCanvas("netCanvas");
    const boardCanvas = this.getCanvas("gameCanvas");
    const chartCanvas = this.getCanvas("chartCanvas");
    const statsElement = this.getElement("stats");
    this.turboToggle = this.getInput("turboToggle");
    const networkToggle = this.getInput("networkToggle");
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

    this.turboMode = this.turboToggle.checked;
    this.turboToggle.addEventListener("change", () => {
      this.turboMode = this.turboToggle.checked;
      this.stepBudget = 0;
    });

    this.renderer.setShowNetwork(networkToggle.checked);
    networkToggle.addEventListener("change", () => {
      this.renderer.setShowNetwork(networkToggle.checked);
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
  }

  private loop = (time: number): void => {
    if (this.lastFrameTime === 0) {
      this.lastFrameTime = time;
    }

    const deltaSeconds = Math.max(0, (time - this.lastFrameTime) / 1000);
    this.lastFrameTime = time;

    let simulatedTicks = 0;

    if (this.turboMode) {
      const start = performance.now();
      do {
        this.trainer.simulate(1);
        simulatedTicks += 1;
      } while (performance.now() - start < TURBO_TIME_BUDGET_MS);
    } else {
      this.stepBudget += deltaSeconds * NORMAL_STEPS_PER_SECOND;
      const stepCount = Math.floor(this.stepBudget);
      if (stepCount > 0) {
        this.stepBudget -= stepCount;
        this.trainer.simulate(stepCount);
        simulatedTicks = stepCount;
      }
    }

    if (simulatedTicks > 0) {
      this.throughputStepAccumulator += simulatedTicks * TRAIN_ENVS;
      this.throughputTimeAccumulator += deltaSeconds;

      if (this.throughputTimeAccumulator >= 0.5) {
        this.envStepsPerSecond =
          this.throughputStepAccumulator / this.throughputTimeAccumulator;
        this.throughputStepAccumulator = 0;
        this.throughputTimeAccumulator = 0;
      }
    }

    this.trainer.setStepsPerSecond(this.envStepsPerSecond);

    this.renderer.render(this.trainer.getState());
    requestAnimationFrame(this.loop);
  };
}
