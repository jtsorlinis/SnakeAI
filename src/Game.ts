import {
  GRID_SIZE,
  MULTI_VIEW_COUNT,
  NORMAL_STEPS_PER_SECOND,
  TURBO_TIME_BUDGET_MS,
  setGridSize,
} from "./config";
import { SnakeRenderer } from "./SnakeRenderer";
import { SnakeTrainer } from "./SnakeTrainer";
import type { TrainerAlgorithm } from "./types";

export class Game {
  private trainer = new SnakeTrainer();
  private readonly renderer: SnakeRenderer;
  private readonly turboToggle: HTMLInputElement;
  private readonly leftViewToggle: HTMLInputElement;
  private readonly algorithmPicker: HTMLElement;
  private readonly boardSizePicker: HTMLElement;
  private readonly resetButton: HTMLButtonElement;

  private turboMode = false;
  private showMiniBoardsInLeft = false;
  private stepBudget = 0;
  private lastFrameTime = 0;

  constructor() {
    const netCanvas = this.getCanvas("netCanvas");
    const boardCanvas = this.getCanvas("gameCanvas");
    const chartCanvas = this.getCanvas("chartCanvas");
    const statsRoot = this.getElement("app");
    this.turboToggle = this.getInput("turboToggle");
    this.leftViewToggle = this.getInput("leftViewToggle");
    this.algorithmPicker = this.getElement("algorithmPicker");
    this.boardSizePicker = this.getElement("boardSizePicker");
    this.resetButton = this.getButton("resetTraining");

    this.renderer = new SnakeRenderer({
      netCanvas,
      boardCanvas,
      chartCanvas,
      statsRoot,
    });

    this.turboMode = this.turboToggle.checked;
    this.turboToggle.addEventListener("change", () => {
      this.turboMode = this.turboToggle.checked;
      this.stepBudget = 0;
    });

    this.showMiniBoardsInLeft = this.leftViewToggle.checked;
    this.renderer.setShowMiniBoardsInLeft(this.showMiniBoardsInLeft);
    this.leftViewToggle.addEventListener("change", () => {
      this.showMiniBoardsInLeft = this.leftViewToggle.checked;
      this.renderer.setShowMiniBoardsInLeft(this.showMiniBoardsInLeft);
    });

    this.trainer.setAlgorithm(this.getSelectedAlgorithm());
    this.algorithmPicker.addEventListener("change", () => {
      this.trainer.setAlgorithm(this.getSelectedAlgorithm());
      this.stepBudget = 0;
      this.lastFrameTime = 0;
    });

    // A reloaded page can come back with either size selected.
    this.applySelectedBoardSize();
    this.boardSizePicker.addEventListener("change", () => {
      this.applySelectedBoardSize();
    });
    this.resetButton.addEventListener("click", () => {
      this.resetTraining();
    });

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

  private getSelectedAlgorithm(): TrainerAlgorithm {
    const checked =
      this.algorithmPicker.querySelector<HTMLInputElement>("input:checked");
    return (checked?.value ?? "ga") as TrainerAlgorithm;
  }

  private getButton(id: string): HTMLButtonElement {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLButtonElement)) {
      throw new Error(`Missing button: ${id}`);
    }
    return element;
  }

  private applySelectedBoardSize(): void {
    const checked =
      this.boardSizePicker.querySelector<HTMLInputElement>("input:checked");
    const previous = GRID_SIZE;
    const next = setGridSize(Number(checked?.value ?? previous));
    if (next === previous) {
      return;
    }

    this.trainer.onGridSizeChanged();
    this.stepBudget = 0;
    this.lastFrameTime = 0;
  }

  private resetTraining(): void {
    this.trainer.reset();
    this.stepBudget = 0;
    this.lastFrameTime = 0;
  }

  private loop = (time: number): void => {
    if (this.lastFrameTime === 0) {
      this.lastFrameTime = time;
    }

    const deltaSeconds = Math.max(0, (time - this.lastFrameTime) / 1000);
    this.lastFrameTime = time;

    if (this.turboMode) {
      const start = performance.now();
      do {
        this.trainer.simulate(1);
      } while (performance.now() - start < TURBO_TIME_BUDGET_MS);
    } else {
      this.stepBudget += deltaSeconds * NORMAL_STEPS_PER_SECOND;
      const stepCount = Math.floor(this.stepBudget);
      if (stepCount > 0) {
        this.stepBudget -= stepCount;
        this.trainer.simulate(stepCount);
      }
    }

    const boardSampleCount = this.showMiniBoardsInLeft ? MULTI_VIEW_COUNT : 0;
    this.renderer.render(this.trainer.getState(boardSampleCount));
    requestAnimationFrame(this.loop);
  };
}
