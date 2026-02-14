import { NORMAL_STEPS_PER_SECOND, TURBO_TIME_BUDGET_MS } from "./config";
import { SnakeRenderer } from "./SnakeRenderer";
import { SnakeTrainer } from "./SnakeTrainer";

export class Game {
  private readonly trainer = new SnakeTrainer();
  private readonly renderer: SnakeRenderer;
  private readonly turboToggle: HTMLInputElement;

  private turboMode = false;
  private stepBudget = 0;
  private lastFrameTime = 0;

  constructor() {
    const netCanvas = this.getCanvas("netCanvas");
    const boardCanvas = this.getCanvas("gameCanvas");
    const chartCanvas = this.getCanvas("chartCanvas");
    const statsElement = this.getElement("stats");
    this.turboToggle = this.getInput("turboToggle");
    const networkToggle = this.getInput("networkToggle");

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

    this.renderer.render(this.trainer.getState());
    requestAnimationFrame(this.loop);
  };
}
