import { A2CTrainer } from "./A2CTrainer";
import { CMAESTrainer } from "./CMAESTrainer";
import { ESTrainer } from "./ESTrainer";
import { GATrainer } from "./GATrainer";
import { OpenAIESTrainer } from "./OpenAIESTrainer";
import { PPOTrainer } from "./PPOTrainer";
import { REINFORCETrainer } from "./REINFORCETrainer";
import type {
  PolicyPlaybackMode,
  TrainerAlgorithm,
  TrainerController,
  TrainerState,
} from "./types";

export class SnakeTrainer {
  private algorithm: TrainerAlgorithm = "ga";
  private playbackMode: PolicyPlaybackMode = "greedy";
  private trainer: TrainerController = this.createTrainer(this.algorithm);

  public reset(): void {
    this.trainer.reset();
  }

  public setAlgorithm(algorithm: TrainerAlgorithm): void {
    if (algorithm === this.algorithm) {
      return;
    }

    this.algorithm = algorithm;
    this.trainer = this.createTrainer(this.algorithm);
  }

  public simulate(stepCount: number): void {
    this.trainer.simulate(stepCount);
  }

  public getState(randomBoardCount = 0): TrainerState {
    return this.trainer.getState(randomBoardCount);
  }

  public onGridSizeChanged(): void {
    this.trainer.onGridSizeChanged();
  }

  public setPlaybackMode(mode: PolicyPlaybackMode): void {
    this.playbackMode = mode;
    this.trainer.setPlaybackMode(mode);
  }

  public getPlaybackMode(): PolicyPlaybackMode {
    return this.trainer.getPlaybackMode();
  }

  public supportsPlaybackMode(): boolean {
    return this.trainer.supportsPlaybackMode();
  }

  private createTrainer(algorithm: TrainerAlgorithm): TrainerController {
    const trainer =
      algorithm === "ga"
        ? new GATrainer()
        : algorithm === "es"
          ? new ESTrainer()
          : algorithm === "cmaes"
            ? new CMAESTrainer()
            : algorithm === "openai-es"
              ? new OpenAIESTrainer()
            : algorithm === "a2c"
              ? new A2CTrainer()
            : algorithm === "reinforce"
              ? new REINFORCETrainer()
              : new PPOTrainer();

    if (trainer.supportsPlaybackMode()) {
      trainer.setPlaybackMode(this.playbackMode);
    }

    return trainer;
  }
}
