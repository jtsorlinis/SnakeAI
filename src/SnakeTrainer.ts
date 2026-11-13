import { CMAESTrainer } from "./CMAESTrainer";
import { GATrainer } from "./GATrainer";
import { PPOTrainer } from "./PPOTrainer";
import { VisionPPOTrainer } from "./VisionPPOTrainer";
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
        : algorithm === "cmaes"
          ? new CMAESTrainer()
          : algorithm === "ppo"
            ? new PPOTrainer()
            : new VisionPPOTrainer();

    if (trainer.supportsPlaybackMode()) {
      trainer.setPlaybackMode(this.playbackMode);
    }

    return trainer;
  }
}
