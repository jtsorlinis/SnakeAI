import {
  BASE_HUNGER,
  GRID_SIZE,
  OUTPUTS,
  PPO_ROLLOUT_HORIZON,
  REPLAY_AGENT_COUNT,
  ROLLOUT_BATCH_SIZE,
  VALIDATION_GAMES,
  VISION_CLEAR_REWARD,
  VISION_TRAIN_SAMPLES_PER_STEP,
} from "./config";
import { BoardVision, visionViewSize } from "./BoardVision";
import { NeuralNetwork } from "./NeuralNetwork";
import { scoreFitness } from "./PolicyValidation";
import { SnakeEnvironment } from "./SnakeEnvironment";
import { VisionPPOOptimizer } from "./VisionPPOOptimizer";
import type {
  Agent,
  PolicyPlaybackMode,
  TrainerController,
  TrainerState,
  VisionView,
} from "./types";

// Board PPO's snakes are driven by the trainer, not by a sensor policy of
// their own.
const NO_POLICY = new Float32Array(0);

// PPO with a policy that reads the whole board (see BoardVision) instead of
// the ten sensors. Each update is a rollout followed by training spread over
// many simulation steps, because the networks are too big to train in one go
// without freezing the page.
export class VisionPPOTrainer implements TrainerController {
  private readonly environment = new SnakeEnvironment(new NeuralNetwork());
  // Sized for the current board; built in reset().
  private vision!: BoardVision;
  private ppo!: VisionPPOOptimizer;
  private view!: VisionView;

  private rolloutAgents: Agent[] = [];
  private rolloutStep = 0;
  private training = false;
  private nextValues = new Float32Array(ROLLOUT_BATCH_SIZE);

  private ppoUpdate = 1;
  private bestEverScore = 0;
  private bestEverFitness = 0;
  private bestFitnessUpdate = 1;
  private fitnessHistory: number[] = [];
  private episodeFitnessTotal = 0;
  private episodeCount = 0;

  // The policy on show, the snake playing it on the main board, and extra
  // snakes replaying it for the replay average.
  private displayPolicy = new Float32Array(0);
  private displayAgent: Agent | null = null;
  private replayAgents: Agent[] = [];
  private replayGames = 0;
  private replayTotalScore = 0;
  private replayWins = 0;

  // A copy of the averaged policy playing fresh games in the background. It
  // takes over the display if it does at least as well as the record.
  private candidatePolicy = new Float32Array(0);
  private candidateUpdate = 0;
  private candidateAgents: Agent[] = [];
  private candidateFitness = 0;
  private candidateClears = 0;
  // Set once a candidate clears the board in at least half of its games;
  // training uses a smaller learning rate from then on.
  private settled = false;

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.vision = new BoardVision();
    this.ppo = new VisionPPOOptimizer(
      ROLLOUT_BATCH_SIZE,
      this.vision.indices.length,
    );
    this.view = this.createView();

    this.rolloutAgents = [];
    for (let i = 0; i < ROLLOUT_BATCH_SIZE; i++) {
      this.rolloutAgents.push(this.environment.createAgent(NO_POLICY));
    }
    this.rolloutStep = 0;
    this.training = false;

    this.ppoUpdate = 1;
    this.bestEverScore = 0;
    this.bestEverFitness = 0;
    this.bestFitnessUpdate = 1;
    this.fitnessHistory = [];
    this.episodeFitnessTotal = 0;
    this.episodeCount = 0;

    this.candidateAgents = [];
    this.candidateUpdate = 0;
    this.settled = false;
    this.setDisplayPolicy(this.ppo.averagedPolicy);
  }

  public simulate(stepCount: number): void {
    for (let i = 0; i < stepCount; i++) {
      if (!this.training) {
        this.stepRollout();
      } else if (this.ppo.trainSlice(VISION_TRAIN_SAMPLES_PER_STEP)) {
        this.finishUpdate();
      }

      this.stepDisplay();
      this.stepCandidate();
    }
  }

  public getState(randomBoardCount = 0): TrainerState {
    const boardAgent = this.displayAgent ?? this.rolloutAgents[0];

    return {
      algorithm: "boardppo",
      boardAgent,
      // Dead snakes are replaced on the spot, so any slice is a live sample.
      boardAgents: this.rolloutAgents.slice(0, Math.max(0, randomBoardCount)),
      fitnessHistory: this.fitnessHistory,
      iteration: this.ppoUpdate,
      iterationLabel: "PPO update",
      alive: this.rolloutAgents.length,
      batchSize: ROLLOUT_BATCH_SIZE,
      batchSizeLabel: "Rollout batch",
      bestEverScore: this.bestEverScore,
      bestEverFitness: this.bestEverFitness,
      replay: {
        games: this.replayGames,
        averageScore:
          this.replayGames > 0 ? this.replayTotalScore / this.replayGames : 0,
        wins: this.replayWins,
      },
      staleIterations: Math.max(0, this.ppoUpdate - this.bestFitnessUpdate),
      staleLabel: "Updates since best",
      historyLabel: "Board PPO average score history",
      policySourceLabel: "Best policy",
      playbackMode: "greedy",
      playbackModeEnabled: false,
      network: { policy: null, activations: null },
      vision: this.updateView(boardAgent),
    };
  }

  // The networks are sized for one board, so a new size starts again.
  public onGridSizeChanged(): void {
    this.reset();
  }

  public setPlaybackMode(_mode: PolicyPlaybackMode): void {}

  public getPlaybackMode(): PolicyPlaybackMode {
    return "greedy";
  }

  public supportsPlaybackMode(): boolean {
    return false;
  }

  private createView(): VisionView {
    const size = visionViewSize();
    return {
      size,
      inside: new Uint8Array(size * size),
      body: new Float32Array(size * size),
      food: 0,
      probabilities: new Float32Array(OUTPUTS),
      best: 0,
    };
  }

  private stepRollout(): void {
    const step = this.rolloutStep;
    for (let index = 0; index < this.rolloutAgents.length; index++) {
      const agent = this.rolloutAgents[index];
      this.vision.observe(agent);
      const action = this.ppo.act(
        this.vision,
        agent.hunger / BASE_HUNGER,
        index,
        step,
      );
      const previousScore = agent.score;
      this.environment.step(agent, action);
      this.ppo.recordOutcome(
        index,
        step,
        this.computeReward(agent, previousScore),
        !agent.alive,
      );

      if (!agent.alive) {
        this.bestEverScore = Math.max(this.bestEverScore, agent.score);
        this.episodeFitnessTotal += scoreFitness(agent);
        this.episodeCount += 1;
        this.rolloutAgents[index] = this.environment.createAgent(NO_POLICY);
      }
    }

    this.rolloutStep += 1;
    if (this.rolloutStep < PPO_ROLLOUT_HORIZON) {
      return;
    }

    for (let index = 0; index < this.rolloutAgents.length; index++) {
      const agent = this.rolloutAgents[index];
      this.vision.observe(agent);
      this.nextValues[index] = this.ppo.estimateValue(
        this.vision,
        agent.hunger / BASE_HUNGER,
      );
    }
    this.ppo.finishRollout(this.nextValues);
    this.training = true;
  }

  private finishUpdate(): void {
    this.training = false;
    this.rolloutStep = 0;

    // Average over the games that ended during this update's rollout; late
    // in training a rollout can pass without any ending.
    if (this.episodeCount > 0) {
      this.fitnessHistory.push(this.episodeFitnessTotal / this.episodeCount);
      if (this.fitnessHistory.length > 500) {
        this.fitnessHistory.shift();
      }
    }
    this.episodeFitnessTotal = 0;
    this.episodeCount = 0;
    this.ppoUpdate += 1;
  }

  private computeReward(agent: Agent, previousScore: number): number {
    let reward = agent.score - previousScore;
    if (agent.terminalReason === "solved") {
      reward += VISION_CLEAR_REWARD;
    } else if (!agent.alive) {
      reward -= 1;
    }
    return reward;
  }

  private stepGreedy(agent: Agent, policy: Float32Array): void {
    this.vision.observe(agent);
    this.environment.step(agent, this.ppo.greedyAction(this.vision, policy));
  }

  private stepDisplay(): void {
    if (this.displayAgent) {
      this.stepGreedy(this.displayAgent, this.displayPolicy);
    }
    if (!this.displayAgent?.alive) {
      this.displayAgent = this.environment.createAgent(NO_POLICY);
    }

    for (let i = 0; i < this.replayAgents.length; i++) {
      const agent = this.replayAgents[i];
      this.stepGreedy(agent, this.displayPolicy);
      if (agent.alive) {
        continue;
      }
      this.replayGames += 1;
      this.replayTotalScore += agent.score;
      if (agent.terminalReason === "solved") {
        this.replayWins += 1;
      }
      this.replayAgents[i] = this.environment.createAgent(NO_POLICY);
    }
  }

  // Plays the candidate's validation games one move per step alongside
  // training, since playing them out in one go would stall the page.
  private stepCandidate(): void {
    if (this.candidateAgents.length === 0) {
      // Nothing new to try until an update has changed the averaged policy.
      if (this.candidateUpdate === this.ppoUpdate) {
        return;
      }
      this.candidatePolicy = new Float32Array(this.ppo.averagedPolicy);
      this.candidateUpdate = this.ppoUpdate;
      this.candidateFitness = 0;
      this.candidateClears = 0;
      for (let i = 0; i < VALIDATION_GAMES; i++) {
        this.candidateAgents.push(this.environment.createAgent(NO_POLICY));
      }
      return;
    }

    let playing = 0;
    for (const agent of this.candidateAgents) {
      if (!agent.alive) {
        continue;
      }
      this.stepGreedy(agent, this.candidatePolicy);
      if (agent.alive) {
        playing += 1;
      } else {
        this.candidateFitness += scoreFitness(agent) / VALIDATION_GAMES;
        if (agent.terminalReason === "solved") {
          this.candidateClears += 1;
        }
      }
    }
    if (playing > 0) {
      return;
    }

    if (!this.settled && this.candidateClears * 2 >= VALIDATION_GAMES) {
      this.settled = true;
      this.ppo.settle();
    }

    // Ties go to the newer policy, as in the other trainers.
    if (this.candidateFitness >= this.bestEverFitness) {
      this.bestEverFitness = this.candidateFitness;
      this.bestFitnessUpdate = this.candidateUpdate;
      this.setDisplayPolicy(this.candidatePolicy);
    }
    this.candidateAgents = [];
  }

  private setDisplayPolicy(policy: Float32Array): void {
    this.displayPolicy = new Float32Array(policy);
    this.displayAgent = this.environment.createAgent(NO_POLICY);
    this.replayAgents = [];
    for (let i = 0; i < REPLAY_AGENT_COUNT; i++) {
      this.replayAgents.push(this.environment.createAgent(NO_POLICY));
    }
    this.replayGames = 0;
    this.replayTotalScore = 0;
    this.replayWins = 0;
  }

  // Fills the view the renderer draws: what the shown snake's policy sees
  // and which way it leans.
  private updateView(agent: Agent): VisionView {
    const view = this.view;
    const head = agent.body[0];
    const length = agent.body.length;

    view.inside.fill(0);
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        view.inside[this.vision.cellIndex(x, y, head, agent.dir)] = 1;
      }
    }
    view.body.fill(0);
    for (let i = 0; i < length; i++) {
      const part = agent.body[i];
      view.body[this.vision.cellIndex(part.x, part.y, head, agent.dir)] =
        (length - i) / length;
    }
    view.food = this.vision.cellIndex(
      agent.food.x,
      agent.food.y,
      head,
      agent.dir,
    );

    this.vision.observe(agent);
    view.best = this.ppo.greedyAction(
      this.vision,
      this.displayPolicy,
      view.probabilities,
    );
    return view;
  }
}
