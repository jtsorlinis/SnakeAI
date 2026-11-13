import {
  OUTPUTS,
  PPO_CLIP_EPSILON,
  PPO_EPOCHS,
  PPO_MINIBATCH_SIZE,
  PPO_ROLLOUT_HORIZON,
  VISION_ENTROPY_COEFFICIENT,
  VISION_GAE_LAMBDA,
  VISION_GAMMA,
  VISION_LEARNING_RATE,
  VISION_POLICY_AVERAGE_DECAY,
} from "./config";
import { visionInputCount, visionTypicalInputs } from "./BoardVision";
import type { BoardVision } from "./BoardVision";
import { VisionNetwork } from "./VisionNetwork";

const EPSILON = 1e-8;

// PPO for Board PPO's two networks: a policy that reads the board, and a
// critic that reads the same board plus how hungry the snake is (which the
// screen doesn't show). One update is collected and trained in pieces so the
// page stays responsive: act() and recordOutcome() fill a rollout,
// finishRollout() turns it into advantages, and trainSlice() works through
// the epochs a few samples at a time.
export class VisionPPOOptimizer {
  // Running average of the policy weights; this is the policy to show.
  public readonly averagedPolicy: Float32Array;
  // Mean entropy of the policy over the last finished update, in nats.
  public policyEntropy = 0;

  private readonly policy: VisionNetwork;
  private readonly critic: VisionNetwork;
  private readonly hungerInput: number;

  private readonly horizon = PPO_ROLLOUT_HORIZON;
  private readonly agentCount: number;
  private readonly sampleCount: number;

  // Rollout storage. Sample t is step (t % horizon) of agent (t / horizon);
  // its inputs are inputIndices[starts[t] .. starts[t] + counts[t]), followed
  // by the critic's hunger input.
  private inputIndices: Int32Array;
  private inputValues: Float32Array;
  private inputCursor = 0;
  private readonly starts: Int32Array;
  private readonly counts: Int32Array;
  private readonly actions: Int8Array;
  private readonly logProbs: Float32Array;
  private readonly values: Float32Array;
  private readonly rewards: Float32Array;
  private readonly dones: Uint8Array;
  private readonly advantages: Float32Array;
  private readonly returns: Float32Array;

  private updates = 0;

  // Position within the update being trained.
  private readonly order: Int32Array;
  private epoch = 0;
  private position = 0;
  private batchStart = 0;
  private batchEnd = 0;
  private entropyTotal = 0;
  private entropySamples = 0;

  private readonly probabilities = new Float32Array(OUTPUTS);
  private readonly logitGradient = new Float32Array(OUTPUTS);
  private readonly valueGradient = new Float32Array(1);

  constructor(agentCount: number, inputsPerSample: number) {
    const inputCount = visionInputCount();
    this.hungerInput = inputCount;
    const typicalInputs = visionTypicalInputs();
    this.policy = new VisionNetwork(inputCount, OUTPUTS, typicalInputs, 0.01);
    this.critic = new VisionNetwork(inputCount + 1, 1, typicalInputs, 1);
    this.averagedPolicy = new Float32Array(this.policy.params);

    this.agentCount = agentCount;
    this.sampleCount = agentCount * this.horizon;
    // Enough for a typical rollout; grows if long snakes need more.
    const capacity = this.sampleCount * Math.ceil(inputsPerSample / 2);
    this.inputIndices = new Int32Array(capacity);
    this.inputValues = new Float32Array(capacity);
    this.starts = new Int32Array(this.sampleCount);
    this.counts = new Int32Array(this.sampleCount);
    this.actions = new Int8Array(this.sampleCount);
    this.logProbs = new Float32Array(this.sampleCount);
    this.values = new Float32Array(this.sampleCount);
    this.rewards = new Float32Array(this.sampleCount);
    this.dones = new Uint8Array(this.sampleCount);
    this.advantages = new Float32Array(this.sampleCount);
    this.returns = new Float32Array(this.sampleCount);
    this.order = new Int32Array(this.sampleCount);
  }

  // Samples a move for what the snake sees and stores the step.
  public act(
    vision: BoardVision,
    hunger: number,
    agent: number,
    step: number,
  ): number {
    const t = agent * this.horizon + step;
    const count = vision.count;
    this.reserveInputs(count + 1);
    const start = this.inputCursor;
    this.inputIndices.set(vision.indices.subarray(0, count), start);
    this.inputValues.set(vision.values.subarray(0, count), start);
    this.inputIndices[start + count] = this.hungerInput;
    this.inputValues[start + count] = hunger;
    this.inputCursor = start + count + 1;
    this.starts[t] = start;
    this.counts[t] = count;

    this.policy.forward(this.inputIndices, this.inputValues, start, count);
    this.softmax(this.policy.outputs);
    const roll = Math.random();
    const action =
      roll < this.probabilities[0]
        ? 0
        : roll < this.probabilities[0] + this.probabilities[1]
          ? 1
          : 2;

    this.critic.forward(this.inputIndices, this.inputValues, start, count + 1);
    this.actions[t] = action;
    this.logProbs[t] = Math.log(Math.max(this.probabilities[action], EPSILON));
    this.values[t] = this.critic.outputs[0];
    return action;
  }

  public recordOutcome(
    agent: number,
    step: number,
    reward: number,
    done: boolean,
  ): void {
    const t = agent * this.horizon + step;
    this.rewards[t] = reward;
    this.dones[t] = done ? 1 : 0;
  }

  // The critic's estimate for a snake that is still mid-game when the
  // rollout ends. Uses the spare slot after the vision inputs.
  public estimateValue(vision: BoardVision, hunger: number): number {
    vision.indices[vision.count] = this.hungerInput;
    vision.values[vision.count] = hunger;
    this.critic.forward(vision.indices, vision.values, 0, vision.count + 1);
    return this.critic.outputs[0];
  }

  // Call once the rollout is full. nextValues holds estimateValue() for each
  // agent's current state; entries for agents that just died are ignored.
  public finishRollout(nextValues: Float32Array): void {
    const horizon = this.horizon;
    for (let agent = 0; agent < this.agentCount; agent++) {
      let nextValue = nextValues[agent];
      let nextAdvantage = 0;
      for (let step = horizon - 1; step >= 0; step--) {
        const t = agent * horizon + step;
        const alive = this.dones[t] ? 0 : 1;
        const delta =
          this.rewards[t] + VISION_GAMMA * nextValue * alive - this.values[t];
        nextAdvantage =
          delta + VISION_GAMMA * VISION_GAE_LAMBDA * alive * nextAdvantage;
        this.advantages[t] = nextAdvantage;
        this.returns[t] = nextAdvantage + this.values[t];
        nextValue = this.values[t];
      }
    }

    let mean = 0;
    for (let t = 0; t < this.sampleCount; t++) {
      mean += this.advantages[t];
    }
    mean /= this.sampleCount;
    let variance = 0;
    for (let t = 0; t < this.sampleCount; t++) {
      const centered = this.advantages[t] - mean;
      variance += centered * centered;
    }
    const std = Math.sqrt(variance / this.sampleCount + EPSILON);
    for (let t = 0; t < this.sampleCount; t++) {
      this.advantages[t] = (this.advantages[t] - mean) / std;
    }

    for (let t = 0; t < this.sampleCount; t++) {
      this.order[t] = t;
    }
    this.epoch = 0;
    this.entropyTotal = 0;
    this.entropySamples = 0;
    this.startEpoch();
  }

  // Trains on up to maxSamples more samples of the stored rollout. Returns
  // true once the update is finished and the rollout can be refilled.
  public trainSlice(maxSamples: number): boolean {
    let budget = maxSamples;
    while (budget > 0) {
      if (this.position === this.batchEnd) {
        this.batchStart = this.position;
        this.batchEnd = Math.min(
          this.sampleCount,
          this.position + PPO_MINIBATCH_SIZE,
        );
      }

      const batchSize = this.batchEnd - this.batchStart;
      const end = Math.min(this.batchEnd, this.position + budget);
      budget -= end - this.position;
      while (this.position < end) {
        this.trainSample(this.order[this.position], batchSize);
        this.position += 1;
      }
      if (this.position < this.batchEnd) {
        break;
      }

      this.policy.applyGradient(VISION_LEARNING_RATE);
      this.critic.applyGradient(VISION_LEARNING_RATE);
      if (this.position === this.sampleCount) {
        this.epoch += 1;
        if (this.epoch === PPO_EPOCHS) {
          this.finishUpdate();
          return true;
        }
        this.startEpoch();
      }
    }
    return false;
  }

  // The move a saved copy of the policy weights prefers; also leaves the
  // preferences for all three moves in probabilities.
  public greedyAction(
    vision: BoardVision,
    params: Float32Array,
    probabilities?: Float32Array,
  ): number {
    this.policy.forward(vision.indices, vision.values, 0, vision.count, params);
    const outputs = this.policy.outputs;
    if (probabilities) {
      this.softmax(outputs);
      probabilities.set(this.probabilities);
    }
    let best = 0;
    for (let o = 1; o < OUTPUTS; o++) {
      if (outputs[o] > outputs[best]) {
        best = o;
      }
    }
    return best;
  }

  private startEpoch(): void {
    for (let i = this.sampleCount - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const swap = this.order[i];
      this.order[i] = this.order[j];
      this.order[j] = swap;
    }
    this.position = 0;
    this.batchStart = 0;
    this.batchEnd = 0;
  }

  private finishUpdate(): void {
    // A plain mean over the first updates, so the random starting weights
    // drop out at once instead of lingering in the average for fifty updates.
    this.updates += 1;
    const decay = Math.min(VISION_POLICY_AVERAGE_DECAY, 1 - 1 / this.updates);
    const live = this.policy.params;
    const averaged = this.averagedPolicy;
    for (let i = 0; i < averaged.length; i++) {
      averaged[i] = decay * averaged[i] + (1 - decay) * live[i];
    }
    this.policyEntropy = this.entropyTotal / Math.max(1, this.entropySamples);
    this.inputCursor = 0;
  }

  private trainSample(t: number, batchSize: number): void {
    const start = this.starts[t];
    const count = this.counts[t];
    const action = this.actions[t];
    const probabilities = this.probabilities;

    this.policy.forward(this.inputIndices, this.inputValues, start, count);
    this.softmax(this.policy.outputs);

    const ratio = Math.exp(
      Math.log(Math.max(probabilities[action], EPSILON)) - this.logProbs[t],
    );
    const advantage = this.advantages[t];
    const clippedRatio = Math.min(
      Math.max(ratio, 1 - PPO_CLIP_EPSILON),
      1 + PPO_CLIP_EPSILON,
    );
    // No gradient where the clipped objective is the one in force.
    const clipActive =
      clippedRatio !== ratio && clippedRatio * advantage < ratio * advantage;
    const dLossDLogProb = clipActive ? 0 : (-advantage * ratio) / batchSize;

    let entropy = 0;
    let weightedLogSum = 0;
    for (let o = 0; o < OUTPUTS; o++) {
      const p = Math.max(probabilities[o], EPSILON);
      entropy -= p * Math.log(p);
      weightedLogSum += p * (Math.log(p) + 1);
    }
    this.entropyTotal += entropy;
    this.entropySamples += 1;

    for (let o = 0; o < OUTPUTS; o++) {
      const p = probabilities[o];
      const logP = Math.log(Math.max(p, EPSILON));
      this.logitGradient[o] =
        dLossDLogProb * ((o === action ? 1 : 0) - p) -
        (VISION_ENTROPY_COEFFICIENT * p * (weightedLogSum - (logP + 1))) /
          batchSize;
    }
    this.policy.backward(
      this.inputIndices,
      this.inputValues,
      start,
      count,
      this.logitGradient,
    );

    this.critic.forward(this.inputIndices, this.inputValues, start, count + 1);
    this.valueGradient[0] =
      (this.critic.outputs[0] - this.returns[t]) / batchSize;
    this.critic.backward(
      this.inputIndices,
      this.inputValues,
      start,
      count + 1,
      this.valueGradient,
    );
  }

  private softmax(logits: Float32Array): void {
    let max = logits[0];
    for (let o = 1; o < OUTPUTS; o++) {
      if (logits[o] > max) {
        max = logits[o];
      }
    }
    let sum = 0;
    for (let o = 0; o < OUTPUTS; o++) {
      this.probabilities[o] = Math.exp(logits[o] - max);
      sum += this.probabilities[o];
    }
    for (let o = 0; o < OUTPUTS; o++) {
      this.probabilities[o] /= sum;
    }
  }

  private reserveInputs(extra: number): void {
    const needed = this.inputCursor + extra;
    if (needed <= this.inputIndices.length) {
      return;
    }
    const capacity = Math.max(needed, Math.ceil(this.inputIndices.length * 1.5));
    const indices = new Int32Array(capacity);
    const values = new Float32Array(capacity);
    indices.set(this.inputIndices.subarray(0, this.inputCursor));
    values.set(this.inputValues.subarray(0, this.inputCursor));
    this.inputIndices = indices;
    this.inputValues = values;
  }
}
