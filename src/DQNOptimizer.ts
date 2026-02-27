import {
  DQN_BATCH_SIZE,
  DQN_GAMMA,
  DQN_LEARNING_RATE,
  DQN_MAX_GRAD_NORM,
  DQN_MIN_REPLAY_SIZE,
  DQN_REPLAY_CAPACITY,
  DQN_TARGET_UPDATE_INTERVAL,
  HIDDEN_LAYER_UNITS,
  INPUTS,
  OFFSET_HH,
  OFFSET_HO,
  OFFSET_H_BIAS,
  OFFSET_IH,
  OFFSET_O_BIAS,
  OUTPUTS,
  POLICY_PARAM_COUNT,
} from "./config";
import type { PolicyParams } from "./types";

const EPSILON = 1e-8;
const HIDDEN_LAYER_COUNT = HIDDEN_LAYER_UNITS.length;

export type DQNTransition = {
  observation: Float32Array;
  action: number;
  reward: number;
  nextObservation: Float32Array;
  done: boolean;
};

export type DQNActionSample = {
  action: number;
  exploratory: boolean;
};

export type DQNUpdateStats = {
  steps: number;
  loss: number;
  replaySize: number;
  batchSize: number;
};

function createHiddenBuffers(): Float32Array[] {
  return HIDDEN_LAYER_UNITS.map((units) => new Float32Array(units));
}

function buildHiddenWeightOffsets(firstOffset: number, hhOffset: number): number[] {
  if (HIDDEN_LAYER_COUNT === 0) {
    return [];
  }

  const offsets: number[] = [firstOffset];
  let nextOffset = hhOffset;
  for (let layer = 1; layer < HIDDEN_LAYER_COUNT; layer++) {
    offsets.push(nextOffset);
    nextOffset += HIDDEN_LAYER_UNITS[layer - 1] * HIDDEN_LAYER_UNITS[layer];
  }
  return offsets;
}

function buildHiddenBiasOffsets(startOffset: number): number[] {
  if (HIDDEN_LAYER_COUNT === 0) {
    return [];
  }

  const offsets: number[] = [];
  let nextOffset = startOffset;
  for (let layer = 0; layer < HIDDEN_LAYER_COUNT; layer++) {
    offsets.push(nextOffset);
    nextOffset += HIDDEN_LAYER_UNITS[layer];
  }
  return offsets;
}

function randomRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

class AdamOptimizer {
  private readonly m: Float32Array;
  private readonly v: Float32Array;
  private beta1Power = 1;
  private beta2Power = 1;

  constructor(
    size: number,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly epsilon = 1e-8,
  ) {
    this.m = new Float32Array(size);
    this.v = new Float32Array(size);
  }

  public reset(): void {
    this.m.fill(0);
    this.v.fill(0);
    this.beta1Power = 1;
    this.beta2Power = 1;
  }

  public step(
    params: Float32Array,
    gradient: Float32Array,
    learningRate: number,
  ): void {
    this.beta1Power *= this.beta1;
    this.beta2Power *= this.beta2;
    const invBias1 = 1 / (1 - this.beta1Power);
    const invBias2 = 1 / (1 - this.beta2Power);

    for (let i = 0; i < params.length; i++) {
      const g = gradient[i];
      const m = this.beta1 * this.m[i] + (1 - this.beta1) * g;
      const v = this.beta2 * this.v[i] + (1 - this.beta2) * g * g;
      this.m[i] = m;
      this.v[i] = v;

      const mHat = m * invBias1;
      const vHat = v * invBias2;
      params[i] -= learningRate * (mHat / (Math.sqrt(vHat) + this.epsilon));
    }
  }
}

export class DQNOptimizer {
  private readonly qParams = new Float32Array(POLICY_PARAM_COUNT);
  private readonly targetParams = new Float32Array(POLICY_PARAM_COUNT);
  private readonly optimizer = new AdamOptimizer(POLICY_PARAM_COUNT);

  private readonly hiddenWeightOffsets = buildHiddenWeightOffsets(
    OFFSET_IH,
    OFFSET_HH,
  );
  private readonly hiddenBiasOffsets = buildHiddenBiasOffsets(OFFSET_H_BIAS);

  private readonly onlineHidden = createHiddenBuffers();
  private readonly targetHidden = createHiddenBuffers();
  private readonly nextOnlineHidden = createHiddenBuffers();
  private readonly hiddenGrad = createHiddenBuffers();
  private readonly qValues = new Float32Array(OUTPUTS);
  private readonly nextQValues = new Float32Array(OUTPUTS);
  private readonly nextOnlineQValues = new Float32Array(OUTPUTS);
  private readonly outputGradient = new Float32Array(OUTPUTS);
  private readonly gradient = new Float32Array(POLICY_PARAM_COUNT);

  private readonly observations = new Float32Array(DQN_REPLAY_CAPACITY * INPUTS);
  private readonly nextObservations = new Float32Array(
    DQN_REPLAY_CAPACITY * INPUTS,
  );
  private readonly actions = new Int16Array(DQN_REPLAY_CAPACITY);
  private readonly rewards = new Float32Array(DQN_REPLAY_CAPACITY);
  private readonly dones = new Uint8Array(DQN_REPLAY_CAPACITY);
  private replaySize = 0;
  private replayCursor = 0;
  private optimizationStep = 0;

  private readonly observationBuffer = new Float32Array(INPUTS);
  private readonly nextObservationBuffer = new Float32Array(INPUTS);

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.randomize(this.qParams);
    this.targetParams.set(this.qParams);
    this.optimizer.reset();
    this.replaySize = 0;
    this.replayCursor = 0;
    this.optimizationStep = 0;
  }

  public getPolicyParams(): PolicyParams {
    return this.qParams;
  }

  public getReplaySize(): number {
    return this.replaySize;
  }

  public sampleAction(
    observation: Float32Array,
    epsilon: number,
  ): DQNActionSample {
    if (Math.random() < epsilon) {
      return {
        action: Math.floor(Math.random() * OUTPUTS),
        exploratory: true,
      };
    }

    this.forward(this.qParams, observation, this.onlineHidden, this.qValues);
    return {
      action: this.argmax(this.qValues),
      exploratory: false,
    };
  }

  public addTransition(transition: DQNTransition): void {
    const index = this.replayCursor;
    const offset = index * INPUTS;

    for (let i = 0; i < INPUTS; i++) {
      this.observations[offset + i] = transition.observation[i];
      this.nextObservations[offset + i] = transition.nextObservation[i];
    }

    this.actions[index] = transition.action;
    this.rewards[index] = transition.reward;
    this.dones[index] = transition.done ? 1 : 0;

    this.replayCursor = (this.replayCursor + 1) % DQN_REPLAY_CAPACITY;
    if (this.replaySize < DQN_REPLAY_CAPACITY) {
      this.replaySize += 1;
    }
  }

  public train(steps: number): DQNUpdateStats {
    if (steps <= 0 || this.replaySize < DQN_MIN_REPLAY_SIZE) {
      return {
        steps: 0,
        loss: 0,
        replaySize: this.replaySize,
        batchSize: Math.min(DQN_BATCH_SIZE, this.replaySize),
      };
    }

    const batchSize = Math.min(DQN_BATCH_SIZE, this.replaySize);
    let totalLoss = 0;
    let trainedSteps = 0;

    for (let step = 0; step < steps; step++) {
      this.gradient.fill(0);
      let batchLoss = 0;

      for (let sample = 0; sample < batchSize; sample++) {
        const index = Math.floor(Math.random() * this.replaySize);
        this.loadObservation(this.observations, index, this.observationBuffer);
        this.forward(
          this.qParams,
          this.observationBuffer,
          this.onlineHidden,
          this.qValues,
        );

        const action = this.actions[index];
        const predicted = this.qValues[action];

        let target = this.rewards[index];
        if (this.dones[index] === 0) {
          this.loadObservation(
            this.nextObservations,
            index,
            this.nextObservationBuffer,
          );
          this.forward(
            this.qParams,
            this.nextObservationBuffer,
            this.nextOnlineHidden,
            this.nextOnlineQValues,
          );
          this.forward(
            this.targetParams,
            this.nextObservationBuffer,
            this.targetHidden,
            this.nextQValues,
          );
          const nextAction = this.argmax(this.nextOnlineQValues);
          target += DQN_GAMMA * this.nextQValues[nextAction];
        }

        const tdError = predicted - target;
        batchLoss += this.huberLoss(tdError);

        this.outputGradient.fill(0);
        this.outputGradient[action] = this.huberGradient(tdError) / batchSize;
        this.backward(
          this.observationBuffer,
          this.onlineHidden,
          this.outputGradient,
          this.gradient,
        );
      }

      this.clipGradient(this.gradient, DQN_MAX_GRAD_NORM);
      this.optimizer.step(this.qParams, this.gradient, DQN_LEARNING_RATE);
      this.optimizationStep += 1;

      if (this.optimizationStep % DQN_TARGET_UPDATE_INTERVAL === 0) {
        this.targetParams.set(this.qParams);
      }

      totalLoss += batchLoss / batchSize;
      trainedSteps += 1;
    }

    return {
      steps: trainedSteps,
      loss: totalLoss / Math.max(1, trainedSteps),
      replaySize: this.replaySize,
      batchSize,
    };
  }

  private randomize(target: Float32Array): void {
    target.fill(0);

    if (HIDDEN_LAYER_COUNT > 0) {
      this.initializeWeightBlock(
        target,
        OFFSET_IH,
        HIDDEN_LAYER_UNITS[0],
        INPUTS,
      );

      let hhOffset = OFFSET_HH;
      for (let layer = 1; layer < HIDDEN_LAYER_COUNT; layer++) {
        const inputSize = HIDDEN_LAYER_UNITS[layer - 1];
        const outputSize = HIDDEN_LAYER_UNITS[layer];
        this.initializeWeightBlock(target, hhOffset, outputSize, inputSize);
        hhOffset += outputSize * inputSize;
      }

      const outputInputSize = HIDDEN_LAYER_UNITS[HIDDEN_LAYER_COUNT - 1];
      this.initializeWeightBlock(target, OFFSET_HO, OUTPUTS, outputInputSize);
      return;
    }

    this.initializeWeightBlock(target, OFFSET_HO, OUTPUTS, INPUTS);
  }

  private argmax(values: Float32Array): number {
    let bestIndex = 0;
    let bestValue = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] > bestValue) {
        bestValue = values[i];
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  private initializeWeightBlock(
    target: Float32Array,
    offset: number,
    outputSize: number,
    inputSize: number,
  ): void {
    const limit = Math.sqrt(6 / (inputSize + outputSize));
    const length = outputSize * inputSize;
    for (let i = 0; i < length; i++) {
      target[offset + i] = randomRange(-limit, limit);
    }
  }

  private huberLoss(error: number): number {
    const absError = Math.abs(error);
    if (absError <= 1) {
      return 0.5 * error * error;
    }
    return absError - 0.5;
  }

  private huberGradient(error: number): number {
    if (error > 1) {
      return 1;
    }
    if (error < -1) {
      return -1;
    }
    return error;
  }

  private loadObservation(
    source: Float32Array,
    index: number,
    target: Float32Array,
  ): void {
    const offset = index * INPUTS;
    for (let i = 0; i < INPUTS; i++) {
      target[i] = source[offset + i];
    }
  }

  private forward(
    params: Float32Array,
    observation: Float32Array,
    hiddenTarget: Float32Array[],
    qValueTarget: Float32Array,
  ): void {
    if (HIDDEN_LAYER_COUNT > 0) {
      for (let layer = 0; layer < HIDDEN_LAYER_COUNT; layer++) {
        const current = hiddenTarget[layer];
        const currentSize = HIDDEN_LAYER_UNITS[layer];
        const prev = layer === 0 ? observation : hiddenTarget[layer - 1];
        const prevSize = layer === 0 ? INPUTS : HIDDEN_LAYER_UNITS[layer - 1];
        const weightsOffset = this.hiddenWeightOffsets[layer];
        const biasOffset = this.hiddenBiasOffsets[layer];

        for (let h = 0; h < currentSize; h++) {
          let sum = params[biasOffset + h];
          const wOffset = weightsOffset + h * prevSize;
          for (let p = 0; p < prevSize; p++) {
            sum += params[wOffset + p] * prev[p];
          }
          current[h] = Math.tanh(sum);
        }
      }
    }

    const outputInput =
      HIDDEN_LAYER_COUNT > 0
        ? hiddenTarget[HIDDEN_LAYER_COUNT - 1]
        : observation;
    const outputInputSize =
      HIDDEN_LAYER_COUNT > 0
        ? HIDDEN_LAYER_UNITS[HIDDEN_LAYER_COUNT - 1]
        : INPUTS;

    for (let output = 0; output < OUTPUTS; output++) {
      let sum = params[OFFSET_O_BIAS + output];
      const wOffset = OFFSET_HO + output * outputInputSize;
      for (let i = 0; i < outputInputSize; i++) {
        sum += params[wOffset + i] * outputInput[i];
      }
      qValueTarget[output] = sum;
    }
  }

  private backward(
    observation: Float32Array,
    hidden: Float32Array[],
    outputGradient: Float32Array,
    gradientTarget: Float32Array,
  ): void {
    if (HIDDEN_LAYER_COUNT === 0) {
      for (let output = 0; output < OUTPUTS; output++) {
        const grad = outputGradient[output];
        gradientTarget[OFFSET_O_BIAS + output] += grad;
        const wOffset = OFFSET_HO + output * INPUTS;
        for (let i = 0; i < INPUTS; i++) {
          gradientTarget[wOffset + i] += grad * observation[i];
        }
      }
      return;
    }

    for (let layer = 0; layer < this.hiddenGrad.length; layer++) {
      this.hiddenGrad[layer].fill(0);
    }

    const lastLayer = HIDDEN_LAYER_COUNT - 1;
    const lastSize = HIDDEN_LAYER_UNITS[lastLayer];
    const lastHidden = hidden[lastLayer];
    const lastGrad = this.hiddenGrad[lastLayer];

    for (let output = 0; output < OUTPUTS; output++) {
      const grad = outputGradient[output];
      gradientTarget[OFFSET_O_BIAS + output] += grad;
      const wOffset = OFFSET_HO + output * lastSize;
      for (let h = 0; h < lastSize; h++) {
        gradientTarget[wOffset + h] += grad * lastHidden[h];
        lastGrad[h] += this.qParams[wOffset + h] * grad;
      }
    }

    for (let layer = lastLayer; layer >= 0; layer--) {
      const current = hidden[layer];
      const currentSize = HIDDEN_LAYER_UNITS[layer];
      const prev = layer === 0 ? observation : hidden[layer - 1];
      const prevSize = layer === 0 ? INPUTS : HIDDEN_LAYER_UNITS[layer - 1];
      const hiddenGrad = this.hiddenGrad[layer];
      const weightsOffset = this.hiddenWeightOffsets[layer];
      const biasOffset = this.hiddenBiasOffsets[layer];

      for (let h = 0; h < currentSize; h++) {
        const activation = current[h];
        const delta = hiddenGrad[h] * (1 - activation * activation);
        gradientTarget[biasOffset + h] += delta;

        const wOffset = weightsOffset + h * prevSize;
        for (let p = 0; p < prevSize; p++) {
          gradientTarget[wOffset + p] += delta * prev[p];
          if (layer > 0) {
            this.hiddenGrad[layer - 1][p] += this.qParams[wOffset + p] * delta;
          }
        }
      }
    }
  }

  private clipGradient(gradient: Float32Array, maxNorm: number): void {
    if (maxNorm <= 0) {
      return;
    }

    let normSquared = 0;
    for (let i = 0; i < gradient.length; i++) {
      const g = gradient[i];
      normSquared += g * g;
    }

    const norm = Math.sqrt(normSquared);
    if (norm <= maxNorm || norm < EPSILON) {
      return;
    }

    const scale = maxNorm / norm;
    for (let i = 0; i < gradient.length; i++) {
      gradient[i] *= scale;
    }
  }
}
