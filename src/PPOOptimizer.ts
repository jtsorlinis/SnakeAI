import {
  GENE_COUNT,
  HIDDEN_LAYER_UNITS,
  HH_COUNT,
  H_BIAS_COUNT,
  IH_COUNT,
  INPUTS,
  OFFSET_HO,
  OFFSET_HH,
  OFFSET_H_BIAS,
  OFFSET_IH,
  OFFSET_O_BIAS,
  OUTPUTS,
  PPO_CLIP_EPSILON,
  PPO_ENTROPY_COEFFICIENT,
  PPO_EPOCHS,
  PPO_GAE_LAMBDA,
  PPO_GAMMA,
  PPO_MAX_GRAD_NORM,
  PPO_MINIBATCH_SIZE,
  PPO_POLICY_LEARNING_RATE,
  PPO_VALUE_COEFFICIENT,
  PPO_VALUE_LEARNING_RATE,
} from "./config";
import type { Genome } from "./types";

const EPSILON = 1e-8;
const HIDDEN_LAYER_COUNT = HIDDEN_LAYER_UNITS.length;

const VALUE_OUTPUT_INPUTS =
  HIDDEN_LAYER_COUNT > 0 ? HIDDEN_LAYER_UNITS[HIDDEN_LAYER_COUNT - 1] : INPUTS;
const VALUE_OFFSET_IH = 0;
const VALUE_OFFSET_HH = VALUE_OFFSET_IH + IH_COUNT;
const VALUE_OFFSET_H_BIAS = VALUE_OFFSET_HH + HH_COUNT;
const VALUE_OFFSET_HO = VALUE_OFFSET_H_BIAS + H_BIAS_COUNT;
const VALUE_OFFSET_O_BIAS = VALUE_OFFSET_HO + VALUE_OUTPUT_INPUTS;
const VALUE_PARAM_COUNT = VALUE_OFFSET_O_BIAS + 1;

export type PPOTransition = {
  observation: Float32Array;
  action: number;
  reward: number;
  done: boolean;
  logProb: number;
  value: number;
};

export type ActionSample = {
  action: number;
  logProb: number;
  value: number;
};

export type PPOUpdateStats = {
  batchSize: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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

type BatchData = {
  observations: Float32Array[];
  actions: Int16Array;
  oldLogProbs: Float32Array;
  advantages: Float32Array;
  returns: Float32Array;
  size: number;
};

export class PPOOptimizer {
  private readonly policyParams = new Float32Array(GENE_COUNT);
  private readonly valueParams = new Float32Array(VALUE_PARAM_COUNT);
  private readonly policyOptimizer = new AdamOptimizer(GENE_COUNT);
  private readonly valueOptimizer = new AdamOptimizer(VALUE_PARAM_COUNT);

  private readonly policyHiddenWeightOffsets = buildHiddenWeightOffsets(
    OFFSET_IH,
    OFFSET_HH,
  );
  private readonly policyHiddenBiasOffsets = buildHiddenBiasOffsets(OFFSET_H_BIAS);
  private readonly valueHiddenWeightOffsets = buildHiddenWeightOffsets(
    VALUE_OFFSET_IH,
    VALUE_OFFSET_HH,
  );
  private readonly valueHiddenBiasOffsets = buildHiddenBiasOffsets(
    VALUE_OFFSET_H_BIAS,
  );

  private readonly policyHidden = createHiddenBuffers();
  private readonly valueHidden = createHiddenBuffers();
  private readonly policyHiddenGrad = createHiddenBuffers();
  private readonly valueHiddenGrad = createHiddenBuffers();
  private readonly logits = new Float32Array(OUTPUTS);
  private readonly probabilities = new Float32Array(OUTPUTS);
  private readonly policyGradient = new Float32Array(GENE_COUNT);
  private readonly valueGradient = new Float32Array(VALUE_PARAM_COUNT);
  private readonly logitsGradient = new Float32Array(OUTPUTS);
  private readonly entropyGradientBuffer = new Float32Array(OUTPUTS);

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.randomize(this.policyParams);
    this.randomize(this.valueParams);
    this.policyOptimizer.reset();
    this.valueOptimizer.reset();
  }

  public getPolicyGenome(): Genome {
    return this.policyParams;
  }

  public sampleAction(observation: Float32Array): ActionSample {
    this.forwardPolicy(observation, this.policyHidden, this.logits);
    this.logitsToProbabilities(this.logits, this.probabilities);

    const action = this.sampleFromProbabilities(this.probabilities);
    const logProb = Math.log(Math.max(this.probabilities[action], EPSILON));
    const value = this.forwardValue(observation, this.valueHidden);

    return { action, logProb, value };
  }

  public train(episodes: readonly PPOTransition[][]): PPOUpdateStats {
    const batch = this.buildBatch(episodes);
    if (batch.size === 0) {
      return {
        batchSize: 0,
        policyLoss: 0,
        valueLoss: 0,
        entropy: 0,
      };
    }

    this.normalizeAdvantages(batch.advantages, batch.size);

    const indices = new Int32Array(batch.size);
    for (let i = 0; i < batch.size; i++) {
      indices[i] = i;
    }

    let totalPolicyLoss = 0;
    let totalValueLoss = 0;
    let totalEntropy = 0;
    let totalMiniBatches = 0;

    for (let epoch = 0; epoch < PPO_EPOCHS; epoch++) {
      this.shuffle(indices);

      for (let start = 0; start < batch.size; start += PPO_MINIBATCH_SIZE) {
        const end = Math.min(batch.size, start + PPO_MINIBATCH_SIZE);
        const miniBatchSize = end - start;
        if (miniBatchSize <= 0) {
          continue;
        }

        this.policyGradient.fill(0);
        this.valueGradient.fill(0);

        let miniPolicyLoss = 0;
        let miniValueLoss = 0;
        let miniEntropy = 0;

        for (let i = start; i < end; i++) {
          const sampleIndex = indices[i];
          const observation = batch.observations[sampleIndex];
          const action = batch.actions[sampleIndex];
          const advantage = batch.advantages[sampleIndex];
          const oldLogProb = batch.oldLogProbs[sampleIndex];
          const returnTarget = batch.returns[sampleIndex];

          this.forwardPolicy(observation, this.policyHidden, this.logits);
          this.logitsToProbabilities(this.logits, this.probabilities);

          const actionProb = Math.max(this.probabilities[action], EPSILON);
          const logProb = Math.log(actionProb);
          const ratio = Math.exp(logProb - oldLogProb);

          const clippedRatio = clamp(
            ratio,
            1 - PPO_CLIP_EPSILON,
            1 + PPO_CLIP_EPSILON,
          );
          const unclippedObjective = ratio * advantage;
          const clippedObjective = clippedRatio * advantage;
          const objective = Math.min(unclippedObjective, clippedObjective);
          miniPolicyLoss += -objective;

          let dLossDLogProb = 0;
          const clipped =
            ratio < 1 - PPO_CLIP_EPSILON || ratio > 1 + PPO_CLIP_EPSILON;
          const useClippedObjective = clippedObjective < unclippedObjective;
          if (!(clipped && useClippedObjective)) {
            dLossDLogProb = (-(advantage * ratio)) / miniBatchSize;
          }

          for (let output = 0; output < OUTPUTS; output++) {
            const oneHot = output === action ? 1 : 0;
            this.logitsGradient[output] =
              dLossDLogProb * (oneHot - this.probabilities[output]);
          }

          const entropy = this.entropy(this.probabilities);
          miniEntropy += entropy;
          if (PPO_ENTROPY_COEFFICIENT > 0) {
            this.entropyGradient(this.probabilities, this.entropyGradientBuffer);
            for (let output = 0; output < OUTPUTS; output++) {
              this.logitsGradient[output] -=
                (PPO_ENTROPY_COEFFICIENT * this.entropyGradientBuffer[output]) /
                miniBatchSize;
            }
          }

          this.backwardPolicy(
            observation,
            this.policyHidden,
            this.logitsGradient,
            this.policyGradient,
          );

          const value = this.forwardValue(observation, this.valueHidden);
          const valueError = value - returnTarget;
          miniValueLoss += 0.5 * valueError * valueError;

          const dLossDValue =
            (PPO_VALUE_COEFFICIENT * valueError) / miniBatchSize;
          this.backwardValue(
            observation,
            this.valueHidden,
            dLossDValue,
            this.valueGradient,
          );
        }

        this.clipGradient(this.policyGradient, PPO_MAX_GRAD_NORM);
        this.clipGradient(this.valueGradient, PPO_MAX_GRAD_NORM);

        this.policyOptimizer.step(
          this.policyParams,
          this.policyGradient,
          PPO_POLICY_LEARNING_RATE,
        );
        this.valueOptimizer.step(
          this.valueParams,
          this.valueGradient,
          PPO_VALUE_LEARNING_RATE,
        );

        totalPolicyLoss += miniPolicyLoss / miniBatchSize;
        totalValueLoss += miniValueLoss / miniBatchSize;
        totalEntropy += miniEntropy / miniBatchSize;
        totalMiniBatches += 1;
      }
    }

    const divisor = Math.max(1, totalMiniBatches);
    return {
      batchSize: batch.size,
      policyLoss: totalPolicyLoss / divisor,
      valueLoss: totalValueLoss / divisor,
      entropy: totalEntropy / divisor,
    };
  }

  private randomize(target: Float32Array): void {
    for (let i = 0; i < target.length; i++) {
      target[i] = this.randomRange(-1, 1);
    }
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
  }

  private sampleFromProbabilities(probabilities: Float32Array): number {
    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (r <= cumulative) {
        return i;
      }
    }
    return probabilities.length - 1;
  }

  private shuffle(indices: Int32Array): void {
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const temp = indices[i];
      indices[i] = indices[j];
      indices[j] = temp;
    }
  }

  private entropy(probabilities: Float32Array): number {
    let entropy = 0;
    for (let i = 0; i < probabilities.length; i++) {
      const p = Math.max(probabilities[i], EPSILON);
      entropy -= p * Math.log(p);
    }
    return entropy;
  }

  private entropyGradient(
    probabilities: Float32Array,
    target: Float32Array,
  ): void {
    let weightedLogSum = 0;
    for (let i = 0; i < probabilities.length; i++) {
      const p = Math.max(probabilities[i], EPSILON);
      weightedLogSum += p * (Math.log(p) + 1);
    }

    for (let i = 0; i < probabilities.length; i++) {
      const p = probabilities[i];
      const logP = Math.log(Math.max(p, EPSILON));
      target[i] = p * (weightedLogSum - (logP + 1));
    }
  }

  private logitsToProbabilities(
    logits: Float32Array,
    target: Float32Array,
  ): void {
    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) {
        maxLogit = logits[i];
      }
    }

    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      const expValue = Math.exp(logits[i] - maxLogit);
      target[i] = expValue;
      sum += expValue;
    }

    const invSum = sum > 0 ? 1 / sum : 1 / logits.length;
    for (let i = 0; i < logits.length; i++) {
      target[i] *= invSum;
    }
  }

  private forwardPolicy(
    observation: Float32Array,
    hiddenTarget: Float32Array[],
    logitsTarget: Float32Array,
  ): void {
    if (HIDDEN_LAYER_COUNT > 0) {
      for (let layer = 0; layer < HIDDEN_LAYER_COUNT; layer++) {
        const current = hiddenTarget[layer];
        const currentSize = HIDDEN_LAYER_UNITS[layer];
        const prev = layer === 0 ? observation : hiddenTarget[layer - 1];
        const prevSize = layer === 0 ? INPUTS : HIDDEN_LAYER_UNITS[layer - 1];
        const weightsOffset = this.policyHiddenWeightOffsets[layer];
        const biasOffset = this.policyHiddenBiasOffsets[layer];

        for (let h = 0; h < currentSize; h++) {
          let sum = this.policyParams[biasOffset + h];
          const wOffset = weightsOffset + h * prevSize;
          for (let p = 0; p < prevSize; p++) {
            sum += this.policyParams[wOffset + p] * prev[p];
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
      let sum = this.policyParams[OFFSET_O_BIAS + output];
      const wOffset = OFFSET_HO + output * outputInputSize;
      for (let i = 0; i < outputInputSize; i++) {
        sum += this.policyParams[wOffset + i] * outputInput[i];
      }
      logitsTarget[output] = sum;
    }
  }

  private forwardValue(
    observation: Float32Array,
    hiddenTarget: Float32Array[],
  ): number {
    if (HIDDEN_LAYER_COUNT > 0) {
      for (let layer = 0; layer < HIDDEN_LAYER_COUNT; layer++) {
        const current = hiddenTarget[layer];
        const currentSize = HIDDEN_LAYER_UNITS[layer];
        const prev = layer === 0 ? observation : hiddenTarget[layer - 1];
        const prevSize = layer === 0 ? INPUTS : HIDDEN_LAYER_UNITS[layer - 1];
        const weightsOffset = this.valueHiddenWeightOffsets[layer];
        const biasOffset = this.valueHiddenBiasOffsets[layer];

        for (let h = 0; h < currentSize; h++) {
          let sum = this.valueParams[biasOffset + h];
          const wOffset = weightsOffset + h * prevSize;
          for (let p = 0; p < prevSize; p++) {
            sum += this.valueParams[wOffset + p] * prev[p];
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

    let value = this.valueParams[VALUE_OFFSET_O_BIAS];
    for (let i = 0; i < outputInputSize; i++) {
      value += this.valueParams[VALUE_OFFSET_HO + i] * outputInput[i];
    }
    return value;
  }

  private backwardPolicy(
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

    for (let layer = 0; layer < this.policyHiddenGrad.length; layer++) {
      this.policyHiddenGrad[layer].fill(0);
    }

    const lastLayer = HIDDEN_LAYER_COUNT - 1;
    const lastSize = HIDDEN_LAYER_UNITS[lastLayer];
    const lastHidden = hidden[lastLayer];
    const lastGrad = this.policyHiddenGrad[lastLayer];

    for (let output = 0; output < OUTPUTS; output++) {
      const grad = outputGradient[output];
      gradientTarget[OFFSET_O_BIAS + output] += grad;
      const wOffset = OFFSET_HO + output * lastSize;
      for (let h = 0; h < lastSize; h++) {
        gradientTarget[wOffset + h] += grad * lastHidden[h];
        lastGrad[h] += this.policyParams[wOffset + h] * grad;
      }
    }

    for (let layer = lastLayer; layer >= 0; layer--) {
      const current = hidden[layer];
      const currentSize = HIDDEN_LAYER_UNITS[layer];
      const prev = layer === 0 ? observation : hidden[layer - 1];
      const prevSize = layer === 0 ? INPUTS : HIDDEN_LAYER_UNITS[layer - 1];
      const hiddenGrad = this.policyHiddenGrad[layer];
      const weightsOffset = this.policyHiddenWeightOffsets[layer];
      const biasOffset = this.policyHiddenBiasOffsets[layer];

      for (let h = 0; h < currentSize; h++) {
        const activation = current[h];
        const delta = hiddenGrad[h] * (1 - activation * activation);
        gradientTarget[biasOffset + h] += delta;

        const wOffset = weightsOffset + h * prevSize;
        for (let p = 0; p < prevSize; p++) {
          gradientTarget[wOffset + p] += delta * prev[p];
          if (layer > 0) {
            this.policyHiddenGrad[layer - 1][p] +=
              this.policyParams[wOffset + p] * delta;
          }
        }
      }
    }
  }

  private backwardValue(
    observation: Float32Array,
    hidden: Float32Array[],
    outputGradient: number,
    gradientTarget: Float32Array,
  ): void {
    if (HIDDEN_LAYER_COUNT === 0) {
      gradientTarget[VALUE_OFFSET_O_BIAS] += outputGradient;
      for (let i = 0; i < INPUTS; i++) {
        gradientTarget[VALUE_OFFSET_HO + i] += outputGradient * observation[i];
      }
      return;
    }

    for (let layer = 0; layer < this.valueHiddenGrad.length; layer++) {
      this.valueHiddenGrad[layer].fill(0);
    }

    const lastLayer = HIDDEN_LAYER_COUNT - 1;
    const lastSize = HIDDEN_LAYER_UNITS[lastLayer];
    const lastHidden = hidden[lastLayer];
    const lastGrad = this.valueHiddenGrad[lastLayer];

    gradientTarget[VALUE_OFFSET_O_BIAS] += outputGradient;
    for (let h = 0; h < lastSize; h++) {
      gradientTarget[VALUE_OFFSET_HO + h] += outputGradient * lastHidden[h];
      lastGrad[h] += this.valueParams[VALUE_OFFSET_HO + h] * outputGradient;
    }

    for (let layer = lastLayer; layer >= 0; layer--) {
      const current = hidden[layer];
      const currentSize = HIDDEN_LAYER_UNITS[layer];
      const prev = layer === 0 ? observation : hidden[layer - 1];
      const prevSize = layer === 0 ? INPUTS : HIDDEN_LAYER_UNITS[layer - 1];
      const hiddenGrad = this.valueHiddenGrad[layer];
      const weightsOffset = this.valueHiddenWeightOffsets[layer];
      const biasOffset = this.valueHiddenBiasOffsets[layer];

      for (let h = 0; h < currentSize; h++) {
        const activation = current[h];
        const delta = hiddenGrad[h] * (1 - activation * activation);
        gradientTarget[biasOffset + h] += delta;

        const wOffset = weightsOffset + h * prevSize;
        for (let p = 0; p < prevSize; p++) {
          gradientTarget[wOffset + p] += delta * prev[p];
          if (layer > 0) {
            this.valueHiddenGrad[layer - 1][p] +=
              this.valueParams[wOffset + p] * delta;
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

  private buildBatch(episodes: readonly PPOTransition[][]): BatchData {
    let totalTransitions = 0;
    for (const episode of episodes) {
      totalTransitions += episode.length;
    }

    const observations: Float32Array[] = new Array(totalTransitions);
    const actions = new Int16Array(totalTransitions);
    const oldLogProbs = new Float32Array(totalTransitions);
    const advantages = new Float32Array(totalTransitions);
    const returns = new Float32Array(totalTransitions);

    let cursor = 0;
    for (const episode of episodes) {
      if (episode.length === 0) {
        continue;
      }

      const episodeAdvantages = new Float32Array(episode.length);
      let nextAdvantage = 0;
      let nextValue = 0;

      for (let t = episode.length - 1; t >= 0; t--) {
        const transition = episode[t];
        const nonTerminal = transition.done ? 0 : 1;
        const delta =
          transition.reward + PPO_GAMMA * nextValue * nonTerminal - transition.value;
        const advantage =
          delta +
          PPO_GAMMA * PPO_GAE_LAMBDA * nonTerminal * nextAdvantage;

        episodeAdvantages[t] = advantage;
        nextAdvantage = advantage;
        nextValue = transition.value;
      }

      for (let t = 0; t < episode.length; t++) {
        const transition = episode[t];
        const advantage = episodeAdvantages[t];

        observations[cursor] = transition.observation;
        actions[cursor] = transition.action;
        oldLogProbs[cursor] = transition.logProb;
        advantages[cursor] = advantage;
        returns[cursor] = transition.value + advantage;
        cursor += 1;
      }
    }

    return {
      observations,
      actions,
      oldLogProbs,
      advantages,
      returns,
      size: cursor,
    };
  }

  private normalizeAdvantages(advantages: Float32Array, size: number): void {
    if (size <= 1) {
      return;
    }

    let mean = 0;
    for (let i = 0; i < size; i++) {
      mean += advantages[i];
    }
    mean /= size;

    let variance = 0;
    for (let i = 0; i < size; i++) {
      const delta = advantages[i] - mean;
      variance += delta * delta;
    }
    variance /= size;

    const std = Math.sqrt(variance + EPSILON);
    for (let i = 0; i < size; i++) {
      advantages[i] = (advantages[i] - mean) / std;
    }
  }
}
