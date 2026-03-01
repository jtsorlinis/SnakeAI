import * as tf from "@tensorflow/tfjs";
import { GRID_SIZE, OBS_CHANNELS, OUTPUTS } from "./config";

export type PolicyAction = {
  action: number;
};

export type PolicyPrediction = {
  policy: Float32Array;
  logits: Float32Array;
};

type WeightSpec = {
  shape: number[];
  size: number;
};

let backendInitPromise: Promise<void> | null = null;

function createPolicyModel(gridSize: number): tf.LayersModel {
  const input = tf.input({ shape: [gridSize, gridSize, OBS_CHANNELS] });

  let trunk = tf.layers
    .conv2d({
      filters: 32,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "conv1",
    })
    .apply(input) as tf.SymbolicTensor;

  trunk = tf.layers
    .conv2d({
      filters: 32,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "conv2",
    })
    .apply(trunk) as tf.SymbolicTensor;

  trunk = tf.layers
    .conv2d({
      filters: 32,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "conv3",
    })
    .apply(trunk) as tf.SymbolicTensor;

  const policyFeatures = tf.layers
    .conv2d({
      filters: 8,
      kernelSize: 1,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "policy_conv",
    })
    .apply(trunk) as tf.SymbolicTensor;

  const policyFlat = tf.layers
    .flatten({ name: "policy_flatten" })
    .apply(policyFeatures) as tf.SymbolicTensor;

  const policyHidden = tf.layers
    .dense({
      units: 64,
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "policy_hidden",
    })
    .apply(policyFlat) as tf.SymbolicTensor;

  const policyLogits = tf.layers
    .dense({
      units: OUTPUTS,
      activation: "linear",
      kernelInitializer: "heNormal",
      name: "policy_logits",
    })
    .apply(policyHidden) as tf.SymbolicTensor;

  return tf.model({
    inputs: input,
    outputs: policyLogits,
    name: "snake_policy_es",
  });
}

function softmax(
  logits: ArrayLike<number>,
  target: Float32Array,
): Float32Array {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.length; i++) {
    max = Math.max(max, logits[i]);
  }

  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp(logits[i] - max);
    target[i] = value;
    sum += value;
  }

  if (sum <= 0) {
    const uniform = 1 / target.length;
    for (let i = 0; i < target.length; i++) {
      target[i] = uniform;
    }
    return target;
  }

  const inv = 1 / sum;
  for (let i = 0; i < target.length; i++) {
    target[i] *= inv;
  }

  return target;
}

function sampleFromDistribution(probabilities: ArrayLike<number>): number {
  const threshold = Math.random();
  let cumulative = 0;

  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (threshold <= cumulative) {
      return i;
    }
  }

  return probabilities.length - 1;
}

function argMax(values: ArrayLike<number>): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function argMaxFromOffset(
  values: ArrayLike<number>,
  offset: number,
  length: number,
): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < length; i++) {
    const value = values[offset + i];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }

  return bestIndex;
}

export async function ensureTfjsBackend(): Promise<void> {
  if (!backendInitPromise) {
    backendInitPromise = (async () => {
      let usingWebgl = false;

      try {
        usingWebgl = await tf.setBackend("webgl");
      } catch (error) {
        console.warn(
          "Failed to set tfjs backend to WebGL, falling back to CPU.",
          error,
        );
      }

      if (!usingWebgl) {
        await tf.setBackend("cpu");
      }

      await tf.ready();
    })();
  }

  await backendInitPromise;
}

export class ConvDQN {
  private readonly width: number;
  private readonly area: number;
  private readonly model: tf.LayersModel;
  private readonly singleInputScratch: Float32Array;
  private readonly singlePolicyScratch: Float32Array;
  private readonly singleLogitsScratch: Float32Array;
  private readonly weightSpecs: WeightSpec[];
  private readonly totalWeightCount: number;

  constructor(gridSize = GRID_SIZE) {
    this.width = gridSize;
    this.area = this.width * this.width;
    this.model = createPolicyModel(this.width);
    this.singleInputScratch = new Float32Array(this.area * OBS_CHANNELS);
    this.singlePolicyScratch = new Float32Array(OUTPUTS);
    this.singleLogitsScratch = new Float32Array(OUTPUTS);

    const initialWeights = this.model.getWeights();
    this.weightSpecs = initialWeights.map((tensor) => ({
      shape: [...tensor.shape],
      size: tensor.size,
    }));
    this.totalWeightCount = initialWeights.reduce(
      (sum, tensor) => sum + tensor.size,
      0,
    );
  }

  public parameterCount(): number {
    return this.totalWeightCount;
  }

  public act(input: ArrayLike<number>, sample = true): PolicyAction {
    const prediction = this.predict(
      input,
      this.singlePolicyScratch,
      this.singleLogitsScratch,
    );

    return {
      action: sample
        ? sampleFromDistribution(prediction.policy)
        : argMax(prediction.policy),
    };
  }

  public actBatch(
    observations: readonly ArrayLike<number>[],
    sample = true,
    target: Int32Array = new Int32Array(observations.length),
  ): Int32Array {
    const batchSize = observations.length;
    let actions = target;

    if (actions.length !== batchSize) {
      actions = new Int32Array(batchSize);
    }

    if (batchSize === 0) {
      return actions;
    }

    const channelSize = this.area * OBS_CHANNELS;
    const statesData = new Float32Array(batchSize * channelSize);

    for (let i = 0; i < batchSize; i++) {
      this.writeObservationToNhwc(observations[i], statesData, i * channelSize);
    }

    tf.tidy(() => {
      const statesTensor = tf.tensor4d(statesData, [
        batchSize,
        this.width,
        this.width,
        OBS_CHANNELS,
      ]);
      const policyLogits = this.predictPolicyLogits(statesTensor, false);
      const probabilities = tf.softmax(policyLogits).dataSync();

      for (let i = 0; i < batchSize; i++) {
        const base = i * OUTPUTS;
        actions[i] = sample
          ? sampleFromDistribution(probabilities.subarray(base, base + OUTPUTS))
          : argMaxFromOffset(probabilities, base, OUTPUTS);
      }
    });

    return actions;
  }

  public predict(
    input: ArrayLike<number>,
    policyTarget: Float32Array = new Float32Array(OUTPUTS),
    logitsTarget: Float32Array = new Float32Array(OUTPUTS),
  ): PolicyPrediction {
    this.writeObservationToNhwc(input, this.singleInputScratch, 0);

    tf.tidy(() => {
      const state = tf.tensor4d(this.singleInputScratch, [
        1,
        this.width,
        this.width,
        OBS_CHANNELS,
      ]);
      const policyLogits = this.predictPolicyLogits(state, false);
      logitsTarget.set(policyLogits.dataSync());
    });

    softmax(logitsTarget, policyTarget);

    return {
      policy: policyTarget,
      logits: logitsTarget,
    };
  }

  public exportWeightsFlat(
    target: Float32Array = new Float32Array(this.totalWeightCount),
  ): Float32Array {
    if (target.length !== this.totalWeightCount) {
      throw new Error(
        `Expected flat weight buffer of length ${this.totalWeightCount}, got ${target.length}.`,
      );
    }

    const weights = this.model.getWeights();
    let offset = 0;

    for (const tensor of weights) {
      const values = tensor.dataSync();
      target.set(values, offset);
      offset += values.length;
    }

    return target;
  }

  public importWeightsFlat(flatWeights: ArrayLike<number>): void {
    if (flatWeights.length !== this.totalWeightCount) {
      throw new Error(
        `Expected ${this.totalWeightCount} flat weights, got ${flatWeights.length}.`,
      );
    }

    const tensors: tf.Tensor[] = [];
    let offset = 0;

    try {
      for (const spec of this.weightSpecs) {
        const end = offset + spec.size;
        let values: Float32Array;
        if (flatWeights instanceof Float32Array) {
          values = flatWeights.subarray(offset, end);
        } else {
          values = new Float32Array(spec.size);
          for (let i = 0; i < spec.size; i++) {
            values[i] = flatWeights[offset + i];
          }
        }

        tensors.push(tf.tensor(values, spec.shape, "float32"));
        offset = end;
      }

      this.model.setWeights(tensors);
    } finally {
      for (const tensor of tensors) {
        tensor.dispose();
      }
    }
  }

  public addScaledNoise(
    baseWeights: ArrayLike<number>,
    noise: ArrayLike<number>,
    noiseScale: number,
    target: Float32Array,
  ): Float32Array {
    if (
      baseWeights.length !== this.totalWeightCount ||
      noise.length !== this.totalWeightCount ||
      target.length !== this.totalWeightCount
    ) {
      throw new Error("Weight/noise buffer lengths must match parameter count.");
    }

    for (let i = 0; i < this.totalWeightCount; i++) {
      target[i] = baseWeights[i] + noise[i] * noiseScale;
    }

    return target;
  }

  public dispose(): void {
    this.model.dispose();
  }

  public async saveToIndexedDb(modelKey: string): Promise<void> {
    await this.model.save(`indexeddb://${modelKey}`);
  }

  public async loadFromIndexedDb(modelKey: string): Promise<boolean> {
    let loadedModel: tf.LayersModel | null = null;

    try {
      loadedModel = await tf.loadLayersModel(`indexeddb://${modelKey}`);
      this.model.setWeights(loadedModel.getWeights());
      return true;
    } catch (error) {
      console.warn("Failed to load model checkpoint from IndexedDB.", error);
      return false;
    } finally {
      loadedModel?.dispose();
    }
  }

  private predictPolicyLogits(input: tf.Tensor4D, training: boolean): tf.Tensor2D {
    const output = this.model.apply(input, { training });

    if (Array.isArray(output)) {
      throw new Error("Policy model must output a single logits tensor.");
    }

    return output as tf.Tensor2D;
  }

  private writeObservationToNhwc(
    input: ArrayLike<number>,
    target: Float32Array,
    targetOffset: number,
  ): void {
    let writeIndex = targetOffset;

    for (let y = 0; y < this.width; y++) {
      for (let x = 0; x < this.width; x++) {
        for (let c = 0; c < OBS_CHANNELS; c++) {
          const sourceIndex = c * this.area + y * this.width + x;
          target[writeIndex] = input[sourceIndex];
          writeIndex += 1;
        }
      }
    }
  }
}

export { argMax };
