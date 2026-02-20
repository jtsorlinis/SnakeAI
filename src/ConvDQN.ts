import * as tf from "@tensorflow/tfjs";
import {
  ADAM_BETA1,
  ADAM_BETA2,
  ADAM_EPSILON,
  GRID_SIZE,
  LEARNING_RATE,
  OBS_CHANNELS,
  OUTPUTS,
} from "./config";

const KERNEL_SIZE = 3;
const STEM_FILTERS = 32;
const TRUNK_FILTERS = 64;
const RESIDUAL_BLOCKS = 3;
const HEAD_HIDDEN_UNITS = 64;
const LOG_EPSILON = 1e-8;

type PPOTrainInputs = {
  observations: readonly ArrayLike<number>[];
  actions: Int32Array;
  oldLogProbs: Float32Array;
  advantages: Float32Array;
  returns: Float32Array;
  clipRange: number;
  valueLossCoef: number;
  entropyCoef: number;
};

export type PPOTrainResult = {
  totalLoss: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  approxKl: number;
  clipFraction: number;
};

export type PolicyAction = {
  action: number;
  logProb: number;
  value: number;
};

export type PolicyValuePrediction = {
  policy: Float32Array;
  logits: Float32Array;
  value: number;
};

let backendInitPromise: Promise<void> | null = null;

function residualBlock(
  input: tf.SymbolicTensor,
  filters: number,
  blockIndex: number,
): tf.SymbolicTensor {
  const conv1 = tf.layers
    .conv2d({
      filters,
      kernelSize: KERNEL_SIZE,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: `res_${blockIndex}_conv1`,
    })
    .apply(input) as tf.SymbolicTensor;

  const conv2 = tf.layers
    .conv2d({
      filters,
      kernelSize: KERNEL_SIZE,
      padding: "same",
      activation: "linear",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: `res_${blockIndex}_conv2`,
    })
    .apply(conv1) as tf.SymbolicTensor;

  const added = tf.layers
    .add({ name: `res_${blockIndex}_add` })
    .apply([input, conv2]) as tf.SymbolicTensor;

  return tf.layers
    .activation({ activation: "relu", name: `res_${blockIndex}_relu` })
    .apply(added) as tf.SymbolicTensor;
}

function createModel(gridSize: number): tf.LayersModel {
  const input = tf.input({ shape: [gridSize, gridSize, OBS_CHANNELS] });

  let trunk = tf.layers
    .conv2d({
      filters: STEM_FILTERS,
      kernelSize: KERNEL_SIZE,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: "stem_conv",
    })
    .apply(input) as tf.SymbolicTensor;

  for (let i = 0; i < RESIDUAL_BLOCKS; i++) {
    trunk = residualBlock(trunk, STEM_FILTERS, i);
  }

  trunk = tf.layers
    .conv2d({
      filters: TRUNK_FILTERS,
      kernelSize: KERNEL_SIZE,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: "trunk_conv",
    })
    .apply(trunk) as tf.SymbolicTensor;

  const flattened = tf.layers
    .flatten({ name: "flatten_features" })
    .apply(trunk) as tf.SymbolicTensor;

  const policyHidden = tf.layers
    .dense({
      units: HEAD_HIDDEN_UNITS,
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: "policy_hidden",
    })
    .apply(flattened) as tf.SymbolicTensor;

  const valueHidden = tf.layers
    .dense({
      units: HEAD_HIDDEN_UNITS,
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: "value_hidden",
    })
    .apply(flattened) as tf.SymbolicTensor;

  const policyLogits = tf.layers
    .dense({
      units: OUTPUTS,
      activation: "linear",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: "policy_logits",
    })
    .apply(policyHidden) as tf.SymbolicTensor;

  const value = tf.layers
    .dense({
      units: 1,
      activation: "linear",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: "value",
    })
    .apply(valueHidden) as tf.SymbolicTensor;

  return tf.model({
    inputs: input,
    outputs: [policyLogits, value],
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
  private readonly optimizer: tf.Optimizer;
  private readonly singleInputScratch: Float32Array;
  private readonly singlePolicyScratch: Float32Array;
  private readonly singleLogitsScratch: Float32Array;

  constructor(gridSize = GRID_SIZE) {
    this.width = gridSize;
    this.area = this.width * this.width;
    this.model = createModel(this.width);
    this.optimizer = tf.train.adam(
      LEARNING_RATE,
      ADAM_BETA1,
      ADAM_BETA2,
      ADAM_EPSILON,
    );
    this.singleInputScratch = new Float32Array(this.area * OBS_CHANNELS);
    this.singlePolicyScratch = new Float32Array(OUTPUTS);
    this.singleLogitsScratch = new Float32Array(OUTPUTS);
  }

  public act(input: ArrayLike<number>, sample = true): PolicyAction {
    const prediction = this.predict(
      input,
      this.singlePolicyScratch,
      this.singleLogitsScratch,
    );
    const action = sample
      ? sampleFromDistribution(prediction.policy)
      : argMax(prediction.policy);

    return {
      action,
      logProb: Math.log(Math.max(LOG_EPSILON, prediction.policy[action])),
      value: prediction.value,
    };
  }

  public predict(
    input: ArrayLike<number>,
    policyTarget: Float32Array = new Float32Array(OUTPUTS),
    logitsTarget: Float32Array = new Float32Array(OUTPUTS),
  ): PolicyValuePrediction {
    this.writeObservationToNhwc(input, this.singleInputScratch, 0);

    let value = 0;
    tf.tidy(() => {
      const state = tf.tensor4d(this.singleInputScratch, [
        1,
        this.width,
        this.width,
        OBS_CHANNELS,
      ]);
      const [policyLogits, valueTensor] = this.predictPolicyAndValue(
        state,
        false,
      );
      logitsTarget.set(policyLogits.dataSync());
      value = valueTensor.dataSync()[0];
    });

    softmax(logitsTarget, policyTarget);

    return {
      policy: policyTarget,
      logits: logitsTarget,
      value,
    };
  }

  public trainBatch(inputs: PPOTrainInputs): PPOTrainResult {
    const batchSize = inputs.observations.length;
    if (
      batchSize === 0 ||
      inputs.actions.length !== batchSize ||
      inputs.oldLogProbs.length !== batchSize ||
      inputs.advantages.length !== batchSize ||
      inputs.returns.length !== batchSize
    ) {
      return {
        totalLoss: 0,
        policyLoss: 0,
        valueLoss: 0,
        entropy: 0,
        approxKl: 0,
        clipFraction: 0,
      };
    }

    const channelSize = this.area * OBS_CHANNELS;
    const statesData = new Float32Array(batchSize * channelSize);

    for (let i = 0; i < batchSize; i++) {
      this.writeObservationToNhwc(
        inputs.observations[i],
        statesData,
        i * channelSize,
      );
    }

    return tf.tidy(() => {
      const statesTensor = tf.tensor4d(statesData, [
        batchSize,
        this.width,
        this.width,
        OBS_CHANNELS,
      ]);
      const actionsTensor = tf.tensor1d(inputs.actions, "int32");
      const oldLogProbsTensor = tf.tensor1d(inputs.oldLogProbs, "float32");
      const advantagesTensor = tf.tensor1d(inputs.advantages, "float32");
      const returnsTensor = tf.tensor1d(inputs.returns, "float32");

      const optimizedLoss = this.optimizer.minimize(() => {
        const [policyLogits, values] = this.predictPolicyAndValue(
          statesTensor,
          true,
        );
        const terms = this.computeLossTerms(
          policyLogits,
          values,
          actionsTensor,
          oldLogProbsTensor,
          advantagesTensor,
          returnsTensor,
          inputs.clipRange,
          inputs.valueLossCoef,
          inputs.entropyCoef,
        );
        return terms.totalLoss;
      }, true);

      const [policyLogits, values] = this.predictPolicyAndValue(
        statesTensor,
        false,
      );
      const terms = this.computeLossTerms(
        policyLogits,
        values,
        actionsTensor,
        oldLogProbsTensor,
        advantagesTensor,
        returnsTensor,
        inputs.clipRange,
        inputs.valueLossCoef,
        inputs.entropyCoef,
      );

      return {
        totalLoss: optimizedLoss
          ? optimizedLoss.dataSync()[0]
          : terms.totalLoss.dataSync()[0],
        policyLoss: terms.policyLoss.dataSync()[0],
        valueLoss: terms.valueLoss.dataSync()[0],
        entropy: terms.entropy.dataSync()[0],
        approxKl: terms.approxKl.dataSync()[0],
        clipFraction: terms.clipFraction.dataSync()[0],
      };
    });
  }

  public dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }

  private predictPolicyAndValue(
    input: tf.Tensor4D,
    training: boolean,
  ): [tf.Tensor2D, tf.Tensor2D] {
    const outputs = this.model.apply(input, { training });

    if (!Array.isArray(outputs) || outputs.length !== 2) {
      throw new Error("Actor-critic model must output [policyLogits, value]");
    }

    return [outputs[0] as tf.Tensor2D, outputs[1] as tf.Tensor2D];
  }

  private computeLossTerms(
    policyLogits: tf.Tensor2D,
    values: tf.Tensor2D,
    actions: tf.Tensor1D,
    oldLogProbs: tf.Tensor1D,
    advantages: tf.Tensor1D,
    returns: tf.Tensor1D,
    clipRange: number,
    valueLossCoef: number,
    entropyCoef: number,
  ): {
    totalLoss: tf.Scalar;
    policyLoss: tf.Scalar;
    valueLoss: tf.Scalar;
    entropy: tf.Scalar;
    approxKl: tf.Scalar;
    clipFraction: tf.Scalar;
  } {
    const actionMask = tf.oneHot(actions, OUTPUTS).asType("float32");
    const logProbAll = tf.logSoftmax(policyLogits);
    const selectedLogProbs = logProbAll.mul(actionMask).sum(1);

    const ratio = selectedLogProbs.sub(oldLogProbs).exp();
    const unclipped = ratio.mul(advantages);
    const clipped = ratio
      .clipByValue(1 - clipRange, 1 + clipRange)
      .mul(advantages);

    const policyLoss = tf.neg(
      tf.minimum(unclipped, clipped).mean(),
    ) as tf.Scalar;

    const valuePredictions = values.squeeze([1]);
    const valueLoss = returns
      .sub(valuePredictions)
      .square()
      .mean()
      .mul(0.5) as tf.Scalar;

    const probabilities = tf.softmax(policyLogits);
    const entropy = probabilities
      .mul(logProbAll)
      .sum(1)
      .neg()
      .mean() as tf.Scalar;

    const approxKl = oldLogProbs.sub(selectedLogProbs).mean() as tf.Scalar;
    const clipFraction = ratio
      .sub(tf.scalar(1))
      .abs()
      .greater(tf.scalar(clipRange))
      .asType("float32")
      .mean() as tf.Scalar;

    const totalLoss = policyLoss
      .add(valueLoss.mul(valueLossCoef))
      .sub(entropy.mul(entropyCoef)) as tf.Scalar;

    return {
      totalLoss,
      policyLoss,
      valueLoss,
      entropy,
      approxKl,
      clipFraction,
    };
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
