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
import type { Transition } from "./types";

const KERNEL_SIZE = 3;
const CONV1_FILTERS = 12;
const CONV2_FILTERS = 24;
const HIDDEN_UNITS = 96;

let backendInitPromise: Promise<void> | null = null;

export type TrainBatchResult = {
  loss: number;
  tdErrors: Float32Array;
};

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

function createModel(gridSize: number): tf.LayersModel {
  const input = tf.input({ shape: [gridSize, gridSize, OBS_CHANNELS] });

  const conv1 = tf.layers
    .conv2d({
      filters: CONV1_FILTERS,
      kernelSize: KERNEL_SIZE,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
    })
    .apply(input) as tf.SymbolicTensor;

  const conv2 = tf.layers
    .conv2d({
      filters: CONV2_FILTERS,
      kernelSize: KERNEL_SIZE,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
    })
    .apply(conv1) as tf.SymbolicTensor;

  const flat = tf.layers.flatten().apply(conv2) as tf.SymbolicTensor;

  const valueHidden = tf.layers
    .dense({
      units: HIDDEN_UNITS,
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
    })
    .apply(flat) as tf.SymbolicTensor;

  const advantageHidden = tf.layers
    .dense({
      units: HIDDEN_UNITS,
      activation: "relu",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
    })
    .apply(flat) as tf.SymbolicTensor;

  const value = tf.layers
    .dense({
      units: 1,
      activation: "linear",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
    })
    .apply(valueHidden) as tf.SymbolicTensor;

  const advantage = tf.layers
    .dense({
      units: OUTPUTS,
      activation: "linear",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
    })
    .apply(advantageHidden) as tf.SymbolicTensor;

  return tf.model({
    inputs: input,
    outputs: [value, advantage],
  });
}

function combineDuelingHeads(
  value: tf.Tensor2D,
  advantage: tf.Tensor2D,
): tf.Tensor2D {
  const centeredAdvantage = advantage.sub(advantage.mean(1, true));
  return value.add(centeredAdvantage);
}

export class ConvDQN {
  private readonly width: number;
  private readonly area: number;
  private readonly model: tf.LayersModel;
  private readonly optimizer: tf.Optimizer;
  private readonly singleInputScratch: Float32Array;

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
  }

  public copyWeightsFrom(other: ConvDQN): void {
    tf.tidy(() => {
      const sourceWeights = other.model.getWeights();
      const copiedWeights = sourceWeights.map((weight) => weight.clone());
      this.model.setWeights(copiedWeights);
    });
  }

  public predict(
    input: ArrayLike<number>,
    target = new Float32Array(OUTPUTS),
  ): Float32Array {
    this.writeObservationToNhwc(input, this.singleInputScratch, 0);

    tf.tidy(() => {
      const state = tf.tensor4d(this.singleInputScratch, [
        1,
        this.width,
        this.width,
        OBS_CHANNELS,
      ]);
      const qTensor = this.predictQ(state, false);
      target.set(qTensor.dataSync());
    });

    return target;
  }

  public trainBatch(
    batch: readonly Transition[],
    importanceWeights: Float32Array,
    targetNetwork: ConvDQN,
    bootstrapDiscount: number,
  ): TrainBatchResult {
    if (batch.length === 0) {
      return { loss: 0, tdErrors: new Float32Array(0) };
    }

    const batchSize = batch.length;
    const channelSize = this.area * OBS_CHANNELS;
    const statesData = new Float32Array(batchSize * channelSize);
    const nextStatesData = new Float32Array(batchSize * channelSize);
    const actionsData = new Int32Array(batchSize);
    const rewardsData = new Float32Array(batchSize);
    const donesData = new Float32Array(batchSize);
    const weightsData = new Float32Array(batchSize);

    for (let i = 0; i < batchSize; i++) {
      const transition = batch[i];
      const offset = i * channelSize;

      this.writeObservationToNhwc(transition.state, statesData, offset);
      this.writeObservationToNhwc(transition.nextState, nextStatesData, offset);

      actionsData[i] = transition.action;
      rewardsData[i] = transition.reward;
      donesData[i] = transition.done ? 1 : 0;
      weightsData[i] = importanceWeights[i] ?? 1;
    }

    return tf.tidy(() => {
      const shape: [number, number, number, number] = [
        batchSize,
        this.width,
        this.width,
        OBS_CHANNELS,
      ];

      const statesTensor = tf.tensor4d(statesData, shape);
      const nextStatesTensor = tf.tensor4d(nextStatesData, shape);
      const actionsTensor = tf.tensor1d(actionsData, "int32");
      const rewardsTensor = tf.tensor1d(rewardsData, "float32");
      const donesTensor = tf.tensor1d(donesData, "float32");
      const weightsTensor = tf.tensor1d(weightsData, "float32");
      const discountTensor = tf.scalar(bootstrapDiscount, "float32");

      const nextOnlineQ = this.predictQ(nextStatesTensor, false);
      const bestNextActions = nextOnlineQ.argMax(1);
      const nextTargetQ = targetNetwork.predictQ(nextStatesTensor, false);
      const nextActionMask = tf
        .oneHot(bestNextActions, OUTPUTS)
        .asType("float32");
      const nextChosenQ = nextTargetQ.mul(nextActionMask).sum(1);
      const targetValues = rewardsTensor.add(
        tf.onesLike(donesTensor)
          .sub(donesTensor)
          .mul(discountTensor)
          .mul(nextChosenQ),
      );

      const currentQ = this.predictQ(statesTensor, false);
      const actionMask = tf.oneHot(actionsTensor, OUTPUTS).asType("float32");
      const predictedQ = currentQ.mul(actionMask).sum(1);
      const tdError = targetValues.sub(predictedQ);
      const tdErrorsArray = tdError.abs().dataSync();

      const lossTensor = this.optimizer.minimize(() => {
        const qValues = this.predictQ(statesTensor, true);
        const predicted = qValues.mul(actionMask).sum(1);
        const error = predicted.sub(targetValues);
        const absError = error.abs();
        const quadratic = absError.clipByValue(0, 1);
        const linear = absError.sub(quadratic);
        const huber = quadratic.square().mul(0.5).add(linear);
        return huber.mul(weightsTensor).mean();
      }, true);

      const loss = lossTensor ? lossTensor.dataSync()[0] : 0;
      return {
        loss,
        tdErrors: new Float32Array(tdErrorsArray),
      };
    });
  }

  public dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }

  private predictQ(input: tf.Tensor4D, training: boolean): tf.Tensor2D {
    const outputs = this.model.apply(input, { training });
    if (!Array.isArray(outputs) || outputs.length !== 2) {
      throw new Error("Dueling model must output [value, advantage]");
    }

    const value = outputs[0] as tf.Tensor2D;
    const advantage = outputs[1] as tf.Tensor2D;
    return combineDuelingHeads(value, advantage);
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
