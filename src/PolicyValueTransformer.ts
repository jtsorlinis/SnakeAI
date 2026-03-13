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
  value: number;
};

let backendInitPromise: Promise<void> | null = null;

const STEM_CHANNELS = 24;
const TRANSFORMER_MODEL_DIM = 48;
const TRANSFORMER_HEADS = 4;
const TRANSFORMER_BLOCKS = 2;
const TRANSFORMER_FF_DIM = 96;
const HEAD_HIDDEN_DIM = 96;
const TRANSFORMER_LAYER_NORM_EPSILON = 1e-5;

type GridPositionEncodingLayerArgs = {
  gridSize: number;
  modelDim: number;
  name?: string;
};

type ScaledDotProductAttentionArgs = {
  numHeads: number;
  modelDim: number;
  name?: string;
};

type TokenPoolingLayerArgs = {
  modelDim: number;
  name?: string;
};

type HeadTokenSelectorLayerArgs = {
  tokenCount: number;
  modelDim: number;
  headChannelIndex: number;
  name?: string;
};

function isShapeList(
  shape: tf.Shape | tf.Shape[],
): shape is tf.Shape[] {
  return Array.isArray(shape) && Array.isArray(shape[0]);
}

function createAxisPositionEncoding(
  length: number,
  modelDim: number,
): Float32Array {
  const encoding = new Float32Array(length * modelDim);

  for (let position = 0; position < length; position++) {
    const rowOffset = position * modelDim;

    for (let dim = 0; dim < modelDim; dim += 2) {
      const angle = position / Math.pow(10000, dim / modelDim);
      encoding[rowOffset + dim] = Math.sin(angle);

      if (dim + 1 < modelDim) {
        encoding[rowOffset + dim + 1] = Math.cos(angle);
      }
    }
  }

  return encoding;
}

function createGridPositionEncoding(
  gridSize: number,
  modelDim: number,
): Float32Array {
  if (modelDim % 2 !== 0) {
    throw new Error(
      `Transformer modelDim (${modelDim}) must be even for 2D positional encoding.`,
    );
  }

  const axisDim = modelDim / 2;
  const rows = createAxisPositionEncoding(gridSize, axisDim);
  const columns = createAxisPositionEncoding(gridSize, axisDim);
  const encoding = new Float32Array(gridSize * gridSize * modelDim);

  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const tokenOffset = (y * gridSize + x) * modelDim;
      encoding.set(rows.subarray(y * axisDim, (y + 1) * axisDim), tokenOffset);
      encoding.set(
        columns.subarray(x * axisDim, (x + 1) * axisDim),
        tokenOffset + axisDim,
      );
    }
  }

  return encoding;
}

class GridPositionEncoding extends tf.layers.Layer {
  public static className = "GridPositionEncoding";

  private readonly gridSize: number;
  private readonly modelDim: number;
  private readonly encodingValues: Float32Array;

  constructor(args: GridPositionEncodingLayerArgs) {
    super(args);
    this.gridSize = args.gridSize;
    this.modelDim = args.modelDim;
    this.encodingValues = createGridPositionEncoding(
      this.gridSize,
      this.modelDim,
    );
  }

  public computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    return inputShape;
  }

  public call(
    inputs: tf.Tensor | tf.Tensor[],
    _kwargs: Record<string, unknown>,
  ): tf.Tensor | tf.Tensor[] {
    const input = Array.isArray(inputs) ? inputs[0] : inputs;

    return tf.tidy(() => {
      const encoding = tf.tensor3d(this.encodingValues, [
        1,
        this.gridSize * this.gridSize,
        this.modelDim,
      ]);

      return input.add(encoding);
    });
  }

  public getConfig(): tf.serialization.ConfigDict {
    return {
      ...super.getConfig(),
      gridSize: this.gridSize,
      modelDim: this.modelDim,
    };
  }
}

class TokenPooling extends tf.layers.Layer {
  public static className = "TokenPooling";

  private readonly modelDim: number;

  constructor(args: TokenPoolingLayerArgs) {
    super(args);
    this.modelDim = args.modelDim;
  }

  public computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    if (isShapeList(inputShape) || !Array.isArray(inputShape) || inputShape.length !== 3) {
      throw new Error("TokenPooling expects a single [batch, tokens, channels] shape.");
    }

    return [inputShape[0] ?? null, this.modelDim * 2];
  }

  public call(
    inputs: tf.Tensor | tf.Tensor[],
    _kwargs: Record<string, unknown>,
  ): tf.Tensor | tf.Tensor[] {
    const input = Array.isArray(inputs) ? inputs[0] : inputs;

    return tf.tidy(() => {
      const tokens = input as tf.Tensor3D;
      const mean = tf.mean(tokens, 1);
      const max = tf.max(tokens, 1);
      return tf.concat([mean, max], 1);
    });
  }

  public getConfig(): tf.serialization.ConfigDict {
    return {
      ...super.getConfig(),
      modelDim: this.modelDim,
    };
  }
}

class HeadTokenSelector extends tf.layers.Layer {
  public static className = "HeadTokenSelector";

  private readonly tokenCount: number;
  private readonly modelDim: number;
  private readonly headChannelIndex: number;

  constructor(args: HeadTokenSelectorLayerArgs) {
    super(args);
    this.tokenCount = args.tokenCount;
    this.modelDim = args.modelDim;
    this.headChannelIndex = args.headChannelIndex;
  }

  public computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    if (!isShapeList(inputShape) || inputShape.length !== 2) {
      throw new Error(
        "HeadTokenSelector expects [encodedTokensShape, observationShape].",
      );
    }

    return [inputShape[0][0] ?? null, this.modelDim];
  }

  public call(
    inputs: tf.Tensor | tf.Tensor[],
    _kwargs: Record<string, unknown>,
  ): tf.Tensor | tf.Tensor[] {
    if (!Array.isArray(inputs) || inputs.length !== 2) {
      throw new Error("HeadTokenSelector expects [encodedTokens, observation].");
    }

    const [encodedTokens, observation] = inputs as [tf.Tensor3D, tf.Tensor4D];
    const observedTokenCount = encodedTokens.shape[1];

    if (observedTokenCount == null || observedTokenCount !== this.tokenCount) {
      throw new Error(
        `HeadTokenSelector expected ${this.tokenCount} tokens, received ${observedTokenCount}.`,
      );
    }

    return tf.tidy(() => {
      const batchSize = observation.shape[0] ?? -1;
      const headMask = observation
        .slice([0, 0, 0, this.headChannelIndex], [-1, -1, -1, 1])
        .reshape([batchSize, this.tokenCount, 1]);

      return encodedTokens.mul(headMask).sum(1);
    });
  }

  public getConfig(): tf.serialization.ConfigDict {
    return {
      ...super.getConfig(),
      tokenCount: this.tokenCount,
      modelDim: this.modelDim,
      headChannelIndex: this.headChannelIndex,
    };
  }
}

class ScaledDotProductAttention extends tf.layers.Layer {
  public static className = "ScaledDotProductAttention";

  private readonly numHeads: number;
  private readonly modelDim: number;
  private readonly headDim: number;

  constructor(args: ScaledDotProductAttentionArgs) {
    super(args);

    if (args.modelDim % args.numHeads !== 0) {
      throw new Error(
        `Transformer modelDim (${args.modelDim}) must be divisible by numHeads (${args.numHeads}).`,
      );
    }

    this.numHeads = args.numHeads;
    this.modelDim = args.modelDim;
    this.headDim = args.modelDim / args.numHeads;
  }

  public computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    if (!isShapeList(inputShape) || inputShape.length !== 3) {
      throw new Error(
        "ScaledDotProductAttention expects [queryShape, keyShape, valueShape].",
      );
    }

    return inputShape[0];
  }

  public call(
    inputs: tf.Tensor | tf.Tensor[],
    _kwargs: Record<string, unknown>,
  ): tf.Tensor | tf.Tensor[] {
    if (!Array.isArray(inputs) || inputs.length !== 3) {
      throw new Error("ScaledDotProductAttention expects [query, key, value].");
    }

    const [query, key, value] = inputs as tf.Tensor3D[];
    const queryTokens = query.shape[1];
    const keyTokens = key.shape[1];
    const valueTokens = value.shape[1];

    if (queryTokens == null || keyTokens == null || valueTokens == null) {
      throw new Error("Transformer attention requires a known token dimension.");
    }

    return tf.tidy(() => {
      const batchSize = query.shape[0] ?? -1;
      const queryHeads = tf.transpose(
        query.reshape([batchSize, queryTokens, this.numHeads, this.headDim]),
        [0, 2, 1, 3],
      );
      const keyHeads = tf.transpose(
        key.reshape([batchSize, keyTokens, this.numHeads, this.headDim]),
        [0, 2, 1, 3],
      );
      const valueHeads = tf.transpose(
        value.reshape([batchSize, valueTokens, this.numHeads, this.headDim]),
        [0, 2, 1, 3],
      );

      const attentionScores = tf
        .matMul(queryHeads, keyHeads, false, true)
        .mul(1 / Math.sqrt(this.headDim));
      const attentionWeights = tf.softmax(attentionScores, -1);
      const contextHeads = tf.matMul(attentionWeights, valueHeads);
      const context = tf.transpose(contextHeads, [0, 2, 1, 3]);

      return context.reshape([batchSize, queryTokens, this.modelDim]);
    });
  }

  public getConfig(): tf.serialization.ConfigDict {
    return {
      ...super.getConfig(),
      numHeads: this.numHeads,
      modelDim: this.modelDim,
    };
  }
}

tf.serialization.registerClass(GridPositionEncoding);
tf.serialization.registerClass(ScaledDotProductAttention);
tf.serialization.registerClass(TokenPooling);
tf.serialization.registerClass(HeadTokenSelector);

function createTransformerModel(gridSize: number): tf.LayersModel {
  const tokenCount = gridSize * gridSize;
  const input = tf.input({ shape: [gridSize, gridSize, OBS_CHANNELS] });

  let trunk = tf.layers
    .conv2d({
      filters: STEM_CHANNELS,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "stem_conv1",
    })
    .apply(input) as tf.SymbolicTensor;

  trunk = tf.layers
    .conv2d({
      filters: STEM_CHANNELS,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "stem_conv2",
    })
    .apply(trunk) as tf.SymbolicTensor;

  trunk = tf.layers
    .conv2d({
      filters: TRANSFORMER_MODEL_DIM,
      kernelSize: 1,
      padding: "same",
      activation: "linear",
      kernelInitializer: "heNormal",
      name: "stem_projection",
    })
    .apply(trunk) as tf.SymbolicTensor;

  trunk = tf.layers
    .reshape({
      targetShape: [tokenCount, TRANSFORMER_MODEL_DIM],
      name: "board_tokens",
    })
    .apply(trunk) as tf.SymbolicTensor;

  trunk = new GridPositionEncoding({
    gridSize,
    modelDim: TRANSFORMER_MODEL_DIM,
    name: "position_encoding",
  }).apply(trunk) as tf.SymbolicTensor;

  for (let block = 0; block < TRANSFORMER_BLOCKS; block++) {
    const prefix = `transformer_${block + 1}`;

    const attentionNorm = tf.layers
      .layerNormalization({
        axis: -1,
        epsilon: TRANSFORMER_LAYER_NORM_EPSILON,
        name: `${prefix}_attn_norm`,
      })
      .apply(trunk) as tf.SymbolicTensor;

    const query = tf.layers
      .dense({
        units: TRANSFORMER_MODEL_DIM,
        activation: "linear",
        useBias: false,
        kernelInitializer: "glorotUniform",
        name: `${prefix}_query`,
      })
      .apply(attentionNorm) as tf.SymbolicTensor;

    const key = tf.layers
      .dense({
        units: TRANSFORMER_MODEL_DIM,
        activation: "linear",
        useBias: false,
        kernelInitializer: "glorotUniform",
        name: `${prefix}_key`,
      })
      .apply(attentionNorm) as tf.SymbolicTensor;

    const value = tf.layers
      .dense({
        units: TRANSFORMER_MODEL_DIM,
        activation: "linear",
        useBias: false,
        kernelInitializer: "glorotUniform",
        name: `${prefix}_value`,
      })
      .apply(attentionNorm) as tf.SymbolicTensor;

    const attention = new ScaledDotProductAttention({
      numHeads: TRANSFORMER_HEADS,
      modelDim: TRANSFORMER_MODEL_DIM,
      name: `${prefix}_attention`,
    }).apply([query, key, value]) as tf.SymbolicTensor;

    const attentionProjection = tf.layers
      .dense({
        units: TRANSFORMER_MODEL_DIM,
        activation: "linear",
        kernelInitializer: "glorotUniform",
        name: `${prefix}_attn_project`,
      })
      .apply(attention) as tf.SymbolicTensor;

    trunk = tf.layers
      .add({ name: `${prefix}_attn_residual` })
      .apply([trunk, attentionProjection]) as tf.SymbolicTensor;

    const feedForwardNorm = tf.layers
      .layerNormalization({
        axis: -1,
        epsilon: TRANSFORMER_LAYER_NORM_EPSILON,
        name: `${prefix}_ff_norm`,
      })
      .apply(trunk) as tf.SymbolicTensor;

    let feedForward = tf.layers
      .dense({
        units: TRANSFORMER_FF_DIM,
        activation: "relu",
        kernelInitializer: "heNormal",
        name: `${prefix}_ff_expand`,
      })
      .apply(feedForwardNorm) as tf.SymbolicTensor;

    feedForward = tf.layers
      .dense({
        units: TRANSFORMER_MODEL_DIM,
        activation: "linear",
        kernelInitializer: "heNormal",
        name: `${prefix}_ff_project`,
      })
      .apply(feedForward) as tf.SymbolicTensor;

    trunk = tf.layers
      .add({ name: `${prefix}_ff_residual` })
      .apply([trunk, feedForward]) as tf.SymbolicTensor;
  }

  const encoded = tf.layers
    .layerNormalization({
      axis: -1,
      epsilon: TRANSFORMER_LAYER_NORM_EPSILON,
      name: "transformer_output_norm",
    })
    .apply(trunk) as tf.SymbolicTensor;

  const pooled = new TokenPooling({
    modelDim: TRANSFORMER_MODEL_DIM,
    name: "token_pooling",
  }).apply(encoded) as tf.SymbolicTensor;

  const pooledFeatures = tf.layers
    .layerNormalization({
      axis: -1,
      epsilon: TRANSFORMER_LAYER_NORM_EPSILON,
      name: "pooled_norm",
    })
    .apply(pooled) as tf.SymbolicTensor;

  const headToken = new HeadTokenSelector({
    tokenCount,
    modelDim: TRANSFORMER_MODEL_DIM,
    headChannelIndex: 1,
    name: "head_token_selector",
  }).apply([encoded, input]) as tf.SymbolicTensor;

  const headFeatures = tf.layers
    .layerNormalization({
      axis: -1,
      epsilon: TRANSFORMER_LAYER_NORM_EPSILON,
      name: "head_token_norm",
    })
    .apply(headToken) as tf.SymbolicTensor;

  const summaryFeatures = tf.layers
    .concatenate({
      axis: -1,
      name: "summary_features",
    })
    .apply([pooledFeatures, headFeatures]) as tf.SymbolicTensor;

  const policyHidden = tf.layers
    .dense({
      units: HEAD_HIDDEN_DIM,
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "policy_hidden",
    })
    .apply(summaryFeatures) as tf.SymbolicTensor;

  const policyLogits = tf.layers
    .dense({
      units: OUTPUTS,
      activation: "linear",
      kernelInitializer: "heNormal",
      name: "policy_logits",
    })
    .apply(policyHidden) as tf.SymbolicTensor;

  const valueHidden = tf.layers
    .dense({
      units: HEAD_HIDDEN_DIM,
      activation: "relu",
      kernelInitializer: "heNormal",
      name: "value_hidden",
    })
    .apply(summaryFeatures) as tf.SymbolicTensor;

  const value = tf.layers
    .dense({
      units: 1,
      activation: "linear",
      kernelInitializer: "heNormal",
      name: "value",
    })
    .apply(valueHidden) as tf.SymbolicTensor;

  return tf.model({
    inputs: input,
    outputs: [policyLogits, value],
    name: "snake_transformers_policy_value",
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

export class PolicyValueTransformer {
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
    this.model = createTransformerModel(this.width);
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
    const prediction = this.predict(input, this.singlePolicyScratch);
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
  ): PolicyValuePrediction {
    this.writeObservationToNhwc(input, this.singleInputScratch, 0);
    const logitsTarget = this.singleLogitsScratch;

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
