import { CRITIC_HIDDEN_UNITS, CRITIC_INPUTS } from "./config";

// Critic used by the PPO optimizer. It is independent of the
// policy network so it can be wider and read extra context (snake length,
// hunger) that the policy never sees.
export class ValueNetwork {
  public readonly params: Float32Array;

  private readonly layerSizes: number[];
  private readonly weightOffsets: number[] = [];
  private readonly biasOffsets: number[] = [];
  private readonly outputWeightOffset: number;
  private readonly outputBiasOffset: number;
  private readonly hidden: Float32Array[];
  private readonly hiddenGrad: Float32Array[];

  constructor(
    private readonly inputSize = CRITIC_INPUTS,
    hiddenUnits: readonly number[] = CRITIC_HIDDEN_UNITS,
  ) {
    this.layerSizes = [...hiddenUnits];

    let offset = 0;
    let prevSize = inputSize;
    for (const size of this.layerSizes) {
      this.weightOffsets.push(offset);
      offset += prevSize * size;
      this.biasOffsets.push(offset);
      offset += size;
      prevSize = size;
    }
    this.outputWeightOffset = offset;
    offset += prevSize;
    this.outputBiasOffset = offset;
    offset += 1;

    this.params = new Float32Array(offset);
    this.hidden = this.layerSizes.map((size) => new Float32Array(size));
    this.hiddenGrad = this.layerSizes.map((size) => new Float32Array(size));
  }

  public get paramCount(): number {
    return this.params.length;
  }

  public randomize(): void {
    // Scaled uniform init so wide layers do not saturate tanh.
    let prevSize = this.inputSize;
    for (let layer = 0; layer < this.layerSizes.length; layer++) {
      const size = this.layerSizes[layer];
      const limit = Math.sqrt(3 / prevSize);
      const start = this.weightOffsets[layer];
      for (let i = 0; i < prevSize * size; i++) {
        this.params[start + i] = (Math.random() * 2 - 1) * limit;
      }
      this.params.fill(0, this.biasOffsets[layer], this.biasOffsets[layer] + size);
      prevSize = size;
    }
    this.params.fill(0, this.outputWeightOffset, this.params.length);
  }

  // Runs the critic and caches activations for a following backward() call.
  public forward(input: Float32Array): number {
    let prev = input;
    let prevSize = this.inputSize;

    for (let layer = 0; layer < this.layerSizes.length; layer++) {
      const size = this.layerSizes[layer];
      const current = this.hidden[layer];
      const weights = this.weightOffsets[layer];
      const biases = this.biasOffsets[layer];

      for (let h = 0; h < size; h++) {
        let sum = this.params[biases + h];
        const row = weights + h * prevSize;
        for (let p = 0; p < prevSize; p++) {
          sum += this.params[row + p] * prev[p];
        }
        current[h] = Math.tanh(sum);
      }

      prev = current;
      prevSize = size;
    }

    let value = this.params[this.outputBiasOffset];
    for (let i = 0; i < prevSize; i++) {
      value += this.params[this.outputWeightOffset + i] * prev[i];
    }
    return value;
  }

  // Accumulates dLoss/dParams into gradient for the input last passed to forward().
  public backward(
    input: Float32Array,
    outputGradient: number,
    gradient: Float32Array,
  ): void {
    const layerCount = this.layerSizes.length;
    gradient[this.outputBiasOffset] += outputGradient;

    if (layerCount === 0) {
      for (let i = 0; i < this.inputSize; i++) {
        gradient[this.outputWeightOffset + i] += outputGradient * input[i];
      }
      return;
    }

    for (const grad of this.hiddenGrad) {
      grad.fill(0);
    }

    const last = layerCount - 1;
    const lastHidden = this.hidden[last];
    const lastGrad = this.hiddenGrad[last];
    for (let h = 0; h < this.layerSizes[last]; h++) {
      gradient[this.outputWeightOffset + h] += outputGradient * lastHidden[h];
      lastGrad[h] = this.params[this.outputWeightOffset + h] * outputGradient;
    }

    for (let layer = last; layer >= 0; layer--) {
      const size = this.layerSizes[layer];
      const current = this.hidden[layer];
      const hiddenGrad = this.hiddenGrad[layer];
      const prev = layer === 0 ? input : this.hidden[layer - 1];
      const prevSize = layer === 0 ? this.inputSize : this.layerSizes[layer - 1];
      const weights = this.weightOffsets[layer];
      const biases = this.biasOffsets[layer];

      for (let h = 0; h < size; h++) {
        const activation = current[h];
        const delta = hiddenGrad[h] * (1 - activation * activation);
        gradient[biases + h] += delta;

        const row = weights + h * prevSize;
        for (let p = 0; p < prevSize; p++) {
          gradient[row + p] += delta * prev[p];
          if (layer > 0) {
            this.hiddenGrad[layer - 1][p] += this.params[row + p] * delta;
          }
        }
      }
    }
  }
}
