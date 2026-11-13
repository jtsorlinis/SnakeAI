import { VISION_HIDDEN_UNITS, VISION_MAX_GRAD_NORM } from "./config";

const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPSILON = 1e-8;

// Two-hidden-layer ReLU network for Board PPO. Its input is a list of the
// board inputs that are switched on (index and value); only 150-300 of a
// couple of thousand are at any time, so the first layer adds up just those
// weight rows instead of multiplying through a mostly-zero vector.
export class VisionNetwork {
  public readonly inputCount: number;
  public readonly outputCount: number;
  public readonly params: Float32Array;
  public readonly outputs: Float32Array;

  private readonly h1 = VISION_HIDDEN_UNITS[0];
  private readonly h2 = VISION_HIDDEN_UNITS[1];
  private readonly offsetB1: number;
  private readonly offsetW2: number;
  private readonly offsetB2: number;
  private readonly offsetW3: number;
  private readonly offsetB3: number;

  private readonly hidden1: Float32Array;
  private readonly hidden2: Float32Array;
  private readonly delta1: Float32Array;
  private readonly delta2: Float32Array;
  private readonly gradient: Float32Array;
  private readonly adamM: Float32Array;
  private readonly adamV: Float32Array;
  private adamSteps = 0;

  // typicalActiveInputs is roughly how many inputs are on at once; the first
  // layer's starting weights are scaled for that, not for the full input
  // count. outputScale shrinks the last layer's starting weights; a small
  // value starts a policy close to choosing every move equally often.
  constructor(
    inputCount: number,
    outputCount: number,
    typicalActiveInputs: number,
    outputScale: number,
  ) {
    this.inputCount = inputCount;
    this.outputCount = outputCount;
    this.offsetB1 = inputCount * this.h1;
    this.offsetW2 = this.offsetB1 + this.h1;
    this.offsetB2 = this.offsetW2 + this.h2 * this.h1;
    this.offsetW3 = this.offsetB2 + this.h2;
    this.offsetB3 = this.offsetW3 + outputCount * this.h2;
    const size = this.offsetB3 + outputCount;

    this.params = new Float32Array(size);
    this.gradient = new Float32Array(size);
    this.adamM = new Float32Array(size);
    this.adamV = new Float32Array(size);
    this.outputs = new Float32Array(outputCount);
    this.hidden1 = new Float32Array(this.h1);
    this.hidden2 = new Float32Array(this.h2);
    this.delta1 = new Float32Array(this.h1);
    this.delta2 = new Float32Array(this.h2);

    this.fillUniform(0, this.offsetB1, Math.sqrt(6 / typicalActiveInputs));
    this.fillUniform(this.offsetW2, this.offsetB2, Math.sqrt(6 / this.h1));
    this.fillUniform(
      this.offsetW3,
      this.offsetB3,
      Math.sqrt(6 / this.h2) * outputScale,
    );
  }

  // Runs the network on inputs indices[start .. start + count) and leaves the
  // result in outputs. Pass params to run a saved copy of the weights.
  public forward(
    indices: Int32Array,
    values: Float32Array,
    start: number,
    count: number,
    params: Float32Array = this.params,
  ): void {
    const h1 = this.h1;
    const h2 = this.h2;
    const hidden1 = this.hidden1;
    const hidden2 = this.hidden2;

    for (let j = 0; j < h1; j++) {
      hidden1[j] = params[this.offsetB1 + j];
    }
    for (let a = start; a < start + count; a++) {
      const row = indices[a] * h1;
      const value = values[a];
      if (value === 1) {
        for (let j = 0; j < h1; j++) {
          hidden1[j] += params[row + j];
        }
      } else {
        for (let j = 0; j < h1; j++) {
          hidden1[j] += value * params[row + j];
        }
      }
    }
    for (let j = 0; j < h1; j++) {
      if (hidden1[j] < 0) {
        hidden1[j] = 0;
      }
    }

    for (let k = 0; k < h2; k++) {
      let sum = params[this.offsetB2 + k];
      const row = this.offsetW2 + k * h1;
      for (let j = 0; j < h1; j++) {
        sum += params[row + j] * hidden1[j];
      }
      hidden2[k] = sum > 0 ? sum : 0;
    }

    for (let o = 0; o < this.outputCount; o++) {
      let sum = params[this.offsetB3 + o];
      const row = this.offsetW3 + o * h2;
      for (let k = 0; k < h2; k++) {
        sum += params[row + k] * hidden2[k];
      }
      this.outputs[o] = sum;
    }
  }

  // Adds dLoss/dParams to the pending gradient. Call straight after forward()
  // on the same inputs and the live weights.
  public backward(
    indices: Int32Array,
    values: Float32Array,
    start: number,
    count: number,
    outputGradient: Float32Array,
  ): void {
    const h1 = this.h1;
    const h2 = this.h2;
    const params = this.params;
    const gradient = this.gradient;
    const hidden1 = this.hidden1;
    const hidden2 = this.hidden2;
    const delta1 = this.delta1;
    const delta2 = this.delta2;

    delta2.fill(0);
    for (let o = 0; o < this.outputCount; o++) {
      const d = outputGradient[o];
      if (d === 0) {
        continue;
      }
      gradient[this.offsetB3 + o] += d;
      const row = this.offsetW3 + o * h2;
      for (let k = 0; k < h2; k++) {
        gradient[row + k] += d * hidden2[k];
        delta2[k] += params[row + k] * d;
      }
    }

    delta1.fill(0);
    for (let k = 0; k < h2; k++) {
      const d = delta2[k];
      if (hidden2[k] <= 0 || d === 0) {
        continue;
      }
      gradient[this.offsetB2 + k] += d;
      const row = this.offsetW2 + k * h1;
      for (let j = 0; j < h1; j++) {
        gradient[row + j] += d * hidden1[j];
        delta1[j] += params[row + j] * d;
      }
    }

    for (let j = 0; j < h1; j++) {
      if (hidden1[j] <= 0) {
        delta1[j] = 0;
      } else {
        gradient[this.offsetB1 + j] += delta1[j];
      }
    }
    for (let a = start; a < start + count; a++) {
      const row = indices[a] * h1;
      const value = values[a];
      for (let j = 0; j < h1; j++) {
        gradient[row + j] += value * delta1[j];
      }
    }
  }

  // One Adam step on the pending gradient (clipped by norm), then clears it.
  public applyGradient(learningRate: number): void {
    const gradient = this.gradient;
    const params = this.params;
    const m = this.adamM;
    const v = this.adamV;

    let normSquared = 0;
    for (let i = 0; i < gradient.length; i++) {
      normSquared += gradient[i] * gradient[i];
    }
    const norm = Math.sqrt(normSquared);
    const scale = norm > VISION_MAX_GRAD_NORM ? VISION_MAX_GRAD_NORM / norm : 1;

    this.adamSteps += 1;
    const stepSize =
      (learningRate * Math.sqrt(1 - Math.pow(ADAM_BETA2, this.adamSteps))) /
      (1 - Math.pow(ADAM_BETA1, this.adamSteps));

    for (let i = 0; i < gradient.length; i++) {
      const g = gradient[i] * scale;
      const mi = (m[i] = ADAM_BETA1 * m[i] + (1 - ADAM_BETA1) * g);
      const vi = (v[i] = ADAM_BETA2 * v[i] + (1 - ADAM_BETA2) * g * g);
      params[i] -= (stepSize * mi) / (Math.sqrt(vi) + ADAM_EPSILON);
      gradient[i] = 0;
    }
  }

  private fillUniform(from: number, to: number, limit: number): void {
    for (let i = from; i < to; i++) {
      this.params[i] = (Math.random() * 2 - 1) * limit;
    }
  }
}
