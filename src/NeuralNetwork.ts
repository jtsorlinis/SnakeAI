import {
  HIDDEN_LAYERS,
  HIDDEN_LAYER_UNITS,
  INPUTS,
  OFFSET_H_BIAS,
  OFFSET_HH,
  OFFSET_HO,
  OFFSET_IH,
  OFFSET_O_BIAS,
  OUTPUTS,
} from "./config";
import type { Genome } from "./types";

function createHiddenLayerBuffers(): Float32Array[] {
  return HIDDEN_LAYER_UNITS.map((units) => new Float32Array(units));
}

export class NeuralNetwork {
  private readonly actionHiddenBuffers = createHiddenLayerBuffers();

  public chooseAction(genome: Genome, inputs: Float32Array): number {
    return this.run(genome, inputs, this.actionHiddenBuffers);
  }

  public run(
    genome: Genome,
    inputs: Float32Array,
    hiddenTarget: Float32Array[],
    outputTarget?: Float32Array,
  ): number {
    if (HIDDEN_LAYERS > 0) {
      const firstLayerSize = HIDDEN_LAYER_UNITS[0];
      const firstHidden = hiddenTarget[0];
      for (let h = 0; h < firstLayerSize; h++) {
        let sum = genome[OFFSET_H_BIAS + h];
        const wOffset = OFFSET_IH + h * INPUTS;
        for (let i = 0; i < INPUTS; i++) {
          sum += genome[wOffset + i] * inputs[i];
        }
        firstHidden[h] = Math.tanh(sum);
      }

      let hhOffset = OFFSET_HH;
      let biasOffset = OFFSET_H_BIAS + firstLayerSize;
      for (let layer = 1; layer < HIDDEN_LAYERS; layer++) {
        const prev = hiddenTarget[layer - 1];
        const current = hiddenTarget[layer];
        const prevSize = HIDDEN_LAYER_UNITS[layer - 1];
        const currentSize = HIDDEN_LAYER_UNITS[layer];

        for (let h = 0; h < currentSize; h++) {
          let sum = genome[biasOffset + h];
          const wOffset = hhOffset + h * prevSize;
          for (let k = 0; k < prevSize; k++) {
            sum += genome[wOffset + k] * prev[k];
          }
          current[h] = Math.tanh(sum);
        }

        hhOffset += prevSize * currentSize;
        biasOffset += currentSize;
      }
    }

    let bestAction = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    const outputInputs =
      HIDDEN_LAYERS > 0 ? hiddenTarget[HIDDEN_LAYERS - 1] : inputs;
    const outputInputSize =
      HIDDEN_LAYERS > 0 ? HIDDEN_LAYER_UNITS[HIDDEN_LAYERS - 1] : INPUTS;

    for (let output = 0; output < OUTPUTS; output++) {
      let value = genome[OFFSET_O_BIAS + output];
      const wOffset = OFFSET_HO + output * outputInputSize;
      for (let h = 0; h < outputInputSize; h++) {
        value += genome[wOffset + h] * outputInputs[h];
      }

      if (outputTarget) {
        outputTarget[output] = value;
      }

      if (value > bestValue) {
        bestValue = value;
        bestAction = output;
      }
    }

    return bestAction;
  }
}

export function createNetworkHiddenBuffers(): Float32Array[] {
  return createHiddenLayerBuffers();
}
