export class NeuralNetwork {
  inputNodes: number;
  hiddenNodes: number;
  outputNodes: number;

  weightsIH: Float32Array; // Size: hidden * input
  weightsHO: Float32Array; // Size: output * hidden
  biasH: Float32Array;
  biasO: Float32Array;

  // Reuse buffers to avoid GC.
  private hiddenBuffer: Float32Array;
  private outputBuffer: Float32Array;

  constructor(inputNodes: number, hiddenNodes: number, outputNodes: number) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    this.weightsIH = new Float32Array(hiddenNodes * inputNodes);
    this.weightsHO = new Float32Array(outputNodes * hiddenNodes);
    this.biasH = new Float32Array(hiddenNodes);
    this.biasO = new Float32Array(outputNodes);

    this.hiddenBuffer = new Float32Array(hiddenNodes);
    this.outputBuffer = new Float32Array(outputNodes);

    this.randomize();
  }

  randomize() {
    for (let i = 0; i < this.weightsIH.length; i++) {
      this.weightsIH[i] = Math.random() * 2 - 1;
    }
    for (let i = 0; i < this.weightsHO.length; i++) {
      this.weightsHO[i] = Math.random() * 2 - 1;
    }
    for (let i = 0; i < this.biasH.length; i++) {
      this.biasH[i] = Math.random() * 2 - 1;
    }
    for (let i = 0; i < this.biasO.length; i++) {
      this.biasO[i] = Math.random() * 2 - 1;
    }
  }

  predict(inputs: number[] | Float32Array): Float32Array {
    // Input -> Hidden
    for (let i = 0; i < this.hiddenNodes; i++) {
      let sum = 0;
      const offset = i * this.inputNodes;
      for (let j = 0; j < this.inputNodes; j++) {
        sum += inputs[j] * this.weightsIH[offset + j];
      }
      sum += this.biasH[i];
      this.hiddenBuffer[i] = 1 / (1 + Math.exp(-sum));
    }

    // Hidden -> Output
    for (let i = 0; i < this.outputNodes; i++) {
      let sum = 0;
      const offset = i * this.hiddenNodes;
      for (let j = 0; j < this.hiddenNodes; j++) {
        sum += this.hiddenBuffer[j] * this.weightsHO[offset + j];
      }
      sum += this.biasO[i];
      this.outputBuffer[i] = 1 / (1 + Math.exp(-sum));
    }

    return this.outputBuffer;
  }

  mutate(rate: number) {
    const mutateArr = (arr: Float32Array) => {
      for (let i = 0; i < arr.length; i++) {
        if (Math.random() < rate) {
          arr[i] += (Math.random() * 2 - 1) * 0.5;
        }
      }
    };

    mutateArr(this.weightsIH);
    mutateArr(this.weightsHO);
    mutateArr(this.biasH);
    mutateArr(this.biasO);
  }

  crossover(partner: NeuralNetwork): NeuralNetwork {
    const child = new NeuralNetwork(
      this.inputNodes,
      this.hiddenNodes,
      this.outputNodes,
    );

    const crossoverArr = (
      childArr: Float32Array,
      myArr: Float32Array,
      partnerArr: Float32Array,
    ) => {
      const midpoint = Math.floor(Math.random() * myArr.length);
      for (let i = 0; i < myArr.length; i++) {
        childArr[i] = i < midpoint ? myArr[i] : partnerArr[i];
      }
    };

    crossoverArr(child.weightsIH, this.weightsIH, partner.weightsIH);
    crossoverArr(child.weightsHO, this.weightsHO, partner.weightsHO);
    crossoverArr(child.biasH, this.biasH, partner.biasH);
    crossoverArr(child.biasO, this.biasO, partner.biasO);

    return child;
  }

  clone(): NeuralNetwork {
    const clone = new NeuralNetwork(
      this.inputNodes,
      this.hiddenNodes,
      this.outputNodes,
    );
    clone.weightsIH.set(this.weightsIH);
    clone.weightsHO.set(this.weightsHO);
    clone.biasH.set(this.biasH);
    clone.biasO.set(this.biasO);
    return clone;
  }

  // Custom JSON serialization for TypedArrays.
  toJSON() {
    return {
      inputNodes: this.inputNodes,
      hiddenNodes: this.hiddenNodes,
      outputNodes: this.outputNodes,
      weightsIH: Array.from(this.weightsIH),
      weightsHO: Array.from(this.weightsHO),
      biasH: Array.from(this.biasH),
      biasO: Array.from(this.biasO),
    };
  }

  static restore(obj: any): NeuralNetwork {
    const nn = new NeuralNetwork(obj.inputNodes, obj.hiddenNodes, obj.outputNodes);

    nn.weightsIH.set(obj.weightsIH);
    nn.weightsHO.set(obj.weightsHO);
    nn.biasH.set(obj.biasH);
    nn.biasO.set(obj.biasO);

    return nn;
  }

  static deserialize(data: string): NeuralNetwork {
    return NeuralNetwork.restore(JSON.parse(data));
  }
}
