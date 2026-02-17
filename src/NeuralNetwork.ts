import { INPUT_LABELS, OUTPUT_LABELS, OUTPUTS } from "./config";
import type {
  ConnectionGene,
  Genome,
  NetworkActivationEdge,
  NetworkActivationNode,
  NetworkActivations,
  NodeGene,
} from "./types";

type EvaluationResult = {
  output: Float32Array;
  best: number;
  valuesByNodeId: Map<number, number>;
};

export class NeuralNetwork {
  public chooseAction(genome: Genome, inputs: Float32Array): number {
    return this.evaluate(genome, inputs).best;
  }

  public computeActivations(
    genome: Genome,
    inputs: Float32Array,
  ): NetworkActivations {
    const evaluated = this.evaluate(genome, inputs);

    const hiddenNodes = genome.nodes
      .filter((node) => node.type === "hidden")
      .sort((left, right) => {
        if (left.layer !== right.layer) {
          return left.layer - right.layer;
        }
        return left.id - right.id;
      });

    const hiddenLabelById = new Map<number, string>();
    for (let i = 0; i < hiddenNodes.length; i++) {
      hiddenLabelById.set(hiddenNodes[i].id, `H${i + 1}`);
    }

    const activationNodes: NetworkActivationNode[] = genome.nodes.map(
      (node) => ({
        id: node.id,
        type: node.type,
        layer: node.layer,
        bias: node.bias,
        ioIndex: node.ioIndex,
        label: this.nodeLabel(node, hiddenLabelById),
        value: evaluated.valuesByNodeId.get(node.id) ?? 0,
      }),
    );

    const labelByNodeId = new Map<number, string>();
    for (const node of activationNodes) {
      labelByNodeId.set(node.id, node.label);
    }

    const activationEdges: NetworkActivationEdge[] = [];
    for (const connection of genome.connections) {
      if (!connection.enabled) {
        continue;
      }

      const fromLabel =
        labelByNodeId.get(connection.from) ?? `N${connection.from}`;
      const toLabel = labelByNodeId.get(connection.to) ?? `N${connection.to}`;
      activationEdges.push({
        from: connection.from,
        to: connection.to,
        weight: connection.weight,
        enabled: connection.enabled,
        label: `${fromLabel} -> ${toLabel}`,
      });
    }

    return {
      nodes: activationNodes,
      edges: activationEdges,
      output: evaluated.output,
      best: evaluated.best,
    };
  }

  private evaluate(genome: Genome, inputs: Float32Array): EvaluationResult {
    const inputNodes = genome.nodes
      .filter((node) => node.type === "input")
      .sort((left, right) => (left.ioIndex ?? 0) - (right.ioIndex ?? 0));

    const outputNodes = genome.nodes
      .filter((node) => node.type === "output")
      .sort((left, right) => (left.ioIndex ?? 0) - (right.ioIndex ?? 0));

    const hiddenAndOutput = genome.nodes
      .filter((node) => node.type !== "input")
      .sort((left, right) => {
        if (left.layer !== right.layer) {
          return left.layer - right.layer;
        }
        return left.id - right.id;
      });

    const incomingByTarget = new Map<number, ConnectionGene[]>();
    for (const connection of genome.connections) {
      if (!connection.enabled) {
        continue;
      }

      const incoming = incomingByTarget.get(connection.to);
      if (incoming) {
        incoming.push(connection);
      } else {
        incomingByTarget.set(connection.to, [connection]);
      }
    }

    const valuesByNodeId = new Map<number, number>();
    for (let i = 0; i < inputNodes.length; i++) {
      const node = inputNodes[i];
      const inputIndex = node.ioIndex ?? i;
      const inputValue = inputIndex < inputs.length ? inputs[inputIndex] : 0;
      valuesByNodeId.set(node.id, inputValue);
    }

    for (const node of hiddenAndOutput) {
      let sum = node.type === "input" ? 0 : node.bias;
      const incoming = incomingByTarget.get(node.id);
      if (incoming) {
        for (const connection of incoming) {
          const sourceValue = valuesByNodeId.get(connection.from) ?? 0;
          sum += sourceValue * connection.weight;
        }
      }

      valuesByNodeId.set(
        node.id,
        node.type === "output" ? sum : Math.tanh(sum),
      );
    }

    const output = new Float32Array(OUTPUTS);
    for (let i = 0; i < outputNodes.length; i++) {
      const outputNode = outputNodes[i];
      const outputIndex = outputNode.ioIndex ?? i;
      if (outputIndex >= 0 && outputIndex < OUTPUTS) {
        output[outputIndex] = valuesByNodeId.get(outputNode.id) ?? 0;
      }
    }

    let best = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < OUTPUTS; i++) {
      if (output[i] > bestValue) {
        bestValue = output[i];
        best = i;
      }
    }

    return {
      output,
      best,
      valuesByNodeId,
    };
  }

  private nodeLabel(
    node: NodeGene,
    hiddenLabelById: Map<number, string>,
  ): string {
    if (node.type === "input") {
      const inputIndex = node.ioIndex ?? 0;
      return INPUT_LABELS[inputIndex] ?? `Input ${inputIndex + 1}`;
    }

    if (node.type === "output") {
      const outputIndex = node.ioIndex ?? 0;
      return OUTPUT_LABELS[outputIndex] ?? `Output ${outputIndex + 1}`;
    }

    return hiddenLabelById.get(node.id) ?? `H${node.id}`;
  }
}
