import { PRIORITY_ALPHA, PRIORITY_EPSILON } from "./config";
import type { Transition } from "./types";

export type ReplaySample = {
  transitions: Transition[];
  indices: Uint32Array;
  weights: Float32Array;
};

class SumTree {
  private readonly leafCount: number;
  private readonly tree: Float32Array;

  constructor(private readonly capacity: number) {
    let leaves = 1;
    while (leaves < capacity) {
      leaves <<= 1;
    }
    this.leafCount = leaves;
    this.tree = new Float32Array(this.leafCount * 2);
  }

  public clear(): void {
    this.tree.fill(0);
  }

  public get(index: number): number {
    return this.tree[this.leafCount + index];
  }

  public set(index: number, value: number): void {
    let treeIndex = this.leafCount + index;
    const delta = value - this.tree[treeIndex];
    this.tree[treeIndex] = value;
    treeIndex >>= 1;

    while (treeIndex >= 1) {
      this.tree[treeIndex] += delta;
      treeIndex >>= 1;
    }
  }

  public total(): number {
    return this.tree[1];
  }

  public sample(prefixSum: number): number {
    let node = 1;
    let remaining = Math.max(0, prefixSum);

    while (node < this.leafCount) {
      const left = node << 1;
      const leftSum = this.tree[left];
      if (remaining <= leftSum) {
        node = left;
      } else {
        remaining -= leftSum;
        node = left + 1;
      }
    }

    const index = node - this.leafCount;
    return Math.min(this.capacity - 1, Math.max(0, index));
  }
}

export class ReplayBuffer {
  private readonly states: Array<Uint8Array | null>;
  private readonly actions: Uint8Array;
  private readonly rewards: Float32Array;
  private readonly nextStates: Array<Uint8Array | null>;
  private readonly dones: Uint8Array;
  private readonly priorities: Float32Array;
  private readonly sumTree: SumTree;

  private cursor = 0;
  private count = 0;
  private maxPriority = 1;

  constructor(private readonly capacity: number) {
    this.states = new Array<Uint8Array | null>(capacity).fill(null);
    this.actions = new Uint8Array(capacity);
    this.rewards = new Float32Array(capacity);
    this.nextStates = new Array<Uint8Array | null>(capacity).fill(null);
    this.dones = new Uint8Array(capacity);
    this.priorities = new Float32Array(capacity);
    this.sumTree = new SumTree(capacity);
  }

  public get size(): number {
    return this.count;
  }

  public clear(): void {
    this.cursor = 0;
    this.count = 0;
    this.maxPriority = 1;
    this.states.fill(null);
    this.nextStates.fill(null);
    this.actions.fill(0);
    this.rewards.fill(0);
    this.dones.fill(0);
    this.priorities.fill(0);
    this.sumTree.clear();
  }

  public push(transition: Transition): void {
    const index = this.cursor;

    this.states[index] = transition.state;
    this.actions[index] = transition.action;
    this.rewards[index] = transition.reward;
    this.nextStates[index] = transition.nextState;
    this.dones[index] = transition.done ? 1 : 0;

    const scaledPriority = Math.pow(this.maxPriority, PRIORITY_ALPHA);
    this.sumTree.set(index, scaledPriority);
    this.priorities[index] = this.maxPriority;

    this.cursor = (this.cursor + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count += 1;
    }
  }

  public sample(batchSize: number, beta: number): ReplaySample {
    const empty: ReplaySample = {
      transitions: [],
      indices: new Uint32Array(0),
      weights: new Float32Array(0),
    };

    if (this.count === 0) {
      return empty;
    }

    const totalPriority = this.sumTree.total();
    if (totalPriority <= 0) {
      return empty;
    }

    const sampleCount = Math.min(batchSize, this.count);
    const transitions: Transition[] = [];
    const indices = new Uint32Array(sampleCount);
    const weights = new Float32Array(sampleCount);

    const segment = totalPriority / sampleCount;
    let maxWeight = 0;

    for (let i = 0; i < sampleCount; i++) {
      const low = segment * i;
      const high = segment * (i + 1);
      const sampleValue = low + Math.random() * (high - low);
      const index = this.sumTree.sample(sampleValue);

      const state = this.states[index];
      const nextState = this.nextStates[index];
      if (!state || !nextState) {
        continue;
      }

      const probability = this.sumTree.get(index) / totalPriority;
      const weight = Math.pow(this.count * probability, -beta);
      maxWeight = Math.max(maxWeight, weight);

      indices[transitions.length] = index;
      weights[transitions.length] = weight;
      transitions.push({
        state,
        action: this.actions[index],
        reward: this.rewards[index],
        nextState,
        done: this.dones[index] === 1,
      });
    }

    const actualSize = transitions.length;
    if (actualSize === 0) {
      return empty;
    }

    const normalizedWeights = new Float32Array(actualSize);
    const outIndices = new Uint32Array(actualSize);
    const normalizer = maxWeight > 0 ? 1 / maxWeight : 1;

    for (let i = 0; i < actualSize; i++) {
      normalizedWeights[i] = weights[i] * normalizer;
      outIndices[i] = indices[i];
    }

    return {
      transitions,
      indices: outIndices,
      weights: normalizedWeights,
    };
  }

  public updatePriorities(
    indices: Uint32Array,
    tdErrors: Float32Array,
  ): void {
    const length = Math.min(indices.length, tdErrors.length);

    for (let i = 0; i < length; i++) {
      const index = indices[i];
      const priority = Math.max(PRIORITY_EPSILON, Math.abs(tdErrors[i]));
      this.priorities[index] = priority;
      this.maxPriority = Math.max(this.maxPriority, priority);
      this.sumTree.set(index, Math.pow(priority, PRIORITY_ALPHA));
    }
  }
}
