import type { Transition } from "./types";

export class ReplayBuffer {
  private readonly states: Array<Uint8Array | null>;
  private readonly actions: Uint8Array;
  private readonly rewards: Float32Array;
  private readonly nextStates: Array<Uint8Array | null>;
  private readonly dones: Uint8Array;

  private cursor = 0;
  private count = 0;

  constructor(private readonly capacity: number) {
    this.states = new Array<Uint8Array | null>(capacity).fill(null);
    this.actions = new Uint8Array(capacity);
    this.rewards = new Float32Array(capacity);
    this.nextStates = new Array<Uint8Array | null>(capacity).fill(null);
    this.dones = new Uint8Array(capacity);
  }

  public get size(): number {
    return this.count;
  }

  public clear(): void {
    this.cursor = 0;
    this.count = 0;
    this.states.fill(null);
    this.nextStates.fill(null);
    this.actions.fill(0);
    this.rewards.fill(0);
    this.dones.fill(0);
  }

  public push(transition: Transition): void {
    const index = this.cursor;

    this.states[index] = transition.state;
    this.actions[index] = transition.action;
    this.rewards[index] = transition.reward;
    this.nextStates[index] = transition.nextState;
    this.dones[index] = transition.done ? 1 : 0;

    this.cursor = (this.cursor + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count += 1;
    }
  }

  public sample(batchSize: number): Transition[] {
    if (this.count === 0) {
      return [];
    }

    const sampleCount = Math.min(batchSize, this.count);
    const batch: Transition[] = [];

    while (batch.length < sampleCount) {
      const index = Math.floor(Math.random() * this.count);
      const state = this.states[index];
      const nextState = this.nextStates[index];

      if (!state || !nextState) {
        continue;
      }

      batch.push({
        state,
        action: this.actions[index],
        reward: this.rewards[index],
        nextState,
        done: this.dones[index] === 1,
      });
    }

    return batch;
  }
}
