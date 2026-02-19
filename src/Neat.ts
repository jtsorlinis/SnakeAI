import {
  GRID_SIZE,
  INPUTS,
  NEAT_ADD_CONNECTION_RATE,
  NEAT_ADD_NODE_RATE,
  NEAT_BIAS_MUTATION_RATE,
  NEAT_COMPATIBILITY_GENE_COEFF,
  NEAT_COMPATIBILITY_THRESHOLD,
  NEAT_COMPATIBILITY_WEIGHT_COEFF,
  NEAT_DISABLE_INHERITED_GENE_RATE,
  NEAT_MAX_NODES,
  NEAT_SURVIVAL_RATIO,
  NEAT_WEIGHT_MUTATION_RATE,
  NEAT_WEIGHT_MUTATION_SIZE,
  OUTPUTS,
  POP_SIZE,
  TOURNAMENT_SIZE,
} from "./config";
import type { Agent, ConnectionGene, Genome, NodeGene } from "./types";

const TOURNAMENT_POOL_RATIO = 0.4;
const SPECIES_STALE_LIMIT = 18;
const MIN_LAYER_GAP = 0.000001;

type Species = {
  id: number;
  representative: Genome;
  members: Agent[];
  bestFitness: number;
  stale: number;
};

type SpeciesHistory = {
  id: number;
  representative: Genome;
  bestFitness: number;
  stale: number;
};

export type EvolutionResult = {
  best: Agent;
  nextGenomes: Genome[];
};

export class NeatAlgorithm {
  private nextNodeId = 0;
  private nextSpeciesId = 1;
  private nextInnovation = 1;

  private readonly inputNodeIds: number[] = [];
  private readonly outputNodeIds: number[] = [];
  private readonly innovationByPair = new Map<string, number>();
  private readonly splitNodeByInnovation = new Map<number, number>();
  private speciesHistory: SpeciesHistory[] = [];

  constructor() {
    for (let i = 0; i < INPUTS; i++) {
      this.inputNodeIds.push(this.nextNodeId);
      this.nextNodeId += 1;
    }

    for (let o = 0; o < OUTPUTS; o++) {
      this.outputNodeIds.push(this.nextNodeId);
      this.nextNodeId += 1;
    }

    for (const from of this.inputNodeIds) {
      for (const to of this.outputNodeIds) {
        this.getInnovation(from, to);
      }
    }
  }

  public randomGenome(): Genome {
    const nodes: NodeGene[] = [];
    for (let i = 0; i < this.inputNodeIds.length; i++) {
      nodes.push({
        id: this.inputNodeIds[i],
        type: "input",
        layer: 0,
        bias: 0,
        ioIndex: i,
      });
    }

    for (let o = 0; o < this.outputNodeIds.length; o++) {
      nodes.push({
        id: this.outputNodeIds[o],
        type: "output",
        layer: 1,
        bias: this.randomRange(-1, 1),
        ioIndex: o,
      });
    }

    const connections: ConnectionGene[] = [];
    for (const from of this.inputNodeIds) {
      for (const to of this.outputNodeIds) {
        connections.push({
          innovation: this.getInnovation(from, to),
          from,
          to,
          weight: this.randomRange(-1, 1),
          enabled: true,
        });
      }
    }

    return { nodes, connections };
  }

  public evolve(population: Agent[]): EvolutionResult {
    for (const agent of population) {
      agent.fitness = this.fitness(agent);
    }

    const ranked = [...population].sort((a, b) => b.fitness - a.fitness);
    const species = this.speciate(ranked);
    const nextGenomes = this.breedNextGeneration(species, ranked[0]);

    return {
      best: ranked[0],
      nextGenomes,
    };
  }

  private fitness(agent: Agent): number {
    const foodReward = agent.score;
    const deathPenalty = !agent.alive && agent.hunger > 0 ? 1 : 0;
    const stepPenalty = agent.steps / (GRID_SIZE * GRID_SIZE);
    return foodReward - deathPenalty - stepPenalty;
  }

  private speciate(population: Agent[]): Species[] {
    const seededSpecies: Species[] = this.speciesHistory.map((entry) => ({
      id: entry.id,
      representative: cloneGenome(entry.representative),
      members: [],
      bestFitness: entry.bestFitness,
      stale: entry.stale,
    }));

    for (const agent of population) {
      const match = this.findMatchingSpecies(agent.genome, seededSpecies);
      if (match) {
        match.members.push(agent);
        agent.speciesId = match.id;
      } else {
        const created: Species = {
          id: this.nextSpeciesId,
          representative: cloneGenome(agent.genome),
          members: [agent],
          bestFitness: agent.fitness,
          stale: 0,
        };
        this.nextSpeciesId += 1;
        agent.speciesId = created.id;
        seededSpecies.push(created);
      }
    }

    const activeSpecies = seededSpecies.filter(
      (species) => species.members.length > 0,
    );
    for (const species of activeSpecies) {
      species.members.sort((a, b) => b.fitness - a.fitness);
      const championFitness = species.members[0].fitness;
      if (championFitness > species.bestFitness) {
        species.bestFitness = championFitness;
        species.stale = 0;
      } else {
        species.stale += 1;
      }
      species.representative = cloneGenome(species.members[0].genome);
    }

    activeSpecies.sort((a, b) => b.members[0].fitness - a.members[0].fitness);

    const pruned =
      activeSpecies.length <= 1
        ? activeSpecies
        : activeSpecies.filter(
            (species, index) =>
              index === 0 || species.stale < SPECIES_STALE_LIMIT,
          );

    this.speciesHistory = pruned.map((species) => ({
      id: species.id,
      representative: cloneGenome(species.representative),
      bestFitness: species.bestFitness,
      stale: species.stale,
    }));

    return pruned;
  }

  private findMatchingSpecies(
    genome: Genome,
    species: Species[],
  ): Species | null {
    let bestMatch: Species | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const group of species) {
      const distance = this.compatibilityDistance(genome, group.representative);
      if (distance <= NEAT_COMPATIBILITY_THRESHOLD && distance < bestDistance) {
        bestDistance = distance;
        bestMatch = group;
      }
    }

    return bestMatch;
  }

  private compatibilityDistance(a: Genome, b: Genome): number {
    const aByInnovation = new Map<number, ConnectionGene>();
    for (const gene of a.connections) {
      aByInnovation.set(gene.innovation, gene);
    }

    const bByInnovation = new Map<number, ConnectionGene>();
    for (const gene of b.connections) {
      bByInnovation.set(gene.innovation, gene);
    }

    const innovations = new Set<number>([
      ...aByInnovation.keys(),
      ...bByInnovation.keys(),
    ]);

    let unmatched = 0;
    let matching = 0;
    let weightDiff = 0;

    for (const innovation of innovations) {
      const left = aByInnovation.get(innovation);
      const right = bByInnovation.get(innovation);

      if (left && right) {
        matching += 1;
        weightDiff += Math.abs(left.weight - right.weight);
      } else {
        unmatched += 1;
      }
    }

    const normalizer = Math.max(
      1,
      Math.max(a.connections.length, b.connections.length),
    );
    const avgWeightDiff = matching > 0 ? weightDiff / matching : 0;
    return (
      NEAT_COMPATIBILITY_GENE_COEFF * (unmatched / normalizer) +
      NEAT_COMPATIBILITY_WEIGHT_COEFF * avgWeightDiff
    );
  }

  private breedNextGeneration(species: Species[], best: Agent): Genome[] {
    if (species.length === 0) {
      const fallback: Genome[] = [];
      for (let i = 0; i < POP_SIZE; i++) {
        fallback.push(cloneGenome(best.genome));
      }
      return fallback;
    }

    const nextGenomes: Genome[] = [];

    for (const group of species) {
      nextGenomes.push(cloneGenome(group.members[0].genome));
      if (nextGenomes.length >= POP_SIZE) {
        return nextGenomes;
      }
    }

    const speciesScores = this.computeSpeciesScores(species);
    const remainingSlots = POP_SIZE - nextGenomes.length;
    const quotas = this.distributeQuotas(speciesScores, remainingSlots);

    for (let i = 0; i < species.length; i++) {
      for (let count = 0; count < quotas[i]; count++) {
        nextGenomes.push(this.spawnChild(species[i]));
      }
    }

    while (nextGenomes.length < POP_SIZE) {
      const sampledSpecies = this.pickSpecies(species, speciesScores);
      if (sampledSpecies) {
        nextGenomes.push(this.spawnChild(sampledSpecies));
      } else {
        nextGenomes.push(cloneGenome(best.genome));
      }
    }

    if (nextGenomes.length > POP_SIZE) {
      nextGenomes.length = POP_SIZE;
    }

    return nextGenomes;
  }

  private computeSpeciesScores(species: Species[]): number[] {
    let minFitness = Number.POSITIVE_INFINITY;
    for (const group of species) {
      for (const member of group.members) {
        minFitness = Math.min(minFitness, member.fitness);
      }
    }

    const scores: number[] = [];
    for (const group of species) {
      const size = Math.max(1, group.members.length);
      let score = 0;
      for (const member of group.members) {
        score += (member.fitness - minFitness + 1) / size;
      }
      scores.push(Math.max(0.0001, score));
    }

    return scores;
  }

  private distributeQuotas(scores: number[], totalSlots: number): number[] {
    const quotas = scores.map(() => 0);
    if (totalSlots <= 0 || scores.length === 0) {
      return quotas;
    }

    const scoreSum = scores.reduce((sum, score) => sum + score, 0);
    if (scoreSum <= 0) {
      let index = 0;
      for (let slot = 0; slot < totalSlots; slot++) {
        quotas[index] += 1;
        index = (index + 1) % quotas.length;
      }
      return quotas;
    }

    const fractions: Array<{ index: number; frac: number }> = [];
    let allocated = 0;
    for (let i = 0; i < scores.length; i++) {
      const raw = (scores[i] / scoreSum) * totalSlots;
      const base = Math.floor(raw);
      quotas[i] = base;
      allocated += base;
      fractions.push({ index: i, frac: raw - base });
    }

    fractions.sort((a, b) => b.frac - a.frac);
    let remaining = totalSlots - allocated;
    let cursor = 0;
    while (remaining > 0 && fractions.length > 0) {
      quotas[fractions[cursor].index] += 1;
      cursor = (cursor + 1) % fractions.length;
      remaining -= 1;
    }

    return quotas;
  }

  private pickSpecies(species: Species[], scores: number[]): Species | null {
    if (species.length === 0) {
      return null;
    }

    const total = scores.reduce((sum, score) => sum + score, 0);
    if (total <= 0) {
      return species[this.randomIndex(species.length)];
    }

    let threshold = Math.random() * total;
    for (let i = 0; i < species.length; i++) {
      threshold -= scores[i];
      if (threshold <= 0) {
        return species[i];
      }
    }

    return species[species.length - 1];
  }

  private spawnChild(species: Species): Genome {
    const survivorCount = Math.max(
      1,
      Math.ceil(species.members.length * NEAT_SURVIVAL_RATIO),
    );
    const survivors = species.members.slice(0, survivorCount);

    let child: Genome;
    if (survivors.length === 1) {
      child = cloneGenome(survivors[0].genome);
    } else {
      const parentA = this.pickParent(survivors);
      const parentB = this.pickParent(survivors);
      child = this.crossover(parentA, parentB);
    }

    this.mutate(child);
    return child;
  }

  private crossover(parentA: Agent, parentB: Agent): Genome {
    let fitter = parentA;
    let other = parentB;
    if (
      parentB.fitness > parentA.fitness ||
      (parentB.fitness === parentA.fitness &&
        parentB.genome.connections.length > parentA.genome.connections.length)
    ) {
      fitter = parentB;
      other = parentA;
    }

    const fitterByInnovation = new Map<number, ConnectionGene>();
    for (const gene of fitter.genome.connections) {
      fitterByInnovation.set(gene.innovation, gene);
    }

    const otherByInnovation = new Map<number, ConnectionGene>();
    for (const gene of other.genome.connections) {
      otherByInnovation.set(gene.innovation, gene);
    }

    const innovations = [
      ...new Set<number>([
        ...fitterByInnovation.keys(),
        ...otherByInnovation.keys(),
      ]),
    ].sort((left, right) => left - right);

    const childConnections: ConnectionGene[] = [];
    for (const innovation of innovations) {
      const left = fitterByInnovation.get(innovation);
      const right = otherByInnovation.get(innovation);

      if (left && right) {
        const source = Math.random() < 0.5 ? left : right;
        const enabled =
          left.enabled && right.enabled
            ? true
            : Math.random() >= NEAT_DISABLE_INHERITED_GENE_RATE;
        childConnections.push({ ...source, enabled });
      } else if (left) {
        childConnections.push({ ...left });
      }
    }

    const fitterNodesById = new Map<number, NodeGene>();
    for (const node of fitter.genome.nodes) {
      fitterNodesById.set(node.id, node);
    }

    const otherNodesById = new Map<number, NodeGene>();
    for (const node of other.genome.nodes) {
      otherNodesById.set(node.id, node);
    }

    const requiredNodeIds = new Set<number>([
      ...this.inputNodeIds,
      ...this.outputNodeIds,
    ]);
    for (const connection of childConnections) {
      requiredNodeIds.add(connection.from);
      requiredNodeIds.add(connection.to);
    }

    const childNodes: NodeGene[] = [];
    for (const nodeId of requiredNodeIds) {
      const fitterNode = fitterNodesById.get(nodeId);
      const otherNode = otherNodesById.get(nodeId);
      const chosen =
        fitterNode && otherNode
          ? Math.random() < 0.5
            ? fitterNode
            : otherNode
          : (fitterNode ?? otherNode);

      if (!chosen) {
        continue;
      }

      childNodes.push({
        id: chosen.id,
        type: chosen.type,
        layer: chosen.layer,
        bias: chosen.type === "input" ? 0 : chosen.bias,
        ioIndex: chosen.ioIndex,
      });
    }

    const child: Genome = {
      nodes: childNodes,
      connections: childConnections,
    };

    this.ensureAtLeastOneConnection(child);
    this.sortGenome(child);
    return child;
  }

  private mutate(genome: Genome): void {
    this.mutateWeights(genome);
    this.mutateBiases(genome);

    if (Math.random() < NEAT_ADD_CONNECTION_RATE) {
      this.mutateAddConnection(genome);
    }

    if (Math.random() < NEAT_ADD_NODE_RATE) {
      this.mutateAddNode(genome);
    }

    this.ensureAtLeastOneConnection(genome);
    this.sortGenome(genome);
  }

  private mutateWeights(genome: Genome): void {
    for (const gene of genome.connections) {
      if (Math.random() < NEAT_WEIGHT_MUTATION_RATE) {
        if (Math.random() < 0.9) {
          gene.weight += this.randomRange(
            -NEAT_WEIGHT_MUTATION_SIZE,
            NEAT_WEIGHT_MUTATION_SIZE,
          );
        } else {
          gene.weight = this.randomRange(-1, 1);
        }
      }
    }
  }

  private mutateBiases(genome: Genome): void {
    for (const node of genome.nodes) {
      if (node.type === "input") {
        continue;
      }

      if (Math.random() < NEAT_BIAS_MUTATION_RATE) {
        node.bias += this.randomRange(
          -NEAT_WEIGHT_MUTATION_SIZE,
          NEAT_WEIGHT_MUTATION_SIZE,
        );
      }
    }
  }

  private mutateAddConnection(genome: Genome): void {
    const existingPairs = new Set<string>();
    for (const connection of genome.connections) {
      existingPairs.add(this.pairKey(connection.from, connection.to));
    }

    const candidates: Array<{ from: number; to: number }> = [];
    for (const fromNode of genome.nodes) {
      if (fromNode.type === "output") {
        continue;
      }

      for (const toNode of genome.nodes) {
        if (toNode.type === "input" || fromNode.id === toNode.id) {
          continue;
        }

        if (fromNode.layer + MIN_LAYER_GAP >= toNode.layer) {
          continue;
        }

        const key = this.pairKey(fromNode.id, toNode.id);
        if (!existingPairs.has(key)) {
          candidates.push({ from: fromNode.id, to: toNode.id });
        }
      }
    }

    if (candidates.length === 0) {
      return;
    }

    const candidate = candidates[this.randomIndex(candidates.length)];
    genome.connections.push({
      innovation: this.getInnovation(candidate.from, candidate.to),
      from: candidate.from,
      to: candidate.to,
      weight: this.randomRange(-1, 1),
      enabled: true,
    });
  }

  private mutateAddNode(genome: Genome): void {
    if (genome.nodes.length >= NEAT_MAX_NODES) {
      return;
    }

    const enabled = genome.connections.filter(
      (connection) => connection.enabled,
    );
    if (enabled.length === 0) {
      return;
    }

    const splitConnection = enabled[this.randomIndex(enabled.length)];
    const fromNode = genome.nodes.find(
      (node) => node.id === splitConnection.from,
    );
    const toNode = genome.nodes.find((node) => node.id === splitConnection.to);
    if (!fromNode || !toNode) {
      return;
    }

    const layer = (fromNode.layer + toNode.layer) / 2;
    if (
      layer <= fromNode.layer + MIN_LAYER_GAP ||
      layer >= toNode.layer - MIN_LAYER_GAP
    ) {
      return;
    }

    splitConnection.enabled = false;

    let nodeId = this.splitNodeByInnovation.get(splitConnection.innovation);
    if (nodeId === undefined) {
      nodeId = this.nextNodeId;
      this.nextNodeId += 1;
      this.splitNodeByInnovation.set(splitConnection.innovation, nodeId);
    }

    if (!genome.nodes.some((node) => node.id === nodeId)) {
      genome.nodes.push({
        id: nodeId,
        type: "hidden",
        layer,
        bias: this.randomRange(-1, 1),
      });
    }

    this.addConnectionIfMissing(genome, fromNode.id, nodeId, 1);
    this.addConnectionIfMissing(
      genome,
      nodeId,
      toNode.id,
      splitConnection.weight,
    );
  }

  private addConnectionIfMissing(
    genome: Genome,
    from: number,
    to: number,
    weight: number,
  ): void {
    const existing = genome.connections.find(
      (connection) => connection.from === from && connection.to === to,
    );

    if (existing) {
      existing.enabled = true;
      existing.weight = weight;
      return;
    }

    genome.connections.push({
      innovation: this.getInnovation(from, to),
      from,
      to,
      weight,
      enabled: true,
    });
  }

  private ensureAtLeastOneConnection(genome: Genome): void {
    if (genome.connections.some((connection) => connection.enabled)) {
      return;
    }

    const from = this.inputNodeIds[this.randomIndex(this.inputNodeIds.length)];
    const to = this.outputNodeIds[this.randomIndex(this.outputNodeIds.length)];
    this.addConnectionIfMissing(genome, from, to, this.randomRange(-1, 1));
  }

  private sortGenome(genome: Genome): void {
    genome.nodes.sort((left, right) => {
      if (left.layer !== right.layer) {
        return left.layer - right.layer;
      }
      return left.id - right.id;
    });
    genome.connections.sort(
      (left, right) => left.innovation - right.innovation,
    );
  }

  private pairKey(from: number, to: number): string {
    return `${from}:${to}`;
  }

  private getInnovation(from: number, to: number): number {
    const key = this.pairKey(from, to);
    const existing = this.innovationByPair.get(key);
    if (existing !== undefined) {
      return existing;
    }

    const innovation = this.nextInnovation;
    this.nextInnovation += 1;
    this.innovationByPair.set(key, innovation);
    return innovation;
  }

  private pickParent(candidates: Agent[]): Agent {
    if (candidates.length === 1) {
      return candidates[0];
    }

    const poolSize = Math.max(
      2,
      Math.floor(candidates.length * TOURNAMENT_POOL_RATIO),
    );
    let winner = candidates[this.randomIndex(poolSize)];

    for (let i = 1; i < TOURNAMENT_SIZE; i++) {
      const challenger = candidates[this.randomIndex(poolSize)];
      if (challenger.fitness > winner.fitness) {
        winner = challenger;
      }
    }

    return winner;
  }

  private randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
  }

  private randomIndex(maxExclusive: number): number {
    return Math.floor(Math.random() * maxExclusive);
  }
}

export function cloneGenome(genome: Genome): Genome {
  return {
    nodes: genome.nodes.map((node) => ({ ...node })),
    connections: genome.connections.map((connection) => ({ ...connection })),
  };
}
