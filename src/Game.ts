import { Snake, BRAIN_CONFIG, WIN_REWARD } from "./Snake";
import { Food } from "./Food";
import { NeuralNetwork } from "./NeuralNetwork";
import {
  TILE_SIZE,
  CANVAS_WIDTH,
  CANVAS_HEIGHT,
  GRID_WIDTH,
  GRID_HEIGHT,
} from "./types";

export class Game {
  private static readonly SAVE_KEY = "bestSnakeBrain";

  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;

  population: Snake[] = [];
  foods: Food[] = [];
  generation: number = 1;
  bestScore: number = 0;
  bestFitness: number = 0;

  scoreElement: HTMLElement;
  currentScoreElement: HTMLElement;
  lastTime: number;
  targetFPS: number = 60;
  turboMode: boolean = false;
  showAll: boolean = false;

  popSize: number = 150;
  mutationRate: number = 0.05;
  currentMutationRate: number = 0.05;

  averageScore: number = 0;
  averageFitness: number = 0;
  gensSinceImprovement: number = 0;
  bestFitnessGen: number = 1;
  lastStatus: string = "";

  turboToggle: HTMLInputElement;
  showAllToggle: HTMLInputElement;
  resetBtn: HTMLButtonElement;
  toastElement: HTMLElement;
  toastTimeout: number | null = null;

  savedStatsDiv: HTMLElement;
  savedGenSpan: HTMLElement;
  savedScoreSpan: HTMLElement;
  savedFitnessSpan: HTMLElement;
  savedDateSpan: HTMLElement;

  constructor() {
    this.canvas = document.getElementById("gameCanvas") as HTMLCanvasElement;
    this.ctx = this.canvas.getContext("2d")!;
    this.scoreElement = document.getElementById("score")!;
    this.currentScoreElement = document.getElementById("currentScore")!;
    this.turboToggle = document.getElementById(
      "turboToggle",
    ) as HTMLInputElement;
    this.showAllToggle = document.getElementById(
      "showAllToggle",
    ) as HTMLInputElement;
    this.resetBtn = document.getElementById("resetBtn") as HTMLButtonElement;
    this.toastElement = document.getElementById("toast")!;

    this.savedStatsDiv = document.getElementById("savedStats")!;
    this.savedGenSpan = document.getElementById("savedGen")!;
    this.savedScoreSpan = document.getElementById("savedScore")!;
    this.savedFitnessSpan = document.getElementById("savedFitness")!;
    this.savedDateSpan = document.getElementById("savedDate")!;

    this.canvas.width = CANVAS_WIDTH;
    this.canvas.height = CANVAS_HEIGHT;

    this.lastTime = 0;

    if (localStorage.getItem(Game.SAVE_KEY)) {
      this.loadBestSnake(true);
    } else {
      this.initAIGame();
    }

    this.updateSavedStatsDisplay();
    this.setupInput();
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  private getScore(snake: Snake): number {
    return snake.score;
  }

  private getStatus(): string {
    if (this.bestFitness <= 0) return "Bootstrapping";
    if (this.gensSinceImprovement >= 100) return "Plateaued";
    if (this.gensSinceImprovement >= 50) return "Stuck";
    if (this.gensSinceImprovement >= 20) return "Slowing";
    return "Improving";
  }

  showToast(message: string) {
    if (this.toastTimeout) {
      clearTimeout(this.toastTimeout);
    }
    this.toastElement.textContent = message;
    this.toastElement.classList.add("visible");

    this.toastTimeout = window.setTimeout(() => {
      this.toastElement.classList.remove("visible");
    }, 2500);
  }

  initAIGame() {
    this.population.length = 0;
    this.foods.length = 0;
    this.averageFitness = 0;
    for (let i = 0; i < this.popSize; i++) {
      this.population.push(new Snake());
      this.foods.push(new Food());
    }
    this.targetFPS = 60;
  }

  setupInput() {
    this.turboToggle.addEventListener("change", () => {
      this.turboMode = this.turboToggle.checked;
    });

    this.showAllToggle.addEventListener("change", () => {
      this.showAll = this.showAllToggle.checked;
    });

    this.resetBtn.addEventListener("click", () => {
      if (
        confirm("Are you sure you want to delete the best snake and restart?")
      ) {
        this.resetBestSnake();
      }
    });
  }

  resetBestSnake() {
    localStorage.removeItem(Game.SAVE_KEY);
    this.bestScore = 0;
    this.bestFitness = 0;
    this.averageFitness = 0;
    this.generation = 1;
    this.bestFitnessGen = 1;
    this.gensSinceImprovement = 0;
    this.updateSavedStatsDisplay();
    this.initAIGame();
    this.showToast("Progress Reset");
  }

  updateSavedStatsDisplay() {
    const raw = localStorage.getItem(Game.SAVE_KEY);
    if (!raw) {
      this.savedStatsDiv.style.display = "none";
      return;
    }

    this.savedStatsDiv.style.display = "block";

    try {
      const data = JSON.parse(raw);
      if (data.brainData) {
        this.savedGenSpan.textContent = data.generation;
        this.savedScoreSpan.textContent = data.score;
        this.savedFitnessSpan.textContent =
          typeof data.fitness === "number" ? data.fitness.toFixed(2) : "?";
        this.savedDateSpan.textContent = data.date;
      } else {
        this.savedGenSpan.textContent = "?";
        this.savedScoreSpan.textContent = "?";
        this.savedFitnessSpan.textContent = "?";
        this.savedDateSpan.textContent = "Old Save Format";
      }
    } catch {
      this.savedStatsDiv.style.display = "none";
    }
  }

  saveBestSnake(targetSnake?: Snake) {
    if (this.population.length === 0 && !targetSnake) return;

    const champion = targetSnake || this.population[0];
    const saveObject = {
      brainData: champion.brain,
      generation: this.generation,
      score: champion.score,
      fitness: champion.fitness,
      date: new Date().toLocaleString(),
    };

    localStorage.setItem(Game.SAVE_KEY, JSON.stringify(saveObject));
    this.updateSavedStatsDisplay();
    this.showToast(`Saved Gen ${this.generation} Champion`);
  }

  loadBestSnake(silent: boolean = false) {
    const raw = localStorage.getItem(Game.SAVE_KEY);
    if (!raw) {
      if (!silent) this.showToast("No save found");
      return;
    }

    try {
      const data = JSON.parse(raw);
      let loadedBrain: NeuralNetwork;

      if (data.brainData) {
        loadedBrain = NeuralNetwork.restore(data.brainData);
      } else if (data.brain) {
        loadedBrain = NeuralNetwork.restore(data.brain);
      } else {
        loadedBrain = NeuralNetwork.deserialize(raw);
      }

      if (
        loadedBrain.inputNodes !== BRAIN_CONFIG.inputNodes ||
        loadedBrain.hiddenNodes !== BRAIN_CONFIG.hiddenNodes ||
        loadedBrain.outputNodes !== BRAIN_CONFIG.outputNodes
      ) {
        localStorage.removeItem(Game.SAVE_KEY);
        if (!silent) {
          this.showToast("Saved brain incompatible, starting fresh");
        }
        this.bestScore = 0;
        this.generation = 1;
        this.gensSinceImprovement = 0;
        this.updateSavedStatsDisplay();
        this.initAIGame();
        return;
      }

      this.population.length = 0;
      this.foods.length = 0;
      this.generation = data.generation || 1;
      this.bestScore = data.score || 0;
      this.bestFitness = typeof data.fitness === "number" ? data.fitness : 0;
      this.bestFitnessGen = this.generation;
      this.gensSinceImprovement = 0;

      const champion = new Snake(loadedBrain);
      this.population.push(champion);
      this.foods.push(new Food());

      for (let i = 1; i < this.popSize; i++) {
        const childBrain = loadedBrain.clone();
        childBrain.mutate(this.mutationRate);
        this.population.push(new Snake(childBrain));
        this.foods.push(new Food());
      }
      if (!silent) this.showToast("Checkpoint Loaded");
    } catch (e) {
      console.error(e);
      if (!silent) this.showToast("Load Failed");
      this.initAIGame();
    }
  }

  update() {
    this.updateAI();
  }

  updateAI() {
    let allDead = true;
    let currentBestScore = 0;
    let aliveCount = 0;
    const maxCells = GRID_WIDTH * GRID_HEIGHT;

    for (let i = 0; i < this.population.length; i++) {
      const snake = this.population[i];
      const food = this.foods[i];

      if (!snake.dead) {
        allDead = false;
        snake.think(food);
        snake.move(food);

        if (!snake.dead) {
          const head = snake.body[0];
          if (head.x === food.position.x && head.y === food.position.y) {
            const willFillBoard = snake.body.length + 1 >= maxCells;
            snake.grow();
            if (willFillBoard) {
              snake.won = true;
              snake.dead = true;
            } else {
              food.respawn(snake.bodySet);
            }
          }
        }
      }

      if (!snake.dead) aliveCount++;
      const score = this.getScore(snake);
      if (score > currentBestScore) currentBestScore = score;
    }

    if (allDead) {
      this.evolve();
    }

    if (currentBestScore > this.bestScore) this.bestScore = currentBestScore;

    const status = this.getStatus();
    if (status !== this.lastStatus) {
      if (status === "Plateaued") {
        this.showToast("Status: Plateaued (no new best in 100 gens)");
      }
      this.lastStatus = status;
    }

    const convergence =
      this.bestFitness > 0 ? this.averageFitness / this.bestFitness : 0;

    this.scoreElement.innerHTML =
      `<strong>Best Fit: ${this.bestFitness.toFixed(2)}</strong><br>` +
      `Best Score: ${this.bestScore}<br>` +
      `Gen: ${this.generation}<br>` +
      `Alive: ${aliveCount}/${this.popSize}<br>` +
      `Avg Fit: ${this.averageFitness.toFixed(2)}<br>` +
      `Stale: ${this.gensSinceImprovement}<br>` +
      `Mut: ${(this.currentMutationRate * 100).toFixed(0)}%<br>` +
      `Convergence: ${convergence.toFixed(2)}`;
  }

  evolve() {
    let totalScore = 0;
    let totalFitness = 0;
    let genBestFitness = -Infinity;
    let bestFitnessSnake: Snake | null = null;

    for (const snake of this.population) {
      snake.calculateFitness();
      totalScore += this.getScore(snake);
      totalFitness += snake.fitness;
      if (snake.fitness > genBestFitness) {
        genBestFitness = snake.fitness;
        bestFitnessSnake = snake;
      }
    }

    this.averageScore = totalScore / this.population.length;
    this.averageFitness = totalFitness / this.population.length;

    if (genBestFitness > this.bestFitness) {
      this.bestFitness = genBestFitness;
      this.bestFitnessGen = this.generation;
      if (bestFitnessSnake) {
        this.saveBestSnake(bestFitnessSnake);
      }
    }

    const convergence =
      this.bestFitness > 0 ? this.averageFitness / this.bestFitness : 0;

    this.currentMutationRate = this.mutationRate;

    if (this.gensSinceImprovement > 100) {
      this.currentMutationRate = 0.2;
    } else if (this.gensSinceImprovement > 50) {
      this.currentMutationRate = 0.1;
    } else if (this.gensSinceImprovement > 20 && convergence > 0.6) {
      this.currentMutationRate = 0.1;
    }

    const newPop: Snake[] = [];
    const eliteCount = Math.min(5, this.popSize);
    const elites = this.pickTopK(eliteCount);
    for (const elite of elites) {
      newPop.push(new Snake(elite.brain));
    }

    let immigrantRate = 0.05;
    if (this.gensSinceImprovement > 100) immigrantRate = 0.2;
    else if (this.gensSinceImprovement > 50) immigrantRate = 0.1;

    const immigrantCount = Math.min(
      Math.floor(this.popSize * immigrantRate),
      this.popSize - newPop.length,
    );
    for (let i = 0; i < immigrantCount; i++) {
      newPop.push(new Snake());
    }

    const tournamentSize = this.gensSinceImprovement > 50 ? 3 : 5;
    for (let i = newPop.length; i < this.popSize; i++) {
      const parentA = this.selectParentTournament(tournamentSize);
      const parentB = this.selectParentTournament(tournamentSize);
      const childBrain = parentA.brain.crossover(parentB.brain);
      childBrain.mutate(this.currentMutationRate);
      newPop.push(new Snake(childBrain));
    }

    this.population.length = 0;
    for (const snake of newPop) {
      this.population.push(snake);
    }
    this.resetFoods();
    this.generation++;
    this.gensSinceImprovement = this.generation - this.bestFitnessGen;
  }

  private selectParentTournament(k: number): Snake {
    const kk = Math.max(1, Math.floor(k));
    let best =
      this.population[Math.floor(Math.random() * this.population.length)];
    for (let i = 1; i < kk; i++) {
      const contender =
        this.population[Math.floor(Math.random() * this.population.length)];
      if (contender.fitness > best.fitness) best = contender;
    }
    return best;
  }

  private resetFoods() {
    if (this.foods.length !== this.popSize) {
      this.foods.length = 0;
      for (let i = 0; i < this.popSize; i++) {
        this.foods.push(new Food());
      }
      return;
    }
    for (let i = 0; i < this.foods.length; i++) {
      this.foods[i].randomize();
    }
  }

  private pickTopK(k: number): Snake[] {
    if (k <= 0) return [];
    const top: Snake[] = [];

    for (const snake of this.population) {
      if (top.length === 0) {
        top.push(snake);
        continue;
      }

      if (top.length < k) {
        let insertAt = top.length;
        while (insertAt > 0 && snake.fitness > top[insertAt - 1].fitness) {
          insertAt--;
        }
        top.splice(insertAt, 0, snake);
        continue;
      }

      if (snake.fitness <= top[top.length - 1].fitness) continue;

      let insertAt = top.length - 1;
      while (insertAt > 0 && snake.fitness > top[insertAt - 1].fitness) {
        insertAt--;
      }
      top.splice(insertAt, 0, snake);
      top.pop();
    }

    return top;
  }

  draw() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    const champion = this.population[0];
    if (champion) {
      champion.calculateFitness();
      this.currentScoreElement.textContent = `Score: ${champion.score} | Fitness: ${champion.fitness.toFixed(2)}`;
    }

    if (this.showAll) {
      for (let i = 0; i < this.population.length; i++) {
        if (!this.population[i].dead) {
          this.population[i].draw(
            this.ctx,
            TILE_SIZE,
            "rgba(76, 175, 80, 0.3)",
          );
          const f = this.foods[i];
          this.ctx.fillStyle = "rgba(244, 67, 54, 0.3)";
          this.ctx.fillRect(
            f.position.x * TILE_SIZE,
            f.position.y * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
          );
        }
      }
      return;
    }

    if (champion && !champion.dead) {
      champion.draw(this.ctx, TILE_SIZE, "#4caf50");
      const f = this.foods[0];
      this.ctx.fillStyle = "#f44336";
      this.ctx.fillRect(
        f.position.x * TILE_SIZE,
        f.position.y * TILE_SIZE,
        TILE_SIZE,
        TILE_SIZE,
      );
    }
  }

  loop(currentTime: number) {
    window.requestAnimationFrame(this.loop);

    const isChampionDead =
      this.population.length > 0 && this.population[0].dead;
    const shouldTurbo = this.turboMode || (!this.showAll && isChampionDead);

    if (shouldTurbo) {
      const start = performance.now();
      while (performance.now() - start < 12) {
        this.update();
        if (
          !this.turboMode &&
          !this.showAll &&
          this.population.length > 0 &&
          !this.population[0].dead
        ) {
          break;
        }
      }
      this.draw();
    } else {
      const secondsSinceLastRender = (currentTime - this.lastTime) / 1000;
      if (secondsSinceLastRender < 1 / this.targetFPS) return;

      this.lastTime = currentTime;

      this.update();
      this.draw();
    }
  }
}
