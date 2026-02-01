import { Snake } from "./Snake";
import { Food } from "./Food";
import { NeuralNetwork } from "./NeuralNetwork";
import { TILE_SIZE, CANVAS_WIDTH, CANVAS_HEIGHT } from "./types";

export class Game {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;

  // AI Mode
  population: Snake[] = [];
  foods: Food[] = [];
  generation: number = 1;
  bestScore: number = 0;

  scoreElement: HTMLElement;
  currentScoreElement: HTMLElement;
  lastTime: number;
  targetFPS: number = 60;
  turboMode: boolean = false;
  showAll: boolean = false;

  // AI Settings
  popSize: number = 500;
  mutationRate: number = 0.05;
  currentMutationRate: number = 0.05;

  // Stats
  averageScore: number = 0;
  gensSinceImprovement: number = 0;
  highScoreGen: number = 1;
  lastStatus: string = "";

  // UI
  turboToggle: HTMLInputElement;
  showAllToggle: HTMLInputElement;
  resetBtn: HTMLButtonElement;
  toastElement: HTMLElement;
  toastTimeout: number | null = null;

  // Saved Stats UI
  savedStatsDiv: HTMLElement;
  savedGenSpan: HTMLElement;
  savedScoreSpan: HTMLElement;
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
    this.savedDateSpan = document.getElementById("savedDate")!;

    this.canvas.width = CANVAS_WIDTH;
    this.canvas.height = CANVAS_HEIGHT;

    this.lastTime = 0;

    if (localStorage.getItem("bestSnakeBrain")) {
        this.loadBestSnake(true); // silent mode
    } else {
        this.initAIGame();
    }

    this.updateSavedStatsDisplay();
    this.setupInput();
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  showToast(message: string) {
    if (this.toastTimeout) {
      clearTimeout(this.toastTimeout);
    }
    this.toastElement.innerText = message;
    this.toastElement.classList.add("visible");

    this.toastTimeout = window.setTimeout(() => {
      this.toastElement.classList.remove("visible");
    }, 2500);
  }

  initAIGame() {
    this.population = [];
    this.foods = [];
    for (let i = 0; i < this.popSize; i++) {
      this.population.push(new Snake());
      this.foods.push(new Food());
    }
    this.targetFPS = 60; // Smooth AI visualization
  }

  setupInput() {
    this.turboToggle.addEventListener("change", () => {
      this.turboMode = this.turboToggle.checked;
    });

    this.showAllToggle.addEventListener("change", () => {
      this.showAll = this.showAllToggle.checked;
    });

    this.resetBtn.addEventListener("click", () => {
        if(confirm("Are you sure you want to delete the best snake and restart?")) {
            this.resetBestSnake();
        }
    });
  }

  resetBestSnake() {
      localStorage.removeItem("bestSnakeBrain");
      this.bestScore = 0;
      this.generation = 1;
      this.highScoreGen = 1;
      this.gensSinceImprovement = 0;
      this.updateSavedStatsDisplay();
      this.initAIGame();
      this.showToast("Progress Reset");
  }

  updateSavedStatsDisplay() {
    const raw = localStorage.getItem("bestSnakeBrain");
    if (!raw) {
      this.savedStatsDiv.style.display = "none";
      return;
    }

    this.savedStatsDiv.style.display = "block";

    try {
      const data = JSON.parse(raw);
      if (data.brainData) {
        this.savedGenSpan.innerText = data.generation;
        this.savedScoreSpan.innerText = data.score;
        this.savedDateSpan.innerText = data.date;
      } else {
        this.savedGenSpan.innerText = "?";
        this.savedScoreSpan.innerText = "?";
        this.savedDateSpan.innerText = "Old Save Format";
      }
    } catch (e) {
      this.savedStatsDiv.style.display = "none";
    }
  }

    saveBestSnake(targetSnake?: Snake) {

      if (this.population.length === 0 && !targetSnake) return;

      

      const champion = targetSnake || this.population[0];

      

      const saveObject = {

          brainData: champion.brain,

          generation: this.generation,

          score: this.bestScore,

          date: new Date().toLocaleString()

      };

  

      localStorage.setItem('bestSnakeBrain', JSON.stringify(saveObject));

      this.updateSavedStatsDisplay();

      this.showToast(`Saved Gen ${this.generation} Champion`);

    }

  loadBestSnake(silent: boolean = false) {
    const raw = localStorage.getItem("bestSnakeBrain");
    if (!raw) {
      if (!silent) this.showToast("No save found");
      return;
    }

    try {
      const data = JSON.parse(raw);
      let loadedBrain: NeuralNetwork;

      if (data.brainData) {
        if (typeof data.brainData === "string") {
          loadedBrain = NeuralNetwork.deserialize(data.brainData);
        } else {
          loadedBrain = NeuralNetwork.restore(data.brainData);
        }
      } else {
        loadedBrain = NeuralNetwork.deserialize(raw);
      }

      if (
        loadedBrain.inputNodes !== 27 ||
        loadedBrain.hiddenNodes !== 64 ||
        loadedBrain.outputNodes !== 3
      ) {
        localStorage.removeItem("bestSnakeBrain");
        if (!silent) {
          this.showToast("Saved brain incompatible, starting fresh");
        }
        this.bestScore = 0;
        this.generation = 1;
        this.highScoreGen = 1;
        this.gensSinceImprovement = 0;
        this.updateSavedStatsDisplay();
        this.initAIGame();
        return;
      }

      this.population = [];
      this.foods = [];
      this.generation = data.generation || 1;
      this.bestScore = data.score || 0;
      this.highScoreGen = this.generation; // Sync high score gen
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
      this.showToast("Load Failed");
    }
  }

  update() {
    this.updateAI();
  }

  updateAI() {
    let allDead = true;

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
            snake.grow();
            food.respawn(snake.bodySet);
            snake.foodProgress = 0;
          }
        }
      }
    }

    if (allDead) {
      this.evolve();
    }

    let currentBest = 0;
    let bestSnakeInGen: Snake | null = null;
    let aliveCount = 0;
    for(let s of this.population) {
        if(!s.dead) aliveCount++;
        const score = s.body.length - 3;
        if(score > currentBest) {
            currentBest = score;
            bestSnakeInGen = s;
        }
    }
    if (currentBest > this.bestScore && bestSnakeInGen) {
      this.bestScore = currentBest;
      this.highScoreGen = this.generation;
      this.gensSinceImprovement = 0;
      this.saveBestSnake(bestSnakeInGen);
    }

    let status = "Improving";
    if (this.bestScore <= 0) {
      status = "Bootstrapping";
    } else if (this.gensSinceImprovement >= 100) {
      status = "Plateaued";
    } else if (this.gensSinceImprovement >= 50) {
      status = "Stuck";
    } else if (this.gensSinceImprovement >= 20) {
      status = "Slowing";
    }

    if (status !== this.lastStatus) {
      if (status === "Plateaued") this.showToast("Status: Plateaued (no new best in 100 gens)");
      this.lastStatus = status;
    }

    this.scoreElement.innerHTML = `<strong>Best: ${this.bestScore}</strong><br>Gen: ${this.generation}<br>Alive: ${aliveCount}/${this.popSize}<br>Avg: ${this.averageScore.toFixed(1)}<br>Stale: ${this.gensSinceImprovement}<br>Mut: ${(this.currentMutationRate * 100).toFixed(0)}%<br>Status: ${status}`;
  }

  evolve() {
    let totalScore = 0;

    for (const snake of this.population) {
      snake.calculateFitness();
      totalScore += snake.body.length - 3;
    }

    this.averageScore = totalScore / this.population.length;

    // Dynamic Mutation Logic
    const convergence = this.bestScore > 0 ? this.averageScore / this.bestScore : 0;
    
    this.currentMutationRate = this.mutationRate; // Reset to base (0.05)
    
    if (this.gensSinceImprovement > 100) {
        this.currentMutationRate = 0.2; // Panic Mode
    } else if (this.gensSinceImprovement > 50) {
        this.currentMutationRate = 0.1; // Hard Stuck
    } else if (this.gensSinceImprovement > 20 && convergence > 0.6) {
        this.currentMutationRate = 0.1; // Converged early
    }

    const newPop: Snake[] = [];
    // Keep a small elite set unchanged each generation to avoid losing good behaviors.
    const elites = [...this.population].sort((a, b) => b.fitness - a.fitness);
    const eliteCount = Math.min(5, this.popSize);
    for (let i = 0; i < eliteCount; i++) {
      newPop.push(new Snake(elites[i].brain));
    }

    // Add random "immigrants" to inject diversity and break local minima.
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

    // Lower selection pressure when stuck to preserve diversity.
    const tournamentSize = this.gensSinceImprovement > 50 ? 3 : 5;
    for (let i = newPop.length; i < this.popSize; i++) {
      const parentA = this.selectParentTournament(tournamentSize);
      const parentB = this.selectParentTournament(tournamentSize);
      const childBrain = parentA.brain.crossover(parentB.brain);
      childBrain.mutate(this.currentMutationRate);
      newPop.push(new Snake(childBrain));
    }

    this.population = newPop;
    this.foods = new Array(this.popSize).fill(0).map(() => new Food());
    this.generation++;
    this.gensSinceImprovement = this.generation - this.highScoreGen;
  }

  selectParentTournament(k: number): Snake {
    if (this.population.length === 0) {
      throw new Error("Population is empty");
    }
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

  selectParent(sumFitness: number): Snake {
    if (sumFitness <= 0) {
      return this.population[Math.floor(Math.random() * this.population.length)];
    }
    const r = Math.random() * sumFitness;
    let runningSum = 0;
    for (const snake of this.population) {
      runningSum += snake.fitness;
      if (runningSum > r) {
        return snake;
      }
    }
    return this.population[this.population.length - 1];
  }

  draw() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    const champion = this.population[0];
    if (champion) {
      const champScore = champion.body.length - 3;
      this.currentScoreElement.textContent = `Score: ${champScore}`;
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
    } else {
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
  }

  loop(currentTime: number) {
    window.requestAnimationFrame(this.loop);

    const isChampionDead =
      this.population.length > 0 && this.population[0].dead;
    const shouldTurbo = this.turboMode || (!this.showAll && isChampionDead);

    if (shouldTurbo) {
      const start = performance.now();
      // Run for up to 12ms per frame to maintain 60fps UI responsiveness
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
