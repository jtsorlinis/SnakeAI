import "./style.css";
import { ensureTfjsBackend } from "./ConvDQN";
import { Game } from "./Game";
import { enableWakeLock } from "./utils/wakeLock";

async function bootstrap(): Promise<void> {
  await ensureTfjsBackend();
  enableWakeLock();
  const game = new Game();
  await game.initialize();
}

void bootstrap();
