import "./style.css";
import { ensureTfjsBackend } from "./ConvDQN";
import { Game } from "./Game";
import { enableWakeLock } from "./utils/wakeLock";

async function bootstrap(): Promise<void> {
  await ensureTfjsBackend();
  enableWakeLock();
  new Game();
}

void bootstrap();
