import "./style.css";
import { ensureTfjsBackend } from "./ConvDQN";
import { Game } from "./Game";

async function bootstrap(): Promise<void> {
  await ensureTfjsBackend();
  new Game();
}

void bootstrap();
