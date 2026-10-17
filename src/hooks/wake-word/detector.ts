import { createPipeline } from "./pipeline";
import { loadSessions } from "./sessions";

const FRAME = 1280;      // 80 ms at 16 kHz
const THRESHOLD = 0.5;   // raise it if it fires too easily
const DEBOUNCE_MS = 2000;
const MAX_QUEUE = 4;     // if inference falls behind, drop the oldest frames

const pipeline = loadSessions().then(createPipeline); // the models load once, on first use

let pending = new Float32Array(0);
const queue: Float32Array[] = [];
let draining = false;
let last = 0;

/** Feed it every buffer of 16 kHz float32 samples, whatever its size. */
export function onSamples(incoming: Float32Array, onDetected: () => void) {
  // Collect samples and cut 1280-sample frames.
  const all = new Float32Array(pending.length + incoming.length);
  all.set(pending);
  all.set(incoming, pending.length);
  let i = 0;
  for (; i + FRAME <= all.length; i += FRAME) queue.push(all.slice(i, i + FRAME));
  pending = all.slice(i);
  while (queue.length > MAX_QUEUE) queue.shift();
  void drain(onDetected);
}

// Frames go through the pipeline one at a time, in order: it keeps a rolling history.
async function drain(onDetected: () => void) {
  if (draining) return;
  draining = true;
  try {
    const runInference = await pipeline;
    for (let frame = queue.shift(); frame; frame = queue.shift()) {
      const score = await runInference(frame);
      if (score >= THRESHOLD && Date.now() - last > DEBOUNCE_MS) {
        last = Date.now();
        onDetected(); // your action here
      }
    }
  } finally {
    draining = false;
  }
}