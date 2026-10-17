import { InferenceSession, Tensor } from "onnxruntime-react-native";

interface Sessions {
  melspectrogram: InferenceSession;
  embedding: InferenceSession;
  wakeWord: InferenceSession;
}

/**
 * The wake word pipeline: audio -> mel frames -> embeddings -> score.
 * It keeps a short history between calls, so give it every 1280-sample frame, in order.
 */
export function createPipeline(sessions: Sessions) {
  let melFrames: Float32Array[] = [];                                        // 32 values per frame
  const embeddings = Array.from({ length: 16 }, () => new Float32Array(96)); // the last 16 embeddings

  // ONNX Runtime reuses its output buffers, so copy whatever you keep.
  async function run(session: InferenceSession, data: Float32Array, dims: number[]) {
    const out = await session.run({ [session.inputNames[0]]: new Tensor("float32", data, dims) });
    return Float32Array.from(out[session.outputNames[0]].data as Float32Array);
  }

  return async function runInference(frame: Float32Array): Promise<number> {
    // 1. 1280 audio samples -> 5 mel frames, rescaled the way the embedding model expects
    const mel = await run(sessions.melspectrogram, frame, [1, 1280]);
    for (let i = 0; i < 5; i++) {
      melFrames.push(mel.slice(i * 32, (i + 1) * 32).map((v) => v / 10 + 2));
    }

    let score = 0;
    // 2. every 76 mel frames -> one embedding, then slide the window by 8 frames
    while (melFrames.length >= 76) {
      const window = new Float32Array(76 * 32);
      melFrames.slice(0, 76).forEach((f, i) => window.set(f, i * 32));
      embeddings.shift();
      embeddings.push(await run(sessions.embedding, window, [1, 76, 32, 1]));
      melFrames = melFrames.slice(8);

      // 3. the last 16 embeddings -> wake word score (0 to 1)
      const history = new Float32Array(16 * 96);
      embeddings.forEach((e, i) => history.set(e, i * 96));
      score = (await run(sessions.wakeWord, history, [1, 16, 96]))[0];
    }
    return score;
  };
}