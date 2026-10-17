
import { Asset } from "expo-asset";
import { InferenceSession } from "onnxruntime-react-native";

async function load(mod: number) {
  const asset = await Asset.fromModule(mod).downloadAsync();
  return InferenceSession.create(asset.localUri!);
}

export async function loadSessions() {
  const [melspectrogram, embedding, wakeWord] = await Promise.all([
    load(require("../../../assets/models/melspectrogram.onnx")),
    load(require("../../../assets/models/embedding_model.onnx")),
    load(require("../../../assets/models/hey_outspoken.onnx")),
  ]);
  return { melspectrogram, embedding, wakeWord };
}