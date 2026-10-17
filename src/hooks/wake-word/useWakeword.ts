import { requestRecordingPermissionsAsync, useAudioStream } from "expo-audio";
import { onSamples } from "./detector";

export function useWakeWord(onDetected: () => void) {
  const { stream, isStreaming } = useAudioStream({
    sampleRate: 16000,     // request 16 kHz; check buffer.sampleRate
    channels: 1,
    encoding: "float32",   // samples are already in [-1, 1]
    onBuffer: (buffer) => onSamples(new Float32Array(buffer.data), onDetected),
  });

  const start = async () => {
    const { granted } = await requestRecordingPermissionsAsync();
    if (granted) await stream.start();
  };

  return { start, stop: () => stream.stop(), isStreaming };
}