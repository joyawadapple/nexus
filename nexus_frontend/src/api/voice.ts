const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function transcribeAudio(blob: Blob): Promise<string> {
  const form = new FormData();
  form.append("file", blob, "recording.webm");

  const resp = await fetch(`${BASE_URL}/voice/transcribe`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Transcription failed (${resp.status})`);
  }

  const data = await resp.json();
  return data.transcript as string;
}

export async function synthesizeText(text: string): Promise<Blob> {
  const resp = await fetch(`${BASE_URL}/voice/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Synthesis failed (${resp.status})`);
  }

  return resp.blob();
}
