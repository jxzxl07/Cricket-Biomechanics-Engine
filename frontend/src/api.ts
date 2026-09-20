import type { Analysis, CameraAngle, Mode } from "./types";

const DEFAULT_API_BASE = import.meta.env.DEV ? "http://localhost:8000" : window.location.origin;
const API_BASE = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE).replace(/\/$/, "");

export async function analyzeClip(file: File, mode: Mode, angle: CameraAngle, useAiCoach: boolean): Promise<Analysis> {
  const body = new FormData();
  body.append("file", file);
  const query = new URLSearchParams({ mode, camera_angle: angle, use_ai_coach: String(useAiCoach) });
  const response = await fetch(`${API_BASE}/api/v1/analyze?${query}`, { method: "POST", body });
  if (!response.ok) {
    let message = "Analysis failed. Please try another clip.";
    try { message = (await response.json()).detail || message; } catch { /* response was not JSON */ }
    throw new Error(message);
  }
  return response.json();
}
