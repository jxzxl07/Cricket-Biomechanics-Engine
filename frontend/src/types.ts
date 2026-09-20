export type Mode = "batting" | "bowling";
export type CameraAngle = "side_on" | "front_on" | "rear" | "unknown";

export type Landmark = [number, number, number, number];
export interface TimelineFrame { frame: number; timestamp_ms: number; landmarks: Landmark[] }
export interface Metric { key: string; label: string; value: number | null; unit: string; description: string; timestamp_ms: number | null }
export interface Phase { id: string; label: string; frame: number; timestamp_ms: number }
export interface Improvement { title: string; evidence: string; cue: string }
export interface Analysis {
  status: "complete" | "needs_better_clip";
  mode: Mode;
  camera_angle: CameraAngle;
  classification: {
    label: string;
    display_label: string;
    confidence: number;
    low_confidence: boolean;
    probabilities: Record<string, number>;
    model: { id: string; kind: string; experimental: boolean; note: string };
  };
  metrics: Metric[];
  phases: Phase[];
  timeline: { fps: number; width: number; height: number; total_frames: number; sample_step: number; frames: TimelineFrame[] };
  quality: { score: number; pose_coverage: number; notes: string[] };
  coach: { provider: string; enhanced: boolean; summary: string; strengths: string[]; improvements: Improvement[]; drill: string; disclaimer: string; note?: string };
  privacy: string;
}
