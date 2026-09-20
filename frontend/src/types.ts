export type Mode = "batting" | "bowling";
export type CameraAngle = "side_on" | "front_on" | "rear" | "unknown";

export type Landmark = [number, number, number, number];
export interface TimelineFrame { frame: number; timestamp_ms: number; landmarks: Landmark[] }
export interface Phase { id: string; label: string; frame: number; timestamp_ms: number }
export interface Metric { key: string; label: string; value: number | null; unit: string; description: string; frame: number | null; timestamp_ms: number | null }
export interface Improvement { title: string; evidence: string; cue: string }
export interface Alternative { label: string; display_label: string; probability: number }
export interface QualityCheck { id: string; label: string; passed: boolean; detail: string }
export interface Coach {
  provider: string;
  enhanced: boolean;
  summary: string;
  strengths: string[];
  improvements: Improvement[];
  drill: string;
  uncertainty: string;
  disclaimer: string;
  note?: string;
}
export interface Analysis {
  schema_version: string;
  status: "complete" | "needs_better_clip";
  mode: Mode;
  camera_angle: CameraAngle;
  video: { width: number; height: number; fps: number; frames: number; duration_ms: number; orientation: "portrait" | "landscape" };
  classification: {
    label: string;
    display_label: string;
    confidence: number;
    raw_confidence: number;
    unknown: boolean;
    low_confidence: boolean;
    top_alternatives: Alternative[];
    probabilities: Record<string, number>;
    model: {
      id: string;
      version: string;
      kind: string;
      experimental: boolean;
      note: string;
      confidence_cap: number;
      benchmark: {
        status?: string;
        top1_accuracy?: number;
        top2_accuracy?: number;
        clips?: number;
        dataset?: string;
        interpretation?: string;
        [key: string]: unknown;
      };
      limitations: string[];
    };
  };
  metrics: Metric[];
  features: Record<string, number | string | null>;
  phases: Phase[];
  timeline: { fps: number; width: number; height: number; total_frames: number; sample_step: number; frames: TimelineFrame[] };
  quality: { score: number; pose_coverage: number; checks: QualityCheck[]; warnings: string[]; notes: string[] };
  warnings: string[];
  coach: Coach;
  timings: { pose_ms: number; features_ms: number; classification_ms: number; coaching_ms: number; total_ms: number };
  privacy: string;
}
