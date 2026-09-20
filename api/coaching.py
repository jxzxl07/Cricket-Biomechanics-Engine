"""Evidence-grounded coaching summaries with an optional OpenAI enhancement."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
from pydantic import BaseModel, Field


class Improvement(BaseModel):
    title: str
    evidence: str
    cue: str


class CoachResponse(BaseModel):
    summary: str
    strengths: list[str] = Field(max_length=3)
    improvements: list[Improvement] = Field(max_length=3)
    drill: str
    disclaimer: str


def _value(features: dict, key: str, digits: int = 1) -> str:
    value = features.get(key)
    return "not measured" if value is None else str(round(float(value), digits))


def rules_coach(mode: str, label: str, features: dict, reason: str | None = None) -> dict:
    if not features.get("action_detected"):
        return {
            "provider": "movement_engine",
            "enhanced": False,
            "summary": "The clip did not contain a clear, fully visible action, so technique feedback would be guesswork.",
            "strengths": [],
            "improvements": [{
                "title": "Re-record the action",
                "evidence": "The pose/action quality gate did not pass.",
                "cue": "Keep your full body in frame, use a steady side-on camera, and leave space above the release or follow-through.",
            }],
            "drill": "Record one natural action at normal speed from 4–6 metres away.",
            "disclaimer": "Coaching aid only; not a medical or officiating assessment.",
            "note": reason,
        }

    if mode == "batting":
        rotation = _value(features, "shoulder_rotation_range")
        head_drop = _value(features, "head_drop", 2)
        knee = _value(features, "max_knee_bend")
        summary = f"The clip reads as {label}. The movement trace shows {rotation}° of shoulder rotation and a {head_drop} torso-length head drop."
        strengths = [
            f"A clear hand-speed peak was captured at {_value(features, 'peak_speed_timing', 2)} through the action.",
            f"The lower body contributed up to {knee}° of knee flexion.",
        ]
        improvements = [{
            "title": "Keep the head quiet through contact",
            "evidence": f"Estimated head drop: {head_drop} torso lengths.",
            "cue": "Pick a contact point and keep your eyes level until the follow-through begins.",
        }, {
            "title": "Sequence, then accelerate",
            "evidence": f"Peak-speed timing: {_value(features, 'peak_speed_timing', 2)} of the detected swing.",
            "cue": "Let the front foot and torso lead; send the hands through last.",
        }]
        drill = "Do 3 × 8 shadow swings with a one-second freeze at the expected contact point."
    else:
        release = _value(features, "release_height", 2)
        lean = _value(features, "torso_lean_at_release")
        rotation = _value(features, "shoulder_rotation_range")
        summary = f"The clip reads as {label}. Release height was about {release} torso lengths with {lean}° of torso lean at release."
        strengths = [
            f"The action produced {_value(features, 'peak_wrist_speed')} body-lengths/s of peak wrist speed.",
            f"Shoulder rotation covered roughly {rotation}° across the detected action.",
        ]
        improvements = [{
            "title": "Own the release position",
            "evidence": f"Estimated release height: {release} torso lengths.",
            "cue": "Finish tall and let the bowling shoulder travel towards the target.",
        }, {
            "title": "Make the follow-through repeatable",
            "evidence": f"Torso lean at release: {lean}°.",
            "cue": "Land balanced, then drive the back hip through the crease on the same line.",
        }]
        drill = "Mark a straight landing channel and bowl 3 × 6 deliveries at 70% effort, holding the finish for two seconds."

    return {
        "provider": "movement_engine",
        "enhanced": False,
        "summary": summary,
        "strengths": strengths,
        "improvements": improvements,
        "drill": drill,
        "disclaimer": "Single-camera pose estimates are approximate. Coaching aid only; not medical or officiating advice.",
        "note": reason,
    }


def _frame_data_urls(video_path: Path, frame_numbers: list[int]) -> list[str]:
    capture = cv2.VideoCapture(str(video_path))
    images = []
    try:
        for frame_number in frame_numbers[:3]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 960 / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if encoded:
                images.append("data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii"))
    finally:
        capture.release()
    return images


def enhanced_coach(
    video_path: Path,
    mode: str,
    camera_angle: str,
    classification: dict,
    features: dict,
    phase_frames: list[int],
) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return rules_coach(mode, classification["display_label"], features, "Enhanced coach is available when OPENAI_API_KEY is configured.")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a careful cricket technique coach. Assess only visible, evidenced movement. "
            "Do not diagnose injury, judge bowling legality, invent ball outcome, or treat 2D pose as lab data. "
            f"Mode: {mode}. Camera angle: {camera_angle}. Classifier result: {classification['display_label']}. "
            f"Measured features: {json.dumps(features, default=str)}. "
            "Give concise, practical feedback tied to measurements or the supplied setup/impact-or-release/follow-through stills."
        )
        content = [{"type": "input_text", "text": prompt}]
        content.extend({"type": "input_image", "image_url": url} for url in _frame_data_urls(video_path, phase_frames))
        response = client.responses.parse(
            model=os.getenv("OPENAI_COACH_MODEL", "gpt-4o-mini"),
            input=[{"role": "user", "content": content}],
            text_format=CoachResponse,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI returned no structured coaching response")
        return {"provider": "openai", "enhanced": True, **parsed.model_dump(), "note": "Three selected stills and the measured metrics were used."}
    except Exception as error:  # The core analysis must not fail with the optional coach.
        return rules_coach(mode, classification["display_label"], features, f"Enhanced coach unavailable: {type(error).__name__}.")
