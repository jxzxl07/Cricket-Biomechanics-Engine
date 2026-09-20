"""Evidence-grounded coaching summaries with an optional OpenAI enhancement.

The classifier decides the action; the coach only explains measured evidence.
Deterministic feedback is always available, so the product works with no API key
and keeps working when OpenAI fails. Enhanced responses are rejected when they
claim things the measurements do not support.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
from pydantic import BaseModel, Field

SAFETY_DISCLAIMER = (
    "Single-camera pose estimates are approximate. Coaching aid only; "
    "not medical, officiating, or ball-tracking advice."
)

# Claims the product never makes, regardless of what a language model produces.
BANNED_CLAIMS = {
    "bowling legality": ("illegal", "legality", "chucking", "chuck", "no-ball", "no ball", "15 degree", "law 21"),
    "medical diagnosis": ("injury", "injured", "diagnos", "strain", "tear", "torn", "tendon", "ligament", "pain"),
    "ball tracking": ("km/h", "kph", "mph", "ball speed", "speed gun", "ball trajectory", "swing of the ball"),
}

# Tokens that tie a claim back to a measured value or a replay timestamp.
GROUNDING_TOKENS = (
    "peak", "timing", "rotation", "knee", "head", "finish", "release", "lean",
    "elbow", "wrist", "reach", "height", "duration", "speed", "setup", "downswing",
    "contact", "stride", "gather", "follow-through", "quality",
)


class Improvement(BaseModel):
    title: str
    evidence: str
    cue: str


class CoachResponse(BaseModel):
    summary: str
    strengths: list[str] = Field(max_length=3)
    improvements: list[Improvement] = Field(max_length=3)
    drill: str
    uncertainty: str
    disclaimer: str


def _value(features: dict, key: str, digits: int = 1) -> str:
    value = features.get(key)
    return "not measured" if value is None else str(round(float(value), digits))


def rules_coach(mode: str, label: str, features: dict, reason: str | None = None) -> dict:
    """Deterministic, metric-grounded feedback used whenever the AI coach is off or fails."""
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
            "drill": "Record one natural action at normal speed from 4-6 metres away.",
            "uncertainty": "No movement measurements were trustworthy enough to coach from.",
            "disclaimer": SAFETY_DISCLAIMER,
            "note": reason,
        }

    if mode == "batting":
        rotation = _value(features, "shoulder_rotation_range")
        head_drop = _value(features, "head_drop", 2)
        knee = _value(features, "max_knee_bend")
        summary = (
            f"The movement trace reads as a {label.lower()}-type action, with {rotation}° of shoulder rotation "
            f"and a {head_drop} torso-length head drop."
        )
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
        drill = "Do 3 x 8 shadow swings with a one-second freeze at the expected contact point."
        uncertainty = (
            "Shot naming comes from an experimental model and is not reliable; "
            "the timing and rotation numbers above are the parts to trust."
        )
    else:
        release = _value(features, "release_height", 2)
        lean = _value(features, "torso_lean_at_release")
        rotation = _value(features, "shoulder_rotation_range")
        summary = (
            f"The movement trace reads as a broad {label.lower()} action, with release height about "
            f"{release} torso lengths and {lean}° of torso lean at release."
        )
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
        drill = "Mark a straight landing channel and bowl 3 x 6 deliveries at 70% effort, holding the finish for two seconds."
        uncertainty = (
            "Pace versus spin is a transparent pose estimate, not a trained bowling model. "
            "No judgement is made about bowling legality."
        )

    return {
        "provider": "movement_engine",
        "enhanced": False,
        "summary": summary,
        "strengths": strengths,
        "improvements": improvements,
        "drill": drill,
        "uncertainty": uncertainty,
        "disclaimer": SAFETY_DISCLAIMER,
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


def find_unsupported_claims(text: str) -> list[str]:
    """Return the policy violations in a piece of coaching text."""
    lowered = text.lower()
    violations = []
    for category, phrases in BANNED_CLAIMS.items():
        if any(phrase in lowered for phrase in phrases):
            violations.append(category)
    return violations


def _is_grounded(evidence: str) -> bool:
    lowered = evidence.lower()
    return any(token in lowered for token in GROUNDING_TOKENS) or any(
        character.isdigit() for character in lowered
    )


def validate_coach_response(parsed: CoachResponse) -> list[str]:
    problems = find_unsupported_claims(parsed.summary + " " + parsed.drill + " " + parsed.uncertainty)
    for item in parsed.improvements:
        problems.extend(find_unsupported_claims(item.title + " " + item.evidence + " " + item.cue))
        if not _is_grounded(item.evidence):
            problems.append("improvement without measured evidence")
    for strength in parsed.strengths:
        problems.extend(find_unsupported_claims(strength))
    return problems


def enhanced_coach(
    video_path: Path,
    mode: str,
    camera_angle: str,
    classification: dict,
    features: dict,
    phase_frames: list[int],
) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    label = classification["display_label"]
    if not api_key:
        return rules_coach(mode, label, features, "Enhanced coach is available when OPENAI_API_KEY is configured.")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        benchmark = classification.get("model", {}).get("benchmark", {})
        prompt = (
            "You are a careful cricket technique coach. Assess only visible, evidenced movement. "
            "Never diagnose injury, judge bowling legality, invent ball outcome or ball speed, or treat "
            "2D pose as laboratory data. Cite a measured value or a supplied phase for every claim. "
            f"Mode: {mode}. Camera angle: {camera_angle}. "
            f"Classifier result: {label} (confidence {classification['confidence']:.2f}, "
            f"unknown={classification.get('unknown')}, experimental={classification.get('model', {}).get('experimental')}). "
            f"Model benchmark: {json.dumps(benchmark, default=str)}. "
            f"Measured features: {json.dumps(features, default=str)}. "
            "Give concise, practical feedback tied to those measurements or the supplied "
            "setup/impact-or-release/follow-through stills. State clearly what the measurements cannot show."
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
        problems = validate_coach_response(parsed)
        if problems:
            return rules_coach(
                mode,
                label,
                features,
                "Enhanced coach reply was rejected for unsupported claims: " + ", ".join(sorted(set(problems))) + ".",
            )
        return {
            "provider": "openai",
            "enhanced": True,
            **parsed.model_dump(),
            "note": "Three selected stills and the measured metrics were used.",
        }
    except Exception as error:  # The core analysis must not fail with the optional coach.
        return rules_coach(mode, label, features, f"Enhanced coach unavailable: {type(error).__name__}.")
