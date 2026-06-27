# Using the trained models: predict_bowling(), predict_batting(), load_model(), save_model()

import joblib

from config import DATA_DIR
from vision.features import (
    batting_feature_vector,
    bowling_feature_vector,
    extract_batting_features_from_data,
    extract_bowling_features_from_data,
    get_person_detection_summary,
    load_landmark_json,
)


CONFIDENCE_THRESHOLD = 0.65


def get_model_path(mode):
    return DATA_DIR / "models" / f"{mode}.joblib"


def load_model_package(mode):
    model_path = get_model_path(mode)

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained {mode} model found at {model_path}. "
            "Train the model first."
        )

    return joblib.load(model_path)


def extract_features_for_mode(landmark_data, mode):
    if mode == "bowling":
        features = extract_bowling_features_from_data(landmark_data)

    elif mode == "batting":
        features = extract_batting_features_from_data(landmark_data)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return features


def feature_vector_for_mode(features, mode):
    if mode == "bowling":
        return bowling_feature_vector(features)

    if mode == "batting":
        return batting_feature_vector(features)

    raise ValueError(f"Unknown mode: {mode}")


def get_top_feature_importances(model_package, count=2):
    model = model_package["model"]
    feature_names = model_package["feature_names"]

    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        return []

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "feature": feature_name,
            "importance": float(importance),
        }
        for feature_name, importance in ranked[:count]
    ]


def build_blocked_result(
    *,
    mode,
    predicted_label,
    display_label,
    message,
    features,
    person_detection,
    blocked_reason,
    blocked_by_legality=False,
):
    features = dict(features or {})
    features["person_detection"] = person_detection

    return {
        "mode": mode,
        "predicted_label": predicted_label,
        "display_label": display_label,
        "confidence": 0.0,
        "is_low_confidence": False,
        "threshold": CONFIDENCE_THRESHOLD,
        "probabilities": {},
        "features": features,
        "top_features": [],
        "blocked_by_precheck": True,
        "blocked_by_legality": blocked_by_legality,
        "blocked_reason": blocked_reason,
        "message": message,
        "person_detection": person_detection,
    }


def predict_from_landmarks(landmarks_path, mode):
    landmark_data = load_landmark_json(landmarks_path)
    person_detection = get_person_detection_summary(landmark_data)

    if not person_detection["has_enough_person"]:
        return build_blocked_result(
            mode=mode,
            predicted_label="no_person_detected",
            display_label="No Person Detected",
            message="No person was detected during the clip, so classification was skipped.",
            features={
                "action_detected": False,
                "action_status": "No person detected during clip",
                "should_classify": False,
                "classification_block_reason": "no_person",
            },
            person_detection=person_detection,
            blocked_reason="no_person",
        )

    features = extract_features_for_mode(landmark_data, mode)

    if not features.get("action_detected", True):
        if mode == "bowling":
            predicted_label = "no_bowling_action"
            display_label = "No Bowling Action Detected"
            message = "No bowling action was detected during the clip, so classification was skipped."
        else:
            predicted_label = "no_batting_shot"
            display_label = "No Batting Shot Detected"
            message = "No batting shot was detected during the clip, so classification was skipped."

        return build_blocked_result(
            mode=mode,
            predicted_label=predicted_label,
            display_label=display_label,
            message=message,
            features=features,
            person_detection=person_detection,
            blocked_reason="no_action",
        )

    if mode == "bowling" and not features.get("should_classify", True):
        return build_blocked_result(
            mode=mode,
            predicted_label="illegal_delivery",
            display_label="Illegal Delivery",
            message="Illegal delivery detected. Bowling type classification was skipped.",
            features=features,
            person_detection=person_detection,
            blocked_reason="illegal_delivery",
            blocked_by_legality=True,
        )

    model_package = load_model_package(mode)
    model = model_package["model"]
    vector = feature_vector_for_mode(features, mode)

    probabilities = model.predict_proba([vector])[0]
    classes = model.classes_

    best_index = int(probabilities.argmax())
    predicted_label = str(classes[best_index])
    confidence = float(probabilities[best_index])

    if confidence < CONFIDENCE_THRESHOLD:
        display_label = "Unknown"
        is_low_confidence = True
    else:
        display_label = predicted_label
        is_low_confidence = False

    top_features = get_top_feature_importances(model_package)

    return {
        "mode": mode,
        "predicted_label": predicted_label,
        "display_label": display_label,
        "confidence": confidence,
        "is_low_confidence": is_low_confidence,
        "threshold": CONFIDENCE_THRESHOLD,
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        },
        "features": features,
        "top_features": top_features,
        "blocked_by_precheck": False,
        "blocked_by_legality": False,
        "blocked_reason": None,
        "message": None,
        "person_detection": person_detection,
    }
