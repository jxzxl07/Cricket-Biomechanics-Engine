import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import (
    BATTING_LABELS,
    BOWLING_LABELS,
    COUNTDOWN_SECONDS,
    DATA_DIR,
    PROJECT_ROOT,
    RECORD_SECONDS,
)
from database.database import (
    save_correction,
    save_recording,
    update_recording_paths,
    update_recording_prediction,
)
from vision.camera import CameraWorker
from vision.pose import extract_landmarks_from_video
from ml.trainer import format_evaluation_summary, train_model
from ml.classifier import predict_from_landmarks
from vision.features import extract_and_save_features


POSE_MISSES_BEFORE_WARNING = 12
NO_PERSON_MESSAGE = "No person detected."


def label_to_folder(label):
    return label.strip().lower().replace(" ", "_")


def timestamp_slug():
    return time.strftime("%Y%m%d_%H%M%S")


def display_path(path):
    path = Path(path)

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_labels_for_mode(mode):
    if mode == "bowling":
        return BOWLING_LABELS

    if mode == "batting":
        return BATTING_LABELS

    return []


def humanise_feature_name(feature_name):
    return feature_name.replace("_", " ").title()


class CorrectionDialog(QDialog):
    def __init__(self, mode, prediction_result, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Correct Label")

        layout = QVBoxLayout()
        layout.setSpacing(12)

        confidence = prediction_result["confidence"] * 100
        predicted_label = prediction_result["predicted_label"]

        message = QLabel(
            "The model is unsure.\n\n"
            f"Best guess: {predicted_label}\n"
            f"Confidence: {confidence:.1f}%\n\n"
            "Choose the correct label to add this clip to the next training run."
        )
        message.setWordWrap(True)

        self.label_selector = QComboBox()
        self.label_selector.addItems(get_labels_for_mode(mode))
        self.label_selector.setMinimumHeight(40)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(message)
        layout.addWidget(self.label_selector)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def selected_label(self):
        return self.label_selector.currentText()


class EvaluationDialog(QDialog):
    def __init__(self, mode, summary, parent=None):
        super().__init__(parent)

        self.setWindowTitle(f"{mode.title()} Evaluation")
        self.resize(820, 620)

        layout = QVBoxLayout()
        layout.setSpacing(12)

        text_area = QPlainTextEdit()
        text_area.setReadOnly(True)
        text_area.setFont(QFont("Menlo", 12))
        text_area.setPlainText(summary)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout.addWidget(text_area)
        layout.addWidget(buttons)

        self.setLayout(layout)


class AnalysisPage(QWidget):
    def __init__(self, mode, title_text, starting_status, back_callback):
        super().__init__()

        self.mode = mode
        self.default_status = starting_status
        self.back_callback = back_callback
        self.camera_worker = None
        self.countdown_timer = None
        self.current_output_path = None
        self.current_recording_id = None
        self.current_prediction_result = None
        self.missed_pose_frames = 0

        self.setObjectName("page")

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(24)

        camera_panel = self.create_camera_panel()
        side_panel = self.create_side_panel(title_text, starting_status)

        main_layout.addWidget(camera_panel)
        main_layout.addWidget(side_panel, 1)

        self.setLayout(main_layout)

    def create_camera_panel(self):
        camera_panel = QFrame()
        camera_panel.setObjectName("cameraPanel")
        camera_panel.setMinimumSize(520, 760)
        camera_panel.setMaximumWidth(620)

        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QLabel("Webcam preview will appear here")
        self.camera_label.setObjectName("cameraPlaceholder")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setScaledContents(False)
        self.camera_label.setMinimumSize(520, 760)
        self.camera_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        camera_layout.addWidget(self.camera_label)
        camera_panel.setLayout(camera_layout)

        return camera_panel

    def create_side_panel(self, title_text, starting_status):
        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")

        side_layout = QVBoxLayout()
        side_layout.setSpacing(16)

        title = QLabel(title_text)
        title.setObjectName("panelTitle")

        self.status_label = QLabel(starting_status)
        self.status_label.setObjectName("statusText")
        self.status_label.setWordWrap(True)

        self.result_label = QLabel("Result: -")
        self.result_label.setObjectName("resultText")

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setObjectName("statusText")

        self.metrics_label = QLabel("Key metrics will appear after analysis.")
        self.metrics_label.setObjectName("statusText")
        self.metrics_label.setWordWrap(True)

        self.record_button = QPushButton("Record")
        self.record_button.setMinimumHeight(46)
        self.record_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_button.clicked.connect(self.start_countdown)

        self.correct_button = QPushButton("Correct Label")
        self.correct_button.setMinimumHeight(42)
        self.correct_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.correct_button.setEnabled(False)
        self.correct_button.clicked.connect(self.open_correction_dialog)

        back_button = QPushButton("Back")
        back_button.setMinimumHeight(42)
        back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        back_button.clicked.connect(self.back_callback)

        side_layout.addWidget(title)
        side_layout.addWidget(self.status_label)
        side_layout.addSpacing(12)
        side_layout.addWidget(self.result_label)
        side_layout.addWidget(self.confidence_label)
        side_layout.addWidget(self.metrics_label)
        side_layout.addSpacing(12)
        side_layout.addWidget(self.record_button)
        side_layout.addWidget(self.correct_button)
        side_layout.addStretch()
        side_layout.addWidget(back_button)

        side_panel.setLayout(side_layout)
        return side_panel

    def start_countdown(self):
        if self.camera_worker is None:
            self.status_label.setText("Camera is not running.")
            return

        self.record_button.setEnabled(False)
        self.correct_button.setEnabled(False)
        self.result_label.setText("Result: -")
        self.confidence_label.setText("Confidence: -")
        self.metrics_label.setText("Waiting to record...")

        self.countdown_number = COUNTDOWN_SECONDS
        self.status_label.setText(f"Recording starts in {self.countdown_number}...")
        self.play_beep()

        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

    def update_countdown(self):
        self.countdown_number -= 1

        if self.countdown_number > 0:
            self.status_label.setText(f"Recording starts in {self.countdown_number}...")
            self.play_beep()
            return

        self.countdown_timer.stop()
        self.start_recording()

    def start_recording(self):
        self.play_beep()

        filename = f"{self.mode}_{timestamp_slug()}.mp4"
        self.current_output_path = (
            DATA_DIR / "raw" / "predictions" / self.mode / filename
        )

        self.status_label.setText(f"Recording {RECORD_SECONDS} second clip...")
        self.camera_worker.start_recording(
            str(self.current_output_path),
            duration_seconds=RECORD_SECONDS,
        )

    def handle_recording_finished(self, output_path):
        self.play_finished_beep()

        try:
            recording_id = save_recording(
                mode=self.mode,
                purpose="prediction",
                clip_path=output_path,
                trained=False,
            )

            self.status_label.setText("Extracting landmarks...")
            landmarks_path = extract_landmarks_from_video(
                output_path,
                mode=self.mode,
            )

            features_path = extract_and_save_features(landmarks_path)

            update_recording_paths(
                recording_id,
                landmarks_path=landmarks_path,
                features_path=features_path,
            )

            self.status_label.setText("Classifying movement...")
            prediction_result = predict_from_landmarks(landmarks_path, self.mode)

            update_recording_prediction(
                recording_id,
                predicted_label=prediction_result["predicted_label"],
                confidence=prediction_result["confidence"],
            )

            database_text = f"Database row: {recording_id}"

        except Exception as error:
            self.status_label.setText(f"Save/extraction failed:\n{error}")
            self.record_button.setEnabled(True)
            return

        self.current_recording_id = recording_id
        self.current_prediction_result = prediction_result

        self.status_label.setText(
            f"Saved analysis clip:\n"
            f"{display_path(output_path)}\n\n"
            f"Saved landmarks:\n"
            f"{display_path(landmarks_path)}\n\n"
            f"Saved features:\n"
            f"{display_path(features_path)}\n\n"
            f"{database_text}"
        )
        self.show_prediction_result(prediction_result)

        self.record_button.setEnabled(True)
        can_correct_prediction = not prediction_result.get("blocked_by_precheck", False)
        self.correct_button.setEnabled(can_correct_prediction)

        if can_correct_prediction and prediction_result["is_low_confidence"]:
            QTimer.singleShot(0, self.open_correction_dialog)



    def show_prediction_result(self, prediction_result):
        display_label = prediction_result["display_label"]
        blocked_by_precheck = prediction_result.get("blocked_by_precheck", False)

        self.result_label.setText(f"Result: {display_label}")

        if blocked_by_precheck:
            self.confidence_label.setText("Model confidence: not used")
        else:
            confidence = prediction_result["confidence"] * 100
            self.confidence_label.setText(f"Confidence: {confidence:.1f}%")

        probabilities = sorted(
            prediction_result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        probability_text = "\n".join(
            f"{label}: {probability * 100:.1f}%"
            for label, probability in probabilities[:3]
        )

        top_features = prediction_result["top_features"]

        if top_features:
            feature_text = ", ".join(
                humanise_feature_name(item["feature"])
                for item in top_features
            )
        else:
            feature_text = ""

        if blocked_by_precheck:
            result_note = prediction_result.get("message") or "Classification skipped."
        elif prediction_result["is_low_confidence"]:
            result_note = (
                "Confidence is low, so this has been marked as Unknown."
            )
        else:
            result_note = "Prediction accepted."

        key_metrics_text = build_key_metrics_text(
            self.mode,
            prediction_result["features"],
        )

        sections = [result_note]

        if probability_text:
            sections.append(f"Top probabilities:\n{probability_text}")

        if feature_text:
            sections.append(f"Main factors:\n{feature_text}")

        if key_metrics_text:
            sections.append(f"Key metrics:\n{key_metrics_text}")

        self.metrics_label.setText("\n\n".join(sections))

    def open_correction_dialog(self):
        if self.current_recording_id is None or self.current_prediction_result is None:
            return

        if self.current_prediction_result.get("blocked_by_precheck", False):
            return

        dialog = CorrectionDialog(
            self.mode,
            self.current_prediction_result,
            parent=self,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        corrected_label = label_to_folder(dialog.selected_label())
        save_correction(self.current_recording_id, corrected_label)

        self.status_label.setText(
            "Correction saved for next retrain.\n\n"
            f"Correct label: {corrected_label}"
        )
        self.result_label.setText(f"Corrected: {corrected_label}")
        self.correct_button.setEnabled(False)

    def start_camera(self):
        if self.camera_worker is not None:
            return

        self.camera_worker = CameraWorker()
        self.camera_worker.frame_ready.connect(self.update_camera_frame)
        self.camera_worker.camera_error.connect(self.show_camera_error)
        self.camera_worker.pose_status.connect(self.update_pose_status)
        self.camera_worker.recording_finished.connect(self.handle_recording_finished)
        self.camera_worker.start()

    def stop_camera(self):
        self.stop_countdown()

        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker = None

    def stop_countdown(self):
        if self.countdown_timer is not None and self.countdown_timer.isActive():
            self.countdown_timer.stop()

    def update_camera_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )

        crop_x = max(0, (scaled_pixmap.width() - self.camera_label.width()) // 2)
        crop_y = max(0, (scaled_pixmap.height() - self.camera_label.height()) // 2)

        cropped_pixmap = scaled_pixmap.copy(
            crop_x,
            crop_y,
            self.camera_label.width(),
            self.camera_label.height(),
        )

        self.camera_label.setPixmap(cropped_pixmap)

    def show_camera_error(self, message):
        self.camera_label.setText(message)

    def update_pose_status(self, person_detected):
        if person_detected:
            self.missed_pose_frames = 0
            self.camera_label.setStyleSheet("")

            if self.status_label.text() == NO_PERSON_MESSAGE:
                self.status_label.setText(self.default_status)

            return

        self.missed_pose_frames += 1

        if (
            self.missed_pose_frames >= POSE_MISSES_BEFORE_WARNING
            and self.record_button.isEnabled()
        ):
            self.status_label.setText(NO_PERSON_MESSAGE)

    def play_beep(self):
        QApplication.beep()

    def play_finished_beep(self):
        QApplication.beep()
        QTimer.singleShot(180, QApplication.beep)


class TrainingPage(QWidget):
    def __init__(self, mode, title_text, starting_status, labels, back_callback):
        super().__init__()

        self.mode = mode
        self.default_status = starting_status
        self.back_callback = back_callback
        self.camera_worker = None
        self.countdown_timer = None
        self.current_label = None
        self.current_label_folder = None
        self.current_output_path = None
        self.clip_count = 0
        self.missed_pose_frames = 0
        self.keep_status_message = False

        self.setObjectName("page")

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(24)

        camera_panel = self.create_camera_panel()
        side_panel = self.create_side_panel(title_text, starting_status, labels)

        main_layout.addWidget(camera_panel)
        main_layout.addWidget(side_panel, 1)

        self.setLayout(main_layout)

    def create_camera_panel(self):
        camera_panel = QFrame()
        camera_panel.setObjectName("cameraPanel")
        camera_panel.setMinimumSize(520, 760)
        camera_panel.setMaximumWidth(620)

        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QLabel("Webcam preview will appear here")
        self.camera_label.setObjectName("cameraPlaceholder")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setScaledContents(False)
        self.camera_label.setMinimumSize(520, 760)
        self.camera_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        camera_layout.addWidget(self.camera_label)
        camera_panel.setLayout(camera_layout)

        return camera_panel

    def create_side_panel(self, title_text, starting_status, labels):
        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")

        side_layout = QVBoxLayout()
        side_layout.setSpacing(16)

        title = QLabel(title_text)
        title.setObjectName("panelTitle")

        self.status_label = QLabel(starting_status)
        self.status_label.setObjectName("statusText")
        self.status_label.setWordWrap(True)

        label_text = QLabel("Training label")
        label_text.setObjectName("statusText")

        self.label_selector = QComboBox()
        self.label_selector.addItems(labels)
        self.label_selector.setMinimumHeight(40)

        self.clip_count_label = QLabel("Clips recorded this session: 0")
        self.clip_count_label.setObjectName("statusText")

        self.record_button = QPushButton("Record Training Clip")
        self.record_button.setMinimumHeight(46)
        self.record_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_button.clicked.connect(self.start_countdown)

        self.train_button = QPushButton("Train Model")
        self.train_button.setMinimumHeight(46)
        self.train_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.train_button.clicked.connect(self.simulate_training)

        self.evaluate_button = QPushButton("View Evaluation")
        self.evaluate_button.setMinimumHeight(42)
        self.evaluate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.evaluate_button.clicked.connect(self.show_evaluation)

        back_button = QPushButton("Back")
        back_button.setMinimumHeight(42)
        back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        back_button.clicked.connect(self.back_callback)

        side_layout.addWidget(title)
        side_layout.addWidget(self.status_label)
        side_layout.addSpacing(12)
        side_layout.addWidget(label_text)
        side_layout.addWidget(self.label_selector)
        side_layout.addWidget(self.clip_count_label)
        side_layout.addSpacing(12)
        side_layout.addWidget(self.record_button)
        side_layout.addWidget(self.train_button)
        side_layout.addWidget(self.evaluate_button)
        side_layout.addStretch()
        side_layout.addWidget(back_button)

        side_panel.setLayout(side_layout)
        return side_panel

    def start_countdown(self):
        if self.camera_worker is None:
            self.status_label.setText("Camera is not running.")
            return

        self.keep_status_message = False

        self.record_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self.evaluate_button.setEnabled(False)

        self.current_label = self.label_selector.currentText()
        self.current_label_folder = label_to_folder(self.current_label)

        self.countdown_number = COUNTDOWN_SECONDS
        self.status_label.setText(f"Recording starts in {self.countdown_number}...")
        self.play_beep()

        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

    def update_countdown(self):
        self.countdown_number -= 1

        if self.countdown_number > 0:
            self.status_label.setText(f"Recording starts in {self.countdown_number}...")
            self.play_beep()
            return

        self.countdown_timer.stop()
        self.start_recording()

    def start_recording(self):
        self.play_beep()

        filename = f"{self.current_label_folder}_{timestamp_slug()}.mp4"
        self.current_output_path = (
            DATA_DIR / "raw" / self.mode / self.current_label_folder / filename
        )

        self.status_label.setText(f"Recording {self.current_label} clip...")
        self.camera_worker.start_recording(
            str(self.current_output_path),
            duration_seconds=RECORD_SECONDS,
        )

    def handle_recording_finished(self, output_path):
        self.play_finished_beep()
        self.clip_count += 1

        try:
            recording_id = save_recording(
                mode=self.mode,
                purpose="training",
                clip_path=output_path,
                label=self.current_label_folder,
                accepted_for_training=True,
                trained=False,
            )

            self.status_label.setText("Extracting landmarks...")
            landmarks_path = extract_landmarks_from_video(
                output_path,
                mode=self.mode,
                label=self.current_label_folder,
            )

            features_path = extract_and_save_features(landmarks_path)

            update_recording_paths(
                recording_id,
                landmarks_path=landmarks_path,
                features_path=features_path,
            )

            database_text = f"Database row: {recording_id}"

        except Exception as error:
            self.status_label.setText(f"Save/extraction failed:\n{error}")
            self.record_button.setEnabled(True)
            self.train_button.setEnabled(True)
            self.evaluate_button.setEnabled(True)
            return

        self.status_label.setText(
            f"Saved {self.current_label} clip:\n"
            f"{display_path(output_path)}\n\n"
            f"Saved landmarks:\n"
            f"{display_path(landmarks_path)}\n\n"
            f"Saved features:\n"
            f"{display_path(features_path)}\n\n"
            f"{database_text}"
        )
        self.clip_count_label.setText(f"Clips recorded this session: {self.clip_count}")

        self.record_button.setEnabled(True)
        self.train_button.setEnabled(True)
        self.evaluate_button.setEnabled(True)



    def simulate_training(self):
        self.keep_status_message = True
        self.train_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.evaluate_button.setEnabled(False)

        self.status_label.setText("Training model...")
        QApplication.processEvents()

        try:
            result = train_model(self.mode)

        except Exception as error:
            self.status_label.setText(f"Training failed:\n{error}")
            self.train_button.setEnabled(True)
            self.record_button.setEnabled(True)
            self.evaluate_button.setEnabled(True)
            return

        if not result["success"]:
            self.status_label.setText(result["message"])
            self.train_button.setEnabled(True)
            self.record_button.setEnabled(True)
            self.evaluate_button.setEnabled(True)
            return

        cv_accuracy = result.get("cv_accuracy_mean")
        holdout_accuracy = result.get("holdout_accuracy")

        if cv_accuracy is None:
            cv_text = "CV accuracy: not available yet"
        else:
            cv_text = f"CV accuracy: {cv_accuracy:.2%}"

        if holdout_accuracy is None:
            holdout_text = "Holdout accuracy: not available yet"
        else:
            holdout_text = f"Holdout accuracy: {holdout_accuracy:.2%}"

        label_counts = result.get("label_counts", {})
        new_clip_count = result.get("new_clip_count", 0)

        label_summary = ", ".join(
            f"{label}: {count}"
            for label, count in label_counts.items()
        )

        confusion_matrix_path = result.get("confusion_matrix_path")

        if confusion_matrix_path is None:
            confusion_matrix_text = "not available yet"
        else:
            confusion_matrix_text = display_path(confusion_matrix_path)

        self.status_label.setText(
            f"{result['message']}\n\n"
            f"Rows: {result['row_count']}\n"
            f"New clips trained: {new_clip_count}\n"
            f"Classes: {label_summary}\n\n"
            f"{cv_text}\n"
            f"{holdout_text}\n\n"
            f"Model saved:\n"
            f"{display_path(result['model_path'])}\n\n"
            f"Metrics saved:\n"
            f"{display_path(result['metrics_path'])}\n\n"
            f"Confusion matrix:\n"
            f"{confusion_matrix_text}"
        )

        self.clip_count_label.setText(f"Training rows: {result['row_count']}")

        self.train_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.evaluate_button.setEnabled(True)


    def show_evaluation(self):
        self.keep_status_message = True

        try:
            summary = format_evaluation_summary(self.mode)
        except Exception as error:
            self.status_label.setText(f"Evaluation unavailable:\n{error}")
            return

        self.status_label.setText("Loaded latest evaluation.")

        dialog = EvaluationDialog(
            self.mode,
            summary,
            parent=self,
        )
        dialog.exec()

    def start_camera(self):
        if self.camera_worker is not None:
            return

        self.camera_worker = CameraWorker()
        self.camera_worker.frame_ready.connect(self.update_camera_frame)
        self.camera_worker.camera_error.connect(self.show_camera_error)
        self.camera_worker.pose_status.connect(self.update_pose_status)
        self.camera_worker.recording_finished.connect(self.handle_recording_finished)
        self.camera_worker.start()

    def stop_camera(self):
        self.stop_countdown()

        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker = None

    def stop_countdown(self):
        if self.countdown_timer is not None and self.countdown_timer.isActive():
            self.countdown_timer.stop()

    def update_camera_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )

        crop_x = max(0, (scaled_pixmap.width() - self.camera_label.width()) // 2)
        crop_y = max(0, (scaled_pixmap.height() - self.camera_label.height()) // 2)

        cropped_pixmap = scaled_pixmap.copy(
            crop_x,
            crop_y,
            self.camera_label.width(),
            self.camera_label.height(),
        )

        self.camera_label.setPixmap(cropped_pixmap)

    def show_camera_error(self, message):
        self.camera_label.setText(message)

    def update_pose_status(self, person_detected):
        if person_detected:
            self.missed_pose_frames = 0
            self.camera_label.setStyleSheet("")

            if self.status_label.text() == NO_PERSON_MESSAGE and not self.keep_status_message:
                self.status_label.setText(self.default_status)

            return

        if self.keep_status_message:
            return

        self.missed_pose_frames += 1

        if (
            self.missed_pose_frames >= POSE_MISSES_BEFORE_WARNING
            and self.record_button.isEnabled()
        ):
            self.status_label.setText(NO_PERSON_MESSAGE)

    def play_beep(self):
        QApplication.beep()

    def play_finished_beep(self):
        QApplication.beep()
        QTimer.singleShot(180, QApplication.beep)


def format_metric_value(value, suffix=""):
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.2f}{suffix}"

    return f"{value}{suffix}"


def build_key_metrics_text(mode, features):
    person_detection = features.get("person_detection") or {}
    detected_frames = person_detection.get("detected_frames")
    total_frames = person_detection.get("total_frames")
    detection_ratio = person_detection.get("detection_ratio")

    if detected_frames is not None and total_frames is not None:
        pose_frames_text = f"{detected_frames}/{total_frames}"
    else:
        pose_frames_text = None

    if detection_ratio is not None:
        pose_ratio_text = f"{detection_ratio * 100:.1f}%"
    else:
        pose_ratio_text = None

    if mode == "bowling":
        metrics = [
            ("Action status", features.get("action_status")),
            ("Pose detected frames", pose_frames_text),
            ("Pose detection ratio", pose_ratio_text),
            ("Detected arm", features.get("detected_bowling_arm")),
            ("Peak wrist speed", features.get("peak_wrist_speed")),
            ("Mean wrist speed near release", features.get("mean_wrist_speed_near_release")),
            ("Release height", features.get("release_height")),
            ("Release reach", features.get("release_forward_reach")),
            ("Elbow angle at release", features.get("elbow_angle_at_release"), " deg"),
            ("Shoulder rotation", features.get("shoulder_rotation_range"), " deg"),
            ("Torso lean at release", features.get("torso_lean_at_release"), " deg"),
            ("Action duration", features.get("action_duration"), " frames"),
            ("Critical phase start", features.get("critical_phase_start_frame")),
            ("Release frame", features.get("release_frame")),
            ("Legality", features.get("legality")),
            ("Legality confidence", features.get("legality_confidence")),
            ("Start elbow angle", features.get("start_elbow_angle"), " deg"),
            ("Release elbow angle", features.get("release_elbow_angle"), " deg"),
            ("Elbow extension", features.get("elbow_extension"), " deg"),
            ("Max elbow extension", features.get("max_elbow_extension"), " deg"),
            ("Rapid elbow extension", features.get("rapid_elbow_extension"), " deg"),
            ("Frames over 15 deg", features.get("frames_over_15_degrees")),
        ]

    elif mode == "batting":
        metrics = [
            ("Action status", features.get("action_status")),
            ("Pose detected frames", pose_frames_text),
            ("Pose detection ratio", pose_ratio_text),
            ("Peak hand speed", features.get("peak_hand_speed")),
            ("Mean hand speed", features.get("mean_hand_speed")),
            ("Peak speed timing", features.get("peak_speed_timing")),
            ("Hand horizontal range", features.get("hand_horizontal_range")),
            ("Hand vertical range", features.get("hand_vertical_range")),
            ("Hand path angle", features.get("hand_path_angle"), " deg"),
            ("Final hand height", features.get("final_hand_height")),
            ("Lowest hand position", features.get("lowest_hand_position")),
            ("Shoulder rotation", features.get("shoulder_rotation_range"), " deg"),
            ("Max knee bend", features.get("max_knee_bend"), " deg"),
            ("Head drop", features.get("head_drop")),
            ("Torso lean range", features.get("torso_lean_range"), " deg"),
            ("Action duration", features.get("action_duration"), " frames"),
        ]

    else:
        return ""

    lines = []

    for metric in metrics:
        if len(metric) == 2:
            label, value = metric
            suffix = ""
        else:
            label, value, suffix = metric

        lines.append(f"{label}: {format_metric_value(value, suffix)}")

    return "\n".join(lines)
