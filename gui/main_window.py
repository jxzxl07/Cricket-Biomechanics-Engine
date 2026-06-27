from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from config import BATTING_LABELS, BOWLING_LABELS
from database.database import init_database
from gui.pages import AnalysisPage, TrainingPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        init_database()

        self.setWindowTitle("Cricket Biomechanics Engine")
        self.resize(1200, 850)

        self.stack = QStackedWidget()

        self.menu_page = self.create_menu_page()

        self.bowling_page = AnalysisPage(
            "bowling",
            "Bowling Mode",
            "Ready to analyse a bowling action.",
            self.show_menu_page,
        )

        self.train_bowling_page = TrainingPage(
            "bowling",
            "Train Bowling Model",
            "Choose the correct label, then record training clips.",
            BOWLING_LABELS,
            self.show_menu_page,
        )

        self.batting_page = AnalysisPage(
            "batting",
            "Batting Mode",
            "Ready to analyse a batting shot.",
            self.show_menu_page,
        )

        self.train_batting_page = TrainingPage(
            "batting",
            "Train Batting Model",
            "Choose the correct label, then record training clips.",
            BATTING_LABELS,
            self.show_menu_page,
        )

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.bowling_page)
        self.stack.addWidget(self.train_bowling_page)
        self.stack.addWidget(self.batting_page)
        self.stack.addWidget(self.train_batting_page)

        self.setCentralWidget(self.stack)
        self.apply_styles()

    def create_menu_page(self):
        page = QWidget()
        page.setObjectName("page")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(18)

        title = QLabel("Cricket Biomechanics Engine")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Select a mode")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        bowling_button = QPushButton("Bowling")
        train_bowling_button = QPushButton("Train Bowling Model")
        batting_button = QPushButton("Batting")
        train_batting_button = QPushButton("Train Batting Model")

        buttons = [
            bowling_button,
            train_bowling_button,
            batting_button,
            train_batting_button,
        ]

        for button in buttons:
            button.setMinimumHeight(48)
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        bowling_button.clicked.connect(self.show_bowling_page)
        train_bowling_button.clicked.connect(self.show_train_bowling_page)
        batting_button.clicked.connect(self.show_batting_page)
        train_batting_button.clicked.connect(self.show_train_batting_page)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(18)

        for button in buttons:
            layout.addWidget(button)

        page.setLayout(layout)
        return page

    def create_simple_page(self, title_text):
        page = QWidget()
        page.setObjectName("page")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(18)

        title = QLabel(title_text)
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        back_button = QPushButton("Back")
        back_button.setMinimumHeight(44)
        back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        back_button.clicked.connect(self.show_menu_page)

        layout.addWidget(title)
        layout.addWidget(back_button)

        page.setLayout(layout)
        return page
    

    def show_menu_page(self):
        self.stop_all_cameras()
        self.stack.setCurrentWidget(self.menu_page)

    def show_bowling_page(self):
        self.stop_all_cameras()
        self.stack.setCurrentWidget(self.bowling_page)
        self.bowling_page.start_camera()

    def show_train_bowling_page(self):
        self.stop_all_cameras()
        self.stack.setCurrentWidget(self.train_bowling_page)
        self.train_bowling_page.start_camera()

    def show_batting_page(self):
        self.stop_all_cameras()
        self.stack.setCurrentWidget(self.batting_page)
        self.batting_page.start_camera()

    def show_train_batting_page(self):
        self.stop_all_cameras()
        self.stack.setCurrentWidget(self.train_batting_page)
        self.train_batting_page.start_camera()

    def stop_all_cameras(self):
        pages = [
            self.bowling_page,
            self.train_bowling_page,
            self.batting_page,
            self.train_batting_page,
        ]

        for page in pages:
            if hasattr(page, "stop_camera"):
                page.stop_camera()


    def closeEvent(self, event):
        self.stop_all_cameras()
        event.accept()

    def apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #eef2f3;
            }

            QWidget#page {
                background-color: #eef2f3;
            }

            QLabel#title {
                color: #17202a;
                font-size: 30px;
                font-weight: 700;
            }

            QLabel#subtitle {
                color: #5f6f7a;
                font-size: 15px;
            }

            QPushButton {
                background-color: #ffffff;
                color: #17202a;
                border: 1px solid #cfd8dc;
                border-radius: 8px;
                padding: 10px 22px;
                font-size: 15px;
                font-weight: 600;
                min-width: 280px;
            }

            QPushButton:hover {
                background-color: #f6fafb;
                border: 1px solid #2f9e8f;
            }

            QPushButton:pressed {
                background-color: #e3f3f1;
                border: 1px solid #22786d;
            }

            QFrame#cameraPanel {
            background-color: #111820;
            border-radius: 8px;
            border: 1px solid #26323d;
        }

            QLabel#cameraPlaceholder {
                color: #d5dde3;
                font-size: 16px;
            }

            QFrame#sidePanel {
                background-color: #ffffff;
                border: 1px solid #cfd8dc;
                border-radius: 8px;
                padding: 18px;
            }

            QLabel#panelTitle {
                color: #17202a;
                font-size: 24px;
                font-weight: 700;
            }

            QLabel#statusText {
                color: #5f6f7a;
                font-size: 14px;
            }

            QLabel#resultText {
            color: #17202a;
            font-size: 18px;
            font-weight: 700;
        }

        QComboBox {
            background-color: #ffffff;
            color: #17202a;
            border: 1px solid #cfd8dc;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 14px;
            font-weight: 600;
        }

        QComboBox:hover {
            border: 1px solid #2f9e8f;
        }

        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #17202a;
            selection-background-color: #e3f3f1;
            selection-color: #17202a;
            border: 1px solid #cfd8dc;
        }

        QPushButton:disabled {
            background-color: #edf1f2;
            color: #9aa7ad;
            border: 1px solid #d8e0e3;
        }
            """
        )
