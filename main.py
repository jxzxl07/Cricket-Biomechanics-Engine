# Starts the application and runs the main loop.

import sys
from PyQt6.QtWidgets import QApplication # type: ignore
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()