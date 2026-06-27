import sys
from PySide6.QtWidgets import QApplication
from src.core.app import CineRestoreApp
def main():
    app = QApplication(sys.argv)
    window = CineRestoreApp()
    window.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()
