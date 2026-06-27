import sys
from PySide6.QtWidgets import QApplication
from src.core.app import CineRestoreApp
from src.core.logger import logger
from src.gui.theme import apply_theme
def main() -> None:
    """
    Entry point for CineRestore AI.
    """
    logger.info("Starting CineRestore AI...")
    app = QApplication(sys.argv)
    apply_theme(app)
    window = CineRestoreApp()
    window.show()
    logger.info("Application started successfully.")
    sys.exit(app.exec())
if __name__ == "__main__":
    main()
