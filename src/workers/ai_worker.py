from pathlib import Path
import subprocess

from PySide6.QtCore import QObject, Signal


class AIWorker(QObject):

    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()

    def run(self):

        try:

            project_root = Path(__file__).resolve().parents[2]

            python_exe = (
                project_root
                / ".venv_ai"
                / "Scripts"
                / "python.exe"
            )

            realesrgan = (
                project_root
                / "models"
                / "RealESRGAN"
            )

            frames_dir = (
                project_root
                / "temp"
                / "frames"
            )

            output_dir = (
                project_root
                / "temp"
                / "enhanced_frames"
            )

            output_dir.mkdir(
                exist_ok=True,
                parents=True
            )

            frames = sorted(frames_dir.glob("*.png"))

            total = len(frames)

            for current, frame in enumerate(frames, start=1):

                output_file = output_dir / f"{frame.stem}_out.png"

                # Skip frames already processed
                if output_file.exists():
                    self.progress.emit(current, total)
                    continue

                command = [
                    str(python_exe),
                    "inference_realesrgan.py",
                    "-n",
                    "realesr-general-x4v3",
                    "-i",
                    str(frame),
                    "-o",
                    str(output_dir),
                    "--tile",
                    "128",
                    "--tile_pad",
                    "20"
                ]

                result = subprocess.run(
                    command,
                    cwd=realesrgan,
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:

                    print("=" * 60)
                    print(f"FAILED : {frame.name}")
                    print(result.stderr)
                    print("=" * 60)

                self.progress.emit(
                    current,
                    total
                )

            self.finished.emit()

        except Exception as e:

            self.error.emit(str(e))