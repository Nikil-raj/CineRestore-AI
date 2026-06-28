from pathlib import Path

from src.ai.engine import AIEngine
from src.ai.gpu_manager import GPUManager
from src.services.batch_manager import BatchManager
from src.services.checkpoint_manager import CheckpointManager
from src.services.gpu_monitor import GPUMonitor

class AIService:

    def __init__(
        self,
        frames_dir,
        output_dir,
        checkpoint_file,
        batch_size=50,
    ):

        self.frames_dir = Path(frames_dir)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.gpu = GPUManager()

        self.engine = AIEngine(
            self.gpu
        )

        self.batch_manager = BatchManager(
            self.frames_dir,
            batch_size=batch_size
        )

        self.checkpoint = CheckpointManager(
            checkpoint_file
        )

        self.gpu_monitor = GPUMonitor()

    # --------------------------------------------------

       # --------------------------------------------------

    def process(
        self,
        on_progress=None,
        on_error=None,
    ):

        last_frame = (
            self.checkpoint.get_last_frame()
        )

        total = self.batch_manager.total_frames()

        for batch in self.batch_manager.batches(
            self.output_dir,
            last_frame
        ):

            for frame in batch:

                frame_number = int(
                    frame.stem.split("_")[1]
                )

                output = (
                    self.output_dir
                    / f"{frame.stem}_out.png"
                )

                try:

                    stats = self.gpu_monitor.read()

                    print()
                    print("=" * 70)
                    print(
                        f"[{frame_number}/{total}] "
                        f"Processing {frame.name}"
                    )
                    print(
                        f"GPU Temp : {stats['temperature']}°C"
                    )
                    print(
                        f"VRAM     : "
                        f"{stats['memory_used']} / "
                        f"{stats['memory_total']} MB"
                    )
                    print(
                        f"GPU Load : "
                        f"{stats['gpu_util']}%"
                    )
                    print(
                        f"Power    : "
                        f"{stats['power']} W"
                    )
                    print("=" * 70)

                    self.engine.enhance_frame(
                        frame,
                        output
                    )

                    self.gpu.cleanup_frame()

                    self.checkpoint.update_progress(
                        frame_number,
                        self.gpu.reload_count
                    )

                    if on_progress:

                        on_progress(
                            frame_number,
                            total
                        )

                except Exception as e:

                    self.checkpoint.add_error(
                        frame_number,
                        str(e)
                    )

                    print()
                    print("=" * 70)
                    print("ERROR")
                    print("=" * 70)
                    print(f"Frame : {frame.name}")
                    print(f"Error : {e}")
                    print("=" * 70)

                    if on_error:

                        on_error(str(e))

                    return False

            print()
            print("=" * 60)
            print("Reloading GPU after batch...")
            print("=" * 60)

            self.engine.release_model()

            self.gpu.reload()

            self.engine.update_model()

        self.checkpoint.finish()

        return True