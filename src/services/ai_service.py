"""
CineRestore AI — AI Service
===========================
Production-grade AI processing engine.

Sprint 1.3 additions
--------------------
- Adaptive tile recovery: tile decreases across retry attempts.
  Attempt 1 → tile 128
  Attempt 2 → tile  96
  Attempt 3 → tile  64
- registry.set_retry_tile(attempt) is called before every GPU recovery
  so ModelLoader picks up the new tile on the next reload.
- registry.reset_tile(model_key) is called after every successful
  frame and after every batch reload.
- ProcessingStats gains average_frame_time and frames_per_second.
- print_summary() emits all nine tracked metrics.

Responsibilities
----------------
- Orchestrate batch-level and frame-level enhancement.
- Enforce a per-frame retry policy (up to MAX_RETRIES attempts).
- Perform adaptive-tile GPU recovery between failed attempts.
- Reload GPU and model after every completed batch.
- Emit structured, professional console logs.
- Track and report processing statistics.

Design Notes
------------
- All tuneable numbers are module-level constants.
- Each method has exactly one clear responsibility.
- The architecture is intentionally open for later addition of:
    - Dynamic reload intervals
    - Pause / resume (inject threading.Event into process())
    - Multi-GPU routing (inject gpu_id into __init__)
    - GFPGAN / CodeFormer / RIFE (registered in ModelRegistry)
  without changing the core processing contract.
"""

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch

from src.ai.engine import AIEngine
from src.ai.gpu_manager import GPUManager
from src.ai.registry import ModelRegistry
from src.services.batch_manager import BatchManager
from src.services.checkpoint_manager import CheckpointManager
from src.services.gpu_monitor import GPUMonitor


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Maximum enhancement attempts per frame before skipping it.
MAX_RETRIES: int = 3

#: Seconds to pause after GPU recovery before retrying.
RECOVERY_WAIT_SECONDS: float = 2.0

#: Tile sizes used on attempt 1, 2, 3 respectively.
#: Must stay in sync with ModelRegistry.DEFAULT_TILE_SEQUENCE.
TILE_SEQUENCE: List[int] = [128, 96, 64]

#: Visual separator used in all log blocks.
SEPARATOR: str = "=" * 60


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------

@dataclass
class ProcessingStats:
    """
    Accumulator for a single enhancement run.

    All metrics are plain integers / floats so they can be serialised
    to JSON or forwarded to a GUI without any conversion.

    Extension: add new fields here (e.g. tile_adaptations,
    gpu_switch_count) without touching the processing loop.
    """

    total_frames: int = 0
    completed_frames: int = 0
    recovered_frames: int = 0
    skipped_frames: int = 0
    retry_attempts: int = 0
    gpu_reloads: int = 0
    start_time: float = field(default_factory=time.time)

    # Timing accumulators for per-frame averages
    _frame_time_total: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------

    def record_frame_time(self, elapsed: float) -> None:
        """Accumulate the wall-clock time spent on one successful frame."""
        self._frame_time_total += elapsed

    # ------------------------------------------------------------------

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    # ------------------------------------------------------------------

    def elapsed_formatted(self) -> str:
        total = int(self.elapsed_seconds())
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    # ------------------------------------------------------------------

    def average_frame_time(self) -> float:
        """Mean wall-clock seconds per successfully enhanced frame."""
        if self.completed_frames == 0:
            return 0.0
        return self._frame_time_total / self.completed_frames

    # ------------------------------------------------------------------

    def frames_per_second(self) -> float:
        """Overall throughput across the entire job."""
        elapsed = self.elapsed_seconds()
        if elapsed == 0.0:
            return 0.0
        return self.completed_frames / elapsed


# ---------------------------------------------------------------------------
# AIService
# ---------------------------------------------------------------------------

class AIService:
    """
    Orchestrates AI-based frame enhancement for a complete video job.

    Public interface (unchanged from Sprint 1.2)
    --------------------------------------------
    process(on_progress, on_error) -> bool

    Sprint 1.3 additions (internal only)
    -------------------------------------
    - Adaptive tile via ModelRegistry.set_retry_tile() /
      reset_tile() before/after every GPU recovery.
    - Average frame time and FPS in the summary.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        frames_dir: str | Path,
        output_dir: str | Path,
        checkpoint_file: str | Path,
        batch_size: int = 50,
    ) -> None:

        self.frames_dir: Path = Path(frames_dir)
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # GPU stack
        self.gpu: GPUManager = GPUManager()
        self.engine: AIEngine = AIEngine(self.gpu)

        # Tile-adaptive registry reference.
        # AIService reaches into the registry to drive tile changes;
        # ModelLoader reads from it transparently.
        self.registry: ModelRegistry = self.gpu.loader.registry

        # Supporting services
        self.batch_manager: BatchManager = BatchManager(
            self.frames_dir,
            batch_size=batch_size,
        )
        self.checkpoint: CheckpointManager = CheckpointManager(
            checkpoint_file
        )
        self.gpu_monitor: GPUMonitor = GPUMonitor()
        self.stats: ProcessingStats = ProcessingStats()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Enhance all remaining frames, batch by batch.

        Returns
        -------
        True  — job completed (all frames attempted).
        False — reserved for future pause/resume signalling.

        Extension points
        ----------------
        - Inject pause_event: threading.Event to add pause/resume.
        - Inject gpu_id: int to route work to a specific CUDA device.
        - Replace batch_manager.batches() with a priority queue for
          adaptive scheduling.
        """

        last_frame: int = self.checkpoint.get_last_frame()
        self.stats.total_frames = self.batch_manager.total_frames()
        self.stats.start_time = time.time()

        for batch in self.batch_manager.batches(
            self.output_dir,
            last_frame,
        ):
            self._process_batch(batch, on_progress, on_error)
            self.reload_batch()

        self.checkpoint.finish()
        self.print_summary()

        return True

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        batch: list,
        on_progress: Optional[Callable[[int, int], None]],
        on_error: Optional[Callable[[str], None]],
    ) -> None:
        """
        Iterate every frame in a batch, delegating per-frame work.

        This method owns only the iteration; all enhancement logic,
        retry logic, and GPU recovery live in process_frame().
        """

        for frame_path in batch:

            frame_number: int = int(
                frame_path.stem.split("_")[1]
            )

            output_path: Path = (
                self.output_dir / f"{frame_path.stem}_out.png"
            )

            self.process_frame(
                frame_path,
                output_path,
                frame_number,
                on_progress,
                on_error,
            )

    # ------------------------------------------------------------------
    # Frame processing — retry loop
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_path: Path,
        output_path: Path,
        frame_number: int,
        on_progress: Optional[Callable[[int, int], None]],
        on_error: Optional[Callable[[str], None]],
    ) -> None:
        """
        Attempt to enhance a single frame with up to MAX_RETRIES
        attempts, each at a progressively smaller tile size.

        Attempt 1 → tile 128
        Attempt 2 → tile  96  (after GPU recovery)
        Attempt 3 → tile  64  (after GPU recovery)

        A failed frame is recorded in the checkpoint and skipped —
        the enhancement job continues with the remaining frames.
        """

        frame_start: float = time.time()

        for attempt in range(1, MAX_RETRIES + 1):

            # Set tile for this attempt before printing status
            self.registry.set_retry_tile(attempt)
            current_tile: int = self.registry.get_tile()

            self.print_gpu_status(frame_number, attempt, current_tile)

            try:

                self.enhance_with_retry(frame_path, output_path)

                # ── Success ─────────────────────────────────────────
                self.gpu.cleanup_frame()

                self.checkpoint.update_progress(
                    frame_number,
                    self.gpu.reload_count,
                )

                self.stats.completed_frames += 1
                self.stats.record_frame_time(time.time() - frame_start)

                if attempt > 1:
                    self.stats.recovered_frames += 1

                # Restore default tile after a successful frame.
                self.registry.reset_tile(self.gpu.model_key)

                if on_progress:
                    on_progress(frame_number, self.stats.total_frames)

                return  # ← done with this frame

            except Exception as error:

                self.stats.retry_attempts += 1
                self.print_error(frame_number, attempt, error, current_tile)

                if on_error:
                    on_error(str(error))

                if attempt < MAX_RETRIES:
                    # Recover and try again with a smaller tile.
                    # set_retry_tile(attempt + 1) will be called at the
                    # top of the next loop iteration, so gpu.reload()
                    # here uses the CURRENT tile; the registry update
                    # happens before the next load() call inside reload.
                    self.recover_gpu(attempt)
                else:
                    # All retries exhausted — record, skip, continue.
                    self.checkpoint.add_error(frame_number, str(error))
                    self.stats.skipped_frames += 1
                    self.registry.reset_tile(self.gpu.model_key)

                    print()
                    print(SEPARATOR)
                    print(
                        f"  Frame {frame_number} skipped after "
                        f"{MAX_RETRIES} failed attempts."
                    )
                    print(SEPARATOR)

    # ------------------------------------------------------------------
    # Enhancement call
    # ------------------------------------------------------------------

    def enhance_with_retry(
        self,
        frame_path: Path,
        output_path: Path,
    ) -> None:
        """
        Thin wrapper around AIEngine.enhance_frame().

        Isolation rationale
        -------------------
        Keeping enhancement as its own method lets future sprints inject:
        - A different outscale per frame (e.g. content-aware scaling).
        - Face restoration as a post-pass (GFPGAN / CodeFormer).
        - Frame-interpolation pre-pass (RIFE).
        without modifying the retry loop in process_frame().
        """

        self.engine.enhance_frame(frame_path, output_path)

    # ------------------------------------------------------------------
    # GPU recovery
    # ------------------------------------------------------------------

    def recover_gpu(self, current_attempt: int) -> None:
        """
        Full GPU teardown → CUDA flush → rebuild sequence.

        The tile for the *next* attempt is set at the top of the retry
        loop in process_frame(), after this method returns, so
        ModelLoader will pick it up automatically on the next reload.

        Recovery order
        --------------
        1. Release model references in the engine layer.
        2. torch.cuda.empty_cache() — free fragmented CUDA allocations.
        3. torch.cuda.synchronize() — wait for pending CUDA ops.
        4. gc.collect()             — release Python-side references.
        5. gpu.reload()             — rebuild GPUManager + reload model.
        6. engine.update_model()    — synchronise engine's model ref.
        7. Sleep RECOVERY_WAIT_SECONDS.

        Parameters
        ----------
        current_attempt : int
            The attempt that just failed (1-indexed).  Logged for
            diagnostics; the next attempt's tile is set externally.
        """

        next_attempt: int = current_attempt + 1
        next_tile: int = TILE_SEQUENCE[
            min(next_attempt - 1, len(TILE_SEQUENCE) - 1)
        ]

        print()
        print(SEPARATOR)
        print("  CUDA ERROR DETECTED")
        print(f"  Recovering GPU...  (next tile → {next_tile} px)")
        print(SEPARATOR)

        self.engine.release_model()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()

        # Advance tile state *before* reload so ModelLoader sees it.
        self.registry.set_retry_tile(next_attempt)
        self.gpu.reload()
        self.stats.gpu_reloads += 1

        self.engine.update_model()

        time.sleep(RECOVERY_WAIT_SECONDS)

        print()
        print(SEPARATOR)
        print(f"  Retrying...  (tile = {next_tile} px)")
        print(SEPARATOR)

    # ------------------------------------------------------------------
    # Batch reload
    # ------------------------------------------------------------------

    def reload_batch(self) -> None:
        """
        Release and reload the GPU at the end of every completed batch.

        Tile is reset to the model default before the reload so the
        first frame of the next batch always starts at the largest tile.

        Extension point
        ---------------
        Replace this method's body with a thermal/VRAM-driven reload
        decision (adaptive reload interval) without touching
        _process_batch().
        """

        print()
        print(SEPARATOR)
        print("  Reloading GPU after batch...")
        print(SEPARATOR)

        self.engine.release_model()

        # Reset tile to default before reloading for the next batch.
        self.registry.reset_tile(self.gpu.model_key)

        self.gpu.reload()
        self.stats.gpu_reloads += 1

        self.engine.update_model()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def print_gpu_status(
        self,
        frame_number: int,
        attempt: int,
        tile_size: int,
    ) -> None:
        """
        Emit a structured per-frame status block to stdout.

        Shows attempt number and tile size only when attempt > 1 to
        keep the happy-path output clean.
        """

        gpu_stats: dict = self.gpu_monitor.read()

        temperature: int  = gpu_stats["temperature"]
        memory_used: int  = gpu_stats["memory_used"]
        memory_total: int = gpu_stats["memory_total"]
        gpu_util: int     = gpu_stats["gpu_util"]
        power: float      = gpu_stats["power"]

        print()
        print(SEPARATOR)
        print(f"  Frame {frame_number} / {self.stats.total_frames}")

        if attempt > 1:
            print(f"  Attempt   : {attempt} / {MAX_RETRIES}")
            print(f"  Tile      : {tile_size} px")

        print(f"  GPU Temp  : {temperature}°C")
        print(f"  VRAM      : {memory_used} / {memory_total} MB")
        print(f"  GPU Load  : {gpu_util}%")
        print(f"  Power     : {power:.1f} W")
        print(SEPARATOR)

    # ------------------------------------------------------------------

    def print_error(
        self,
        frame_number: int,
        attempt: int,
        error: Exception,
        tile_size: int,
    ) -> None:
        """
        Emit a structured error block to stdout.
        """

        print()
        print(SEPARATOR)
        print(
            f"  ERROR — Frame {frame_number}  |  "
            f"Attempt {attempt} / {MAX_RETRIES}  |  "
            f"Tile {tile_size} px"
        )
        print(f"  {type(error).__name__}: {error}")
        print(SEPARATOR)

    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """
        Emit a structured job summary to stdout once all frames have
        been attempted.
        """

        avg_time: float = self.stats.average_frame_time()
        fps: float = self.stats.frames_per_second()

        print()
        print(SEPARATOR)
        print("  PROCESSING SUMMARY")
        print(SEPARATOR)
        print(f"  Frames Total       : {self.stats.total_frames}")
        print(f"  Frames Completed   : {self.stats.completed_frames}")
        print(f"  Frames Recovered   : {self.stats.recovered_frames}")
        print(f"  Frames Skipped     : {self.stats.skipped_frames}")
        print(f"  Retry Attempts     : {self.stats.retry_attempts}")
        print(f"  GPU Reloads        : {self.stats.gpu_reloads}")
        print(f"  Elapsed Time       : {self.stats.elapsed_formatted()}")
        print(f"  Avg Frame Time     : {avg_time:.2f} s/frame")
        print(f"  Frames Per Second  : {fps:.3f} fps")
        print(SEPARATOR)