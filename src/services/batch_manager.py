from pathlib import Path


class BatchManager:
    """
    Generates batches of frames for processing.
    Resume automatically skips frames that already
    have enhanced output.
    """

    def __init__(self, frames_dir, batch_size=50):

        self.frames_dir = Path(frames_dir)
        self.batch_size = batch_size

    # ----------------------------------------------------

    def all_frames(self):

        return sorted(
            self.frames_dir.glob("frame_*.png")
        )

    # ----------------------------------------------------

    def remaining_frames(
        self,
        output_dir,
        last_completed_frame=0
    ):

        output_dir = Path(output_dir)

        remaining = []

        for frame in self.all_frames():

            output = (
                output_dir /
                f"{frame.stem}_out.png"
            )

            # Source of truth:
            # already enhanced -> skip
            if output.exists():

                print(
                    f"✓ Skipping {frame.name}"
                )

                continue

            frame_number = int(
                frame.stem.split("_")[1]
            )

            # Fallback to checkpoint
            if frame_number <= last_completed_frame:

                continue

            remaining.append(frame)

        return remaining

    # ----------------------------------------------------

    def batches(
        self,
        output_dir,
        last_completed_frame=0
    ):

        frames = self.remaining_frames(
            output_dir,
            last_completed_frame
        )

        for i in range(
            0,
            len(frames),
            self.batch_size
        ):

            yield frames[
                i:i + self.batch_size
            ]

    # ----------------------------------------------------

    def total_frames(self):

        return len(
            self.all_frames()
        )