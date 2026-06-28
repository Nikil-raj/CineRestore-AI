import json
from datetime import datetime
from pathlib import Path


class CheckpointManager:
    """
    Manages enhancement progress and allows jobs to resume.
    """

    AUTO_SAVE_INTERVAL = 10

    def __init__(self, checkpoint_file):

        self.file = Path(checkpoint_file)

        self.unsaved_changes = 0

        self.data = {
            "version": 1,

            "project": {
                "created": datetime.now().isoformat()
            },

            "movie": {
                "input_video": "",
                "total_frames": 0,
                "fps": 0.0,
                "width": 0,
                "height": 0
            },

            "ai": {
                "model": "general",
                "scale": 4,
                "tile": 200,
                "tile_pad": 20,
                "reload_interval": 50
            },

            "progress": {
                "last_frame": 0,
                "completed_frames": 0,
                "reloads": 0,
                "elapsed_seconds": 0
            },

            "stats": {
                "average_frame_time": 0.0,
                "frames_per_second": 0.0
            },

            "errors": [],

            "output": {
                "enhanced_folder": "",
                "output_video": ""
            }
        }

        self.load()

    # -----------------------------------------------------

    def load(self):

        if not self.file.exists():
            return

        with open(
            self.file,
            "r",
            encoding="utf-8"
        ) as f:

            loaded = json.load(f)

        # Old checkpoint format
        if "movie" not in loaded:

            print("Old checkpoint detected.")
            print("Creating a new checkpoint structure.")

            return

        self.data = loaded

    # -----------------------------------------------------

    def save(self):

        self.file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.data,
                f,
                indent=4
            )

        self.unsaved_changes = 0

    # -----------------------------------------------------

    def configure_movie(
        self,
        input_video,
        total_frames,
        fps,
        width,
        height
    ):

        movie = self.data["movie"]

        movie["input_video"] = str(input_video)
        movie["total_frames"] = total_frames
        movie["fps"] = fps
        movie["width"] = width
        movie["height"] = height

        self.save()

    # -----------------------------------------------------

    def configure_ai(
        self,
        model,
        scale,
        tile,
        tile_pad,
        reload_interval
    ):

        ai = self.data["ai"]

        ai["model"] = model
        ai["scale"] = scale
        ai["tile"] = tile
        ai["tile_pad"] = tile_pad
        ai["reload_interval"] = reload_interval

        self.save()

    # -----------------------------------------------------

    def update_progress(
        self,
        frame_number,
        reload_count
    ):

        progress = self.data["progress"]

        progress["last_frame"] = frame_number
        progress["completed_frames"] = frame_number
        progress["reloads"] = reload_count

        self.unsaved_changes += 1

        if self.unsaved_changes >= self.AUTO_SAVE_INTERVAL:
            self.save()

    # -----------------------------------------------------

    def add_error(
        self,
        frame_number,
        message
    ):

        self.data["errors"].append(
            {
                "frame": frame_number,
                "message": message,
                "time": datetime.now().isoformat()
            }
        )

        self.save()

    # -----------------------------------------------------

    def finish(self):

        self.save()

    # -----------------------------------------------------

    def get_last_frame(self):

        return self.data["progress"]["last_frame"]

    # -----------------------------------------------------

    def stats(self):

        progress = self.data["progress"]

        return {
            "last_frame": progress["last_frame"],
            "completed": progress["completed_frames"],
            "reloads": progress["reloads"],
            "errors": len(self.data["errors"])
        }