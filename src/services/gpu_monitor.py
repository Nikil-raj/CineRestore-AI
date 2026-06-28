import subprocess


class GPUMonitor:
    """
    Reads NVIDIA GPU statistics using nvidia-smi.

    Returned values:
        - temperature (°C)
        - memory used (MB)
        - memory total (MB)
        - gpu utilization (%)
        - power draw (W)
    """

    def __init__(self):

        self.command = [
            "nvidia-smi",
            "--query-gpu="
            "temperature.gpu,"
            "memory.used,"
            "memory.total,"
            "utilization.gpu,"
            "power.draw",
            "--format=csv,noheader,nounits"
        ]

    # --------------------------------------------------

    def read(self):

        try:

            result = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                check=True
            )

            values = [
                value.strip()
                for value in result.stdout.strip().split(",")
            ]

            return {

                "temperature": int(values[0]),

                "memory_used": int(values[1]),

                "memory_total": int(values[2]),

                "gpu_util": int(values[3]),

                "power": float(values[4])

            }

        except Exception:

            return {

                "temperature": -1,

                "memory_used": -1,

                "memory_total": -1,

                "gpu_util": -1,

                "power": -1

            }

    # --------------------------------------------------

    def pretty(self):

        s = self.read()

        return (
            f"GPU "
            f"{s['temperature']}°C | "
            f"VRAM {s['memory_used']}/{s['memory_total']} MB | "
            f"GPU {s['gpu_util']}% | "
            f"{s['power']:.1f}W"
        )