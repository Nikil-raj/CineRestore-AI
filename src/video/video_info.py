import cv2
class VideoInfo:
    @staticmethod
    def get_metadata(video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Unable to open video.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps else 0
        cap.release()
        return {
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "frames": frames,
            "duration": duration,
        }
