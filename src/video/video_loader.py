import cv2
class VideoLoader:
    @staticmethod
    def load_preview_frame(video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Unable to open video.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Jump to 25% of the movie
        preview_frame = max(0, int(total_frames * 0.25))
        cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame)
        success, frame = cap.read()
        cap.release()
        if not success:
            raise Exception("Unable to read preview frame.")
        return frame
