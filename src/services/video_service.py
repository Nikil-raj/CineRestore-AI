from src.video.video_info import VideoInfo
class VideoService:
    @staticmethod
    def load_video(video_path: str):
        return VideoInfo.get_metadata(video_path)
