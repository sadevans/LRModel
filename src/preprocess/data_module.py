import torch
import torchaudio
import torchvision


class AVSRDataLoader:
    def __init__(self, modality, detector="retinaface", convert_gray=True):
        self.modality = modality
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if modality == "video":
            if detector == "retinaface":
                from detectors.retinaface.detector import LandmarksDetector
                from detectors.retinaface.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector(device=self.device)
                self.video_process = VideoProcess(convert_gray=convert_gray)

            if detector == "mediapipe":
                from detectors.mediapipe.detector import LandmarksDetector
                from detectors.mediapipe.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=convert_gray)

    def load_data(self, data_filename, landmarks=None, transform=True):
        video = self.load_video(data_filename)
        if not landmarks:
            # print("=======PROCESSING VIDEO=======")
            landmarks = self.landmarks_detector(video)
        
        if landmarks is not None:
            video = self.video_process(video, landmarks)
            if video is None:
                raise TypeError("video cannot be None")
            video = torch.tensor(video)
            torch.cuda.empty_cache()
            return video
        else: return None


    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
