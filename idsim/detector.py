from typing import Any, Union, Tuple, Optional
import cv2
import numpy as np

# Link -> https://github.com/timesler/facenet-pytorch
# Examples https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch


class FacialAreaRegion:
    x: int
    y: int
    w: int
    h: int
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    confidence: float

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        left_eye: Optional[Tuple[int, int]] = None,
        right_eye: Optional[Tuple[int, int]] = None,
        keypoints=None,
        confidence: Optional[float] = None,
    ):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.keypoints = keypoints
        self.confidence = confidence


class FastMtCnnClient:
    def __init__(self):
        self.model = self.build_model()

    def detect_faces(self, img: np.ndarray):
        """
        Detect and align face with mtcnn

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        img_rgb = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # mtcnn expects RGB but OpenCV read BGR
        detections = self.model.detect(
            img_rgb, landmarks=True
        )  # returns boundingbox, prob, landmark
        if detections is not None and len(detections) > 0:
            for current_detection in zip(*detections):
                x, y, w, h = xyxy_to_xywh(current_detection[0])
                confidence = current_detection[1]
                left_eye = current_detection[2][0]
                right_eye = current_detection[2][1]

                facial_area = FacialAreaRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    keypoints=current_detection[2],
                    confidence=confidence,
                )
                resp.append(facial_area)

        return resp

    def build_model(self) -> Any:
        """
        Build a fast mtcnn face detector model
        Returns:
            model (Any)
        """
        # this is not a must dependency. do not import it in the global level.
        try:
            from facenet_pytorch import MTCNN as fast_mtcnn
        except ModuleNotFoundError as e:
            raise ImportError(
                "FastMtcnn is an optional detector, ensure the library is installed."
                "Please install using 'pip install facenet-pytorch' "
            ) from e

        face_detector = fast_mtcnn(
            image_size=160,
            thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
            post_process=True,
            device="cpu",
            select_largest=False,  # return result in descending order
        )
        return face_detector


def xyxy_to_xywh(xyxy: Union[list, tuple]) -> list:
    """
    Convert xyxy format to xywh format.
    """
    x, y = xyxy[0], xyxy[1]
    w = xyxy[2] - x + 1
    h = xyxy[3] - y + 1
    return [x, y, w, h]
