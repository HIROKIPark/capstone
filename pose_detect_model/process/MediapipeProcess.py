from typing import Optional

import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

from Image import Image
from PipedProcess import PipedProcess


class MediapipeProcess(PipedProcess):
    def __init__(self, model_params: dict):
        super().__init__()
        self.model_params = model_params
        self.model: Optional[Pose] = None
        self.counter = 0  # 폴 감지를 위한 카운터

    def init(self):
        self.model = Pose(**self.model_params)

    def process(self, input_data: tuple[Image, Optional[tuple[int, ...]]]) -> tuple[Image, Optional[bool]]:
        img, coords = input_data
        fall_detected = False
        pose_landmarks = {}  # 관절 랜드마크 저장

        if coords is not None:
            x1, y1, x2, y2 = coords
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img.size > 0:
                rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                results = self.model.process(rgb_img)

                if results.pose_landmarks:
                    # 관절 랜드마크를 이미지에 그리기
                    draw_landmarks(
                        cropped_img,
                        results.pose_landmarks,
                        POSE_CONNECTIONS,
                        draw_landmarks.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        draw_landmarks.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                    )
                    img[y1:y2, x1:x2] = cropped_img

                    # 관절 랜드마크 저장
                    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                        pose_landmarks[landmark_id] = (landmark.x, landmark.y, landmark.z)

                    # 폴 감지 로직
                    landmarks = results.pose_landmarks.landmark
                    left_hip = landmarks[Pose.PoseLandmark.LEFT_HIP.value]
                    right_hip = landmarks[Pose.PoseLandmark.RIGHT_HIP.value]
                    left_foot = landmarks[Pose.PoseLandmark.LEFT_HEEL.value]
                    right_foot = landmarks[Pose.PoseLandmark.RIGHT_HEEL.value]

                    hip_avg_y = (left_hip.y + right_hip.y) / 2
                    foot_avg_y = (left_foot.y + right_foot.y) / 2

                    if hip_avg_y > (foot_avg_y - 0.2):
                        self.counter += 1
                        if self.counter >= 5:
                            fall_detected = True
                            self.counter = 0
                    else:
                        self.counter = 0

        # 결과를 표시 (옵션)
        cv2.imshow('Result', img)
        cv2.waitKey(1)

        return img, fall_detected
