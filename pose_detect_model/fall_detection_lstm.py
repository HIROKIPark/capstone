import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from collections import deque
from data.feature_extraction import extract_keypoints

# LSTM 모델 로드
lstm_model = load_model("model/lstm_model.h5")

# 프레임 버퍼 초기화
frame_buffer = deque(maxlen=150)

# LSTM 검증 함수
def lstm_fall_verification(img, fall_detected, mediapipe_model):
    if img is not None:
        frame_buffer.append(img)

    if fall_detected and len(frame_buffer) == 150:
        keypoints_sequence = []

        for frame in frame_buffer:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mediapipe_model.process(rgb_frame)
            keypoints = extract_keypoints(results)
            keypoints_sequence.append(keypoints)

        # 시퀀스 패딩 및 reshape
        lstm_input_sequence = pad_sequences([keypoints_sequence], maxlen=150, dtype='float32', padding='post', truncating='post')
        lstm_input = np.array(lstm_input_sequence).reshape(1, 150, 99)

        # LSTM 모델 예측
        lstm_prediction = lstm_model.predict(lstm_input)
        if lstm_prediction > 0.5:
            print("Fall confirmed by LSTM. Taking action...")
