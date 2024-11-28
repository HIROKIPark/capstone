import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import mediapipe as mp
from process.MediapipeProcess import MediapipeProcess

# 관절 좌표 추출 함수
def extract_keypoints(results, prev_keypoints=None):
    keypoints = []
    if results and results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
    else:
        if prev_keypoints is not None:
            keypoints = prev_keypoints
        else:
            keypoints = [[0, 0, 0]] * 33  # 기본값 반환
    return np.array(keypoints).flatten()

# 비디오 처리 및 관절 좌표 저장 함수
def process_videos(video_dirs, output_dir):
    # MediaPipe 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    feature_sequences = []
    labels = []

    for label, video_dir in video_dirs.items():
        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}")
                continue

            keypoints_sequence = []
            prev_keypoints = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. 관절 랜드마크 추출 (MediaPipe)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_frame)
                keypoints = extract_keypoints(pose_results, prev_keypoints)

                # 관절 좌표 추가
                keypoints_sequence.append(keypoints)
                prev_keypoints = keypoints

            cap.release()

            # 2. 시퀀스 패딩 (길이 150 프레임으로 고정)
            if keypoints_sequence:
                target_length = 150
                padded_sequence = pad_sequences([keypoints_sequence], maxlen=target_length, dtype='float32', padding='post', truncating='post')
                feature_sequences.append(padded_sequence[0])
                labels.append(label)

                # 3. 특징 벡터 저장
                feature_file_name = f"{os.path.splitext(video_file)[0]}_{label}_features.npy"
                feature_path = os.path.join(output_dir, feature_file_name)
                np.save(feature_path, padded_sequence[0])

    return feature_sequences, labels

# 특징 저장 실행
if __name__ == "__main__":
    video_dirs = {
        0: "/media/bh/PortableSSD/labelnvideo/normal_crop",  # 정상 비디오 디렉터리
        1: "/media/bh/PortableSSD/labelnvideo/crop"  # 비정상 비디오 디렉터리
    }
    output_dir = "/media/bh/PortableSSD/labelnvideo/features"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_videos(video_dirs, output_dir)
