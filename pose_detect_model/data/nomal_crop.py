import os
import cv2
import json

def load_json_annotations(json_file):
    """JSON 파일에서 시작 및 종료 프레임을 로드합니다."""
    with open(json_file, 'r') as file:
        data = json.load(file)

        if 'annotations' not in data:
            raise KeyError(f"'annotations' key not found in JSON file: {json_file}")
        
        annotations = data['annotations']
        
        if 'object' not in annotations or not isinstance(annotations['object'], list):
            raise KeyError(f"'object' key not found or not a list in JSON file: {json_file}")
        
        first_object = annotations['object'][0]
        start_frame = int(float(first_object.get('startFrame', 0)))
        end_frame = int(float(first_object.get('endFrame', 0)))

        return start_frame, end_frame

def crop_normal_videos_pre_fall(label_dir, video_dir, crop_dir, context_seconds_before_fall=10, crop_duration=5, fps=30):
    """
    정상적인 행동 비디오를 크롭하는데, 넘어지기 전 구간(10초 전)을 기준으로 5초 구간을 선택하여 저장합니다.

    Args:
        label_dir (str): JSON 라벨 파일들이 위치한 디렉터리.
        video_dir (str): 비디오 파일들이 위치한 디렉터리.
        crop_dir (str): 크롭된 비디오를 저장할 디렉터리.
        context_seconds_before_fall (int): 넘어지기 전 기준 (초 단위).
        crop_duration (int): 각 크롭된 비디오의 길이 (초 단위).
        fps (int): 비디오의 프레임 속도 (기본값 30 FPS).
    """
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    context_frames_before_fall = context_seconds_before_fall * fps  # 넘어지기 전 기준 (예: 10초 = 300프레임)
    crop_frame_count = crop_duration * fps  # 크롭할 구간 길이 (예: 5초 = 150프레임)

    # 라벨 파일 목록을 읽어들임
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".json"):
            continue

        # 라벨 파일 경로 및 비디오 파일 경로 설정
        label_path = os.path.join(label_dir, label_file)
        video_filename = label_file.replace(".json", ".mp4")
        video_path = os.path.join(video_dir, video_filename)

        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_filename} not found.")
            continue

        # 시작 및 종료 프레임 로드
        start_frame, _ = load_json_annotations(label_path)

        # 정상적인 구간 크롭 시작 프레임 계산
        # 넘어지기 10초 전에서 5초 길이를 확보할 수 있는지 확인
        normal_start_frame = max(0, start_frame - context_frames_before_fall)
        normal_end_frame = normal_start_frame + crop_frame_count

        if normal_end_frame > start_frame:
            print(f"Warning: Not enough frames for normal crop in video {video_filename}")
            continue

        # 비디오 로드 및 크롭된 구간 추출
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cropped_video_filename = f"{video_filename}"
        cropped_video_path = os.path.join(crop_dir, cropped_video_filename)
        out = cv2.VideoWriter(cropped_video_path, fourcc, fps, (width, height))

        # 크롭된 구간 저장
        cap.set(cv2.CAP_PROP_POS_FRAMES, normal_start_frame)
        for frame_idx in range(normal_start_frame, normal_end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        # 비디오 파일 닫기
        cap.release()
        out.release()

        print(f"Saved normal cropped video to: {cropped_video_path}")

if __name__ == "__main__":
    label_dir = "/media/bh/PortableSSD/labelnvideo/label"
    video_dir = "/media/bh/PortableSSD/labelnvideo/video"
    crop_dir = "/media/bh/PortableSSD/labelnvideo/normal_crop"
    crop_normal_videos_pre_fall(label_dir, video_dir, crop_dir)
