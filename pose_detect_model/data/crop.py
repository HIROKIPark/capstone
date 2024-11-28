import os
import cv2
import json

def load_json_annotations(json_file):
    """JSON 파일에서 시작 및 종료 프레임을 로드합니다."""
    with open(json_file, 'r') as file:
        data = json.load(file)
        print(f"Loaded JSON from {json_file}: {data}")  # JSON 데이터 구조 확인을 위한 출력
        
        # 이후 기존 로직대로 키 확인 및 데이터 추출
        if 'annotations' not in data:
            raise KeyError(f"'annotations' key not found in JSON file: {json_file}")
        
        annotations = data['annotations']

        if 'object' not in annotations or not isinstance(annotations['object'], list):
            raise KeyError(f"'object' key not found or not a list in JSON file: {json_file}")

        first_object = annotations['object'][0]

        start_frame = first_object.get('startFrame')
        end_frame = first_object.get('endFrame')

        if start_frame is None or end_frame is None:
            raise KeyError(f"'startFrame' or 'endFrame' key not found in JSON file: {json_file}")

        return int(float(start_frame)), int(float(end_frame))


def crop_videos(label_dir, video_dir, crop_dir, context_frames=30):
    """
    비디오를 크롭하고 넘어지는 순간 전후의 문맥을 포함하여 저장합니다.

    Args:
        label_dir (str): JSON 라벨 파일들이 위치한 디렉터리.
        video_dir (str): 비디오 파일들이 위치한 디렉터리.
        crop_dir (str): 크롭된 비디오를 저장할 디렉터리.
        context_frames (int): 넘어짐 전후로 추가할 문맥 프레임 수.
    """
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    # 라벨 파일 목록을 읽어들임
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".json"):
            continue

        # 라벨 파일 경로 및 비디오 파일 경로 설정
        label_path = os.path.join(label_dir, label_file)
        video_filename = label_file.replace(".json", ".mp4")
        video_path = os.path.join(video_dir, video_filename)
        cropped_video_path = os.path.join(crop_dir, video_filename)

        # 이미 크롭된 비디오가 존재하는 경우 스킵
        if os.path.exists(cropped_video_path):
            print(f"Skipping already processed video: {video_filename}")
            continue

        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_filename} not found.")
            continue

        # 시작 및 종료 프레임 로드
        start_frame, end_frame = load_json_annotations(label_path)

        # 비디오 로드 및 크롭된 구간 추출
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 문맥 프레임을 추가하여 시작 및 종료 프레임 조정
        start_frame = max(0, start_frame - context_frames)
        end_frame = min(total_frames - 1, end_frame + context_frames)

        # 크롭된 비디오 저장을 위한 VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cropped_video_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 지정된 프레임 구간에서만 비디오에 저장
            if start_frame <= frame_idx <= end_frame:
                out.write(frame)

            frame_idx += 1
            if frame_idx > end_frame:
                break

        # 비디오 파일 닫기
        cap.release()
        out.release()

        print(f"Saved cropped video to: {cropped_video_path}")

if __name__ == "__main__":
    label_dir = "/media/bh/PortableSSD/labelnvideo/label"
    video_dir = "/media/bh/PortableSSD/labelnvideo/video"
    crop_dir = "/media/bh/PortableSSD/labelnvideo/crop"
    crop_videos(label_dir, video_dir, crop_dir)

