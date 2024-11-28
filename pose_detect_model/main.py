from process.VideoProcess import VideoProcess
from process.CameraProcess import CameraProcess
from process.MediapipeProcess import MediapipeProcess
from process.YoloProcess import YoloProcess
from PipedProcess import Pipeline
from fall_detection_lstm import lstm_fall_verification
import cv2


if __name__ == "__main__":
    camera_process = CameraProcess(camera_id=0)
    mediapipe_process = MediapipeProcess(
        model_params={
            'min_detection_confidence': 0.6,
            'min_tracking_confidence': 0.6,
        }
    )

    pipeline = Pipeline(camera_process, mediapipe_process)

    try:
        pipeline.start()
        while True:
            # 파이프라인의 마지막 단계에서 fall_detected 플래그 확인
            img, fall_detected = mediapipe_process.get_output()

            # LSTM을 사용한 폴 검증 호출
            lstm_fall_verification(img, fall_detected, mediapipe_process.model)

            # 실시간 영상 유지
            if img is not None:
                cv2.imshow('Fall Detection', img)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Pipeline stopped manually.")
    finally:
        camera_process.release()
        mediapipe_process.release()
        cv2.destroyAllWindows()