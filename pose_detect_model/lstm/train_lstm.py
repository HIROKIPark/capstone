import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# LSTM 모델 정의
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # 이진 분류
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 특징 벡터 파일 불러오기
if __name__ == "__main__":
    output_dir = "/media/bh/PortableSSD/labelnvideo/features"

    feature_sequences = []
    labels = []
    for file_name in os.listdir(output_dir):
        if file_name.endswith("_features.npy"):
            feature_path = os.path.join(output_dir, file_name)
            feature_sequences.append(np.load(feature_path))
            # 라벨은 파일명에 따라 결정 (normal은 0, unnormal은 1로 라벨링)
            if "_0_" in file_name:
                labels.append(0)
            else:
                labels.append(1)
    print("Loaded features from saved files.")

    # 시퀀스 패딩 (길이 150 프레임으로 고정)
    target_length = 150
    padded_sequences = pad_sequences(feature_sequences, maxlen=target_length, dtype='float32', padding='post', truncating='post')
    y_data = np.array(labels)

    # 데이터 분리 (학습 데이터와 테스트 데이터 분리)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y_data, test_size=0.2, random_state=42)

    # 학습을 위해 모델 빌드
    input_shape = (padded_sequences.shape[1], padded_sequences.shape[2] if len(padded_sequences.shape) > 2 else 1)
    model = build_lstm_model(input_shape)

    # 모델 학습
    model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))

    # 모델 저장
    model.save("lstm_fall_detection_model.h5")

    # 테스트 결과 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")