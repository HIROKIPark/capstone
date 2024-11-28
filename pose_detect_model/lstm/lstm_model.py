# lstm_model.py
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# 모델 학습 및 저장
def train_and_save_model(padded_sequences, y_train):
    input_shape = (padded_sequences.shape[1], padded_sequences.shape[2])
    model = build_lstm_model(input_shape)
    model.fit(padded_sequences, y_train, epochs=10, batch_size=16, validation_split=0.2)
    model.save("lstm_fall_detection_model.h5")