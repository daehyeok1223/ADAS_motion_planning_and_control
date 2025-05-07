import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_Measure_init, step_time=0.1, m=0.1,
                 modelVariance=0.01, measureVariance=1.0, errorVariance_init=10.0):
        self.A = 1.0  # 상태 전이
        self.B = step_time / m  # 입력에 대한 영향
        self.C = 1.0  # 관측 모델
        self.D = 0.0
        self.Q = modelVariance  # 프로세스 잡음 공분산
        self.R = measureVariance  # 측정 잡음 공분산
        self.x_estimate = y_Measure_init  # 초기 추정값
        self.P_estimate = errorVariance_init  # 초기 오차 공분산

    def estimate(self, y_measure, input_u):
        # 1. 예측 (Prediction)
        x_predict = self.A * self.x_estimate + self.B * input_u
        P_predict = self.A * self.P_estimate * self.A + self.Q

        # 2. 갱신 (Update)
        K = P_predict * self.C / (self.C * P_predict * self.C + self.R)
        self.x_estimate = x_predict + K * (y_measure - self.C * x_predict)
        self.P_estimate = (1 - K * self.C) * P_predict

if __name__ == "__main__":
    signal = pd.read_csv("01_filter/Data/example_KalmanFilter_1.csv")

    signal["y_estimate"] = 0.0
    y_estimate = KalmanFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i], signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate

    plt.figure()
    plt.plot(signal.time, signal.y_measure, 'k.', label="Measure")
    plt.plot(signal.time, signal.y_estimate, 'r-', label="Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
