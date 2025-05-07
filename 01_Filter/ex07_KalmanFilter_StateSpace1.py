import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_Measure_init, step_time=0.1, m=1.0,
                 Q_x=0.02, Q_v=0.05, R=5.0, errorCovariance_init=10.0):
        # 시스템 모델 정의
        self.A = np.array([[1.0, step_time],
                           [0.0, 1.0]])
        self.B = np.array([[0.0],
                           [step_time / m]])
        self.C = np.array([[1.0, 0.0]])  # 위치만 측정
        self.D = 0.0

        # 잡음 공분산
        self.Q = np.array([[Q_x, 0.0],
                           [0.0, Q_v]])  # 상태 노이즈
        self.R = R                      # 측정 노이즈

        # 초기 추정값
        self.x_estimate = np.array([[y_Measure_init], [0.0]])  # 위치, 속도
        self.P_estimate = np.array([[errorCovariance_init, 0.0],
                                    [0.0, errorCovariance_init]])

    def estimate(self, y_measure, input_u):
        # 1. 예측
        x_predict = self.A @ self.x_estimate + self.B * input_u
        P_predict = self.A @ self.P_estimate @ self.A.T + self.Q

        # 2. 갱신
        S = self.C @ P_predict @ self.C.T + self.R
        K = P_predict @ self.C.T @ np.linalg.inv(S)
        self.x_estimate = x_predict + K @ (y_measure - self.C @ x_predict)
        self.P_estimate = (np.eye(2) - K @ self.C) @ P_predict


if __name__ == "__main__":
    # 데이터 불러오기
    signal = pd.read_csv("01_filter/Data/example07.csv")
    signal["y_estimate"] = 0.0
    signal["v_estimate"] = 0.0

    # 칼만 필터 초기화
    y_estimate = KalmanFilter(signal.y_measure[0])

    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i], signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate[0][0]  # 위치 추정
        signal.v_estimate[i] = y_estimate.x_estimate[1][0]  # 속도 추정

    # 위치 추정 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(signal.time, signal.y_measure, 'k.', label="Measured Position")
    plt.plot(signal.time, signal.y_estimate, 'r-', label="Estimated Position")
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title("Kalman Filter (State-Space, Position Measurement)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 속도 추정 시각화 (선택사항)
    plt.figure(figsize=(10, 5))
    plt.plot(signal.time, signal.v_estimate, 'b-', label="Estimated Velocity")
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.title("Estimated Velocity over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
