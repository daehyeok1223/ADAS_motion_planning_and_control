from ex01_AverageFilter import AverageFilter
from ex02_MovingAverageFilter import MovingAverageFilter
from ex03_LowPassFilter import LowPassFilter
from ex04_KalmanFilter import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 데이터 불러오기
    signal = pd.read_csv("01_filter/Data/example_KalmanFilter_1.csv")
    
    t = list(signal["time"])
    y_measure = list(signal["y_measure"])
    u = list(signal["u"])

    # 필터 인스턴스 초기화
    y_estimate_AF = AverageFilter(y_measure[0], window_size=5)
    y_estimate_MAF = MovingAverageFilter(y_measure[0], num_average=5)
    y_estimate_LPF = LowPassFilter(y_measure[0], alpha=0.9)
    y_estimate_KF = KalmanFilter(y_measure[0])

    # 결과 저장 리스트
    y_AF = []
    y_MAF = []
    y_LPF = []
    y_KF = []

    for i in range(len(signal)):
        # 평균 필터
        y_estimate_AF.estimate(y_measure[i])
        y_AF.append(y_estimate_AF.y_estimate)

        # 이동 평균 필터
        y_estimate_MAF.estimate(y_measure[i])
        y_MAF.append(y_estimate_MAF.y_estimate)

        # 로우패스 필터
        y_estimate_LPF.estimate(y_measure[i])
        y_LPF.append(y_estimate_LPF.y_estimate)

        # 칼만 필터
        y_estimate_KF.estimate(y_measure[i], u[i])
        y_KF.append(y_estimate_KF.x_estimate)

    # 그래프 출력
    plt.figure(figsize=(12, 6))
    plt.plot(t, y_measure, 'k.', label="Measured")
    plt.plot(t, y_AF, 'm-', label="Average Filter")
    plt.plot(t, y_MAF, 'b-', label="Moving Average Filter")
    plt.plot(t, y_LPF, 'c-', label="Low Pass Filter")
    plt.plot(t, y_KF, 'r-', label="Kalman Filter")
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Comparison of Filters')
    plt.legend(loc="best")
    plt.grid(True)
    plt.yticks([-5, 0, 5, 10, 15])  # Y축 눈금 고정
    plt.ylim(-5, 15)                # Y축 범위 고정
    plt.tight_layout()
    plt.show()
