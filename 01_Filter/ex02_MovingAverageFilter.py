import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MovingAverageFilter:
    def __init__(self, y_initial_measure, num_average=5):
        self.num_average = num_average                    # 평균 낼 개수
        self.buffer = [y_initial_measure]                 # 초기 측정값으로 버퍼 시작
        self.y_estimate = y_initial_measure               # 첫 추정값도 초기값과 동일하게 설정
        
    def estimate(self, y_measure):
        self.buffer.append(y_measure)                     # 새로운 측정값 추가
        if len(self.buffer) > self.num_average:           # 만약 버퍼가 너무 길어지면
            self.buffer.pop(0)                            # 가장 오래된 값 제거
        self.y_estimate = np.mean(self.buffer)            # 버퍼의 평균값 계산하여 추정값 업데이트

    
if __name__ == "__main__":
    # CSV 파일 읽기
    # signal = pd.read_csv("week_01_filter/Data/example_Filter_1.csv")      
    signal = pd.read_csv("01_Filter/Data/example_Filter_2.csv")

    signal['y_estimate'] = 0.0                             # y_estimate 열 초기화

    y_estimator = MovingAverageFilter(signal.y_measure[0], num_average=15)  # 필터 생성

    for i, row in signal.iterrows():
        y_estimator.estimate(signal.y_measure[i])          # 새 측정값으로 필터 업데이트
        signal.y_estimate[i] = y_estimator.y_estimate      # 추정값 기록

    # 그래프 출력
    plt.figure()
    plt.plot(signal.time, signal.y_measure, 'k.', label="Measure")
    plt.plot(signal.time, signal.y_estimate, 'r-', label="Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)

    plt.ylim(-5, 15)  # Y축 범위 지정: -5 ~ 15

    plt.show()
