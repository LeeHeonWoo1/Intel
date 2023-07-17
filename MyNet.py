import numpy as np

class MyNet:
    """
    2x3x1 신경망(사용자의 몸무게, 운동 시간을 받아 하루에 감량할 수 있는 몸무게를 예측하는 클래스)
    """
    def __init__(self, alpha, epochs, kg, time_per_min, MET):
        self.alpha = alpha
        self.epochs = epochs
        self.kg = kg
        self.time = time_per_min
        self.MET = MET
        
    def get_weights(self): # 랜덤으로 가중치 입력받는 메소드
        self.wt = []
        for _ in range(13):
            self.wt.append(np.random.rand())
        return self.wt
        
    def loss_weight(self, kg, time):
        return np.round(((self.MET*3.5*kg*time)/1000)*5 / 7800, 4)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fit_and_predict(self): # 수정된 부분 : 사용자 입력값 두 개 입력받기
        # 학습 시작과 동시에 가중치 배열 생성
        self.wt = self.get_weights()
        input_data = np.array([[80, 120], [100, 180], [90, 60], [72, 30]])
        teaching_data = np.array([[self.loss_weight(80, 120)], [self.loss_weight(100, 180)], [self.loss_weight(90, 60)], [self.loss_weight(72, 30)]])

        for n in range(1, self.epochs+1):
            for i in range(len(input_data)):
                x1 = input_data[i][0] / 74
                x2 = input_data[i][1] / 60
                t = teaching_data[i]
                ########## 순방향 계산 #########
                u1 = self.sigmoid(self.wt[0]*x1 + self.wt[3]*x2 + self.wt[6])
                u2 = self.sigmoid(self.wt[1]*x1 + self.wt[4]*x2 + self.wt[7])
                u3 = self.sigmoid(self.wt[2]*x1 + self.wt[5]*x2 + self.wt[8])
                y = self.wt[9]*u1 + self.wt[10]*u2 + self.wt[11]*u3 + self.wt[12]
                ######## 역방향 계산(오차역전파) ########
                E = 0.5 * (y - t)**2
                dE_dw_0 = (y-t)*self.wt[9]* (1-u1)*u1*x1
                dE_dw_1 = (y-t)*self.wt[10]*(1-u2)*u2*x1
                dE_dw_2 = (y-t)*self.wt[11]*(1-u3)*u3*x1
                dE_dw_3 = (y-t)*self.wt[9]* (1-u1)*u1*x2
                dE_dw_4 = (y-t)*self.wt[10]*(1-u2)*u2*x2
                dE_dw_5 = (y-t)*self.wt[11]*(1-u3)*u3*x2
                dE_dw_6 = (y-t)*self.wt[9]* (1-u1)*u1
                dE_dw_7 = (y-t)*self.wt[10]*(1-u2)*u2
                dE_dw_8 = (y-t)*self.wt[11]*(1-u3)*u3
                dE_dw_9 = (y-t)*u1
                dE_dw_10 = (y-t)*u2
                dE_dw_11 = (y-t)*u3
                dE_dw_12 = (y-t)
                ########## 가중치 업데이트(경사하강법) #########
                self.wt[0] = self.wt[0] - self.alpha * dE_dw_0
                self.wt[1] = self.wt[1] - self.alpha * dE_dw_1
                self.wt[2] = self.wt[2] - self.alpha * dE_dw_2
                self.wt[3] = self.wt[3] - self.alpha * dE_dw_3
                self.wt[4] = self.wt[4] - self.alpha * dE_dw_4
                self.wt[5] = self.wt[5] - self.alpha * dE_dw_5
                self.wt[6] = self.wt[6] - self.alpha * dE_dw_6
                self.wt[7] = self.wt[7] - self.alpha * dE_dw_7
                self.wt[8] = self.wt[8] - self.alpha * dE_dw_8
                self.wt[9] = self.wt[9] - self.alpha * dE_dw_9
                self.wt[10] = self.wt[10] - self.alpha * dE_dw_10
                self.wt[11] = self.wt[11] - self.alpha * dE_dw_11
                self.wt[12] = self.wt[12] - self.alpha * dE_dw_12
            print("{} EPOCH-ERROR: {}".format(n, E))

        self.kg = self.kg / 74
        self.time = self.time / 60
        u1 = self.sigmoid(self.wt[0]*self.kg + self.wt[3]*self.time + self.wt[6])
        u2 = self.sigmoid(self.wt[1]*self.kg + self.wt[4]*self.time + self.wt[7])
        u3 = self.sigmoid(self.wt[2]*self.kg + self.wt[5]*self.time + self.wt[8])
        y = self.wt[9]*u1 + self.wt[10]*u2 + self.wt[11]*u3 + self.wt[12]
        
        return y