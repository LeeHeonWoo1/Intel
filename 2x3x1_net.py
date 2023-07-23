import numpy as np

# 엄마는 얼마나 화를 낼까?

alpha = 0.3
epochs = 1000

wt = []
for i in range(13):
    w = np.random.rand()
    wt.append(w)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 입력(input)값과 정답(teaching data)
input_data = np.array([[0,0], [0,1], [1,0], [1,1]]) # 각 배열이 의미하는 것 : [게임하는 시간, 공부하는 시간]
teaching_data = np.array([[-1], [3], [-3], [1]]) # 엄마의 기분에 따른 값(값이 작을수록 화가 많이 난 상태를 의미)

for n in range(1, epochs+1):
    for i in range(len(input_data)):
        x1 = input_data[i][0] # 게임하는 시간
        x2 = input_data[i][1] # 공부하는 시간
        t = teaching_data[i] # 엄마의 기분
        ########## 순방향 계산 #########
        u1 = sigmoid(wt[0]*x1 + wt[3]*x2 + wt[6])
        u2 = sigmoid(wt[1]*x1 + wt[4]*x2 + wt[7])
        u3 = sigmoid(wt[2]*x1 + wt[5]*x2 + wt[8])
        y = wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + wt[12]
        ######## 역방향 계산(오차역전파) ########
        E = 0.5 * (y - t)**2
        dE_dw_0 = (y-t)*wt[9]*(1-u1)*u1*x1
        dE_dw_1 = (y-t)*wt[10]*(1-u2)*u2*x1
        dE_dw_2 = (y-t)*wt[11]*(1-u3)*u3*x1
        dE_dw_3 = (y-t)*wt[9]* (1-u1)*u1*x2
        dE_dw_4 = (y-t)*wt[10]*(1-u2)*u2*x2
        dE_dw_5 = (y-t)*wt[11]*(1-u3)*u3*x2
        dE_dw_6 = (y-t)*wt[9]* (1-u1)*u1
        dE_dw_7 = (y-t)*wt[10]*(1-u2)*u2
        dE_dw_8 = (y-t)*wt[11]*(1-u3)*u3
        dE_dw_9 = (y-t)*u1
        dE_dw_10 = (y-t)*u2
        dE_dw_11 = (y-t)*u3
        dE_dw_12 = (y-t)
        ########## 가중치 업데이트(경사하강법) #########
        wt[0] = wt[0] - alpha * dE_dw_0
        wt[1] = wt[1] - alpha * dE_dw_1
        wt[2] = wt[2] - alpha * dE_dw_2
        wt[3] = wt[3] - alpha * dE_dw_3
        wt[4] = wt[4] - alpha * dE_dw_4
        wt[5] = wt[5] - alpha * dE_dw_5
        wt[6] = wt[6] - alpha * dE_dw_6
        wt[7] = wt[7] - alpha * dE_dw_7
        wt[8] = wt[8] - alpha * dE_dw_8
        wt[9] = wt[9] - alpha * dE_dw_9
        wt[10] = wt[10] - alpha * dE_dw_10
        wt[11] = wt[11] - alpha * dE_dw_11
        wt[12] = wt[12] - alpha * dE_dw_12
    print("{} EPOCH-ERROR: {}".format(n, E))

# Test: 입력값 x1, x2에 대하여 본 신경망으로 예측(순방향 계산)
x1 = 0.5 # 게임 시간 지정
x2 = 0.5 # 공부 시간 지정
u1 = sigmoid(wt[0]*x1 + wt[3]*x2 + wt[6])
u2 = sigmoid(wt[1]*x1 + wt[4]*x2 + wt[7])
u3 = sigmoid(wt[2]*x1 + wt[5]*x2 + wt[8])
y = wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + wt[12]
print("="*60)
print("엄마의 기분은???")
print("게임:{}시간, 공부:{}시간 엄마 기분:{}".format(x1, x2, y))
print("="*60)