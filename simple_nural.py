import numpy as np

alpha = 1.0
epoch = 5000

w1 = 1.0
w2 = -1.0
w3 = 2.0
w4 = -2.0
b1 = -1.0
b2 = 1.0
b3 = 2.0

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

input_data = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) # 10개의 입력데이터
teaching_data = []
for i in input_data:
    teaching_data.append(sigmoid(i)) # 각 입력값에 대한 sigmoid 계산값을 빈 리스트에 저장

for n in range(1, epoch + 1): # 5000회의 학습 진행
    for i in range(len(input_data)): 
        x = input_data[i] # input값을 x에 할당
        t = teaching_data[i] # 학습 데이터는 t에 할당
        # 순전파 계산
        u1 = sigmoid(w1*x + b1)
        u2 = sigmoid(w2*x + b2)
        y = sigmoid(w3*u1 + w4*u2 + b3)
        # 역전파 계산
        E = 0.5*(y-t)**2 # 손실함수
        dE_dw_3 = (y-t)*(1-y)*y*u1 # 출력층 가중치
        dE_dw_4 = (y-t)*(1-y)*y*u2 # 출력층 가중치2
        dE_db_3 = (y-t)*(1-y)*y # 출력층 기준치
        
        dE_dw_1 = (y-t)*(1-y)*y*w3*(1-u1)*u1*x # 은닉층 가중치
        dE_dw_2 = (y-t)*(1-y)*y*w4*(1-u2)*u2*x # 은닉층 가중치
        dE_db_1 = (y-t)*(1-y)*y*w3*(1-u1)*u1 # 은닉층 기준치
        dE_db_2 = (y-t)*(1-y)*y*w4*(1-u2)*u2 # 은닉층 기준치
        
        # 가중치, 기준치 업데이트(Gradient Descent)
        w1 = w1 - alpha * dE_dw_1
        w2 = w2 - alpha * dE_dw_2
        w3 = w3 - alpha * dE_dw_3
        w4 = w4 - alpha * dE_dw_4
        b1 = b1 - alpha * dE_db_1
        b2 = b2 - alpha * dE_db_2
        b3 = b3 - alpha * dE_db_3
        
    print(f"{n} Epoch - Error : {E}")
 
x = 0.5
u1 = sigmoid(w1*x + b1)
u2 = sigmoid(w2*x + b2)
y = sigmoid(w3*u1 + w4*u2 + b3)
print("신경망의 예측값 {}".format(round(y, 2))) 
print("계산된 값(정답) : {}".format(round(sigmoid(x), 2)))