import numpy as np, matplotlib.pyplot as plt

alpha = 1.0 # learning rate
epoch = 5000 # number of learning iteration

# 가중치, 편향 초기값 설정
w1 = 1.0 # 히든 레이어의 1번째 가중치
w2 = -1.0 # 2번째 가중치
w3 = 2.0 # 3번째 가중치
w4 = -2.0 # 4번째 가중치
b1 = -1.0 # 1번째 편향
b2 = 1.0 # 2번째 편향
b3 = 2.0 # 3번째 편향

def sigmoid(x): # 활성화 함수 정의
    y = 1/(1+np.exp(-x))
    return y

input_data = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) # 10개의 입력데이터
teaching_data = [sigmoid(i) for i in input_data]

result_array = [] # 각 학습에 대한 모든 결과를 담을 빈 리스트 생성
for n in range(1, epoch + 1): # 5000회의 학습 진행
    result_per_each = [] # 회당 학습 결과를 담을 리스트 생성 
    for i in range(len(input_data)): 
        x = input_data[i] # input값을 x에 할당
        t = teaching_data[i] # 타겟값은 t에 할당    
        # 순전파 계산
        u1 = sigmoid(w1*x + b1) # -5
        u2 = sigmoid(w2*x + b2) # 5
        y = sigmoid(w3*u1 + w4*u2 + b3)
        result_per_each.append(y)
        # 역전파 계산
        E = 0.5*(y-t)**2 # 손실함수
        dE_dw_3 = (y-t)*(1-y)*y*u1 # 출력층 가중치
        dE_dw_4 = (y-t)*(1-y)*y*u2 # 출력층 가중치2
        dE_db_3 = (y-t)*(1-y)*y # 출력층 편향값
        
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
    result_array.append(result_per_each)
        
    print(f"{n} Epoch - Error : {E}")
    
x = 0.5
u1 = sigmoid(w1*x + b1)
u2 = sigmoid(w2*x + b2)
y = sigmoid(w3*u1 + w4*u2 + b3)

print(f"실측값 t : {sigmoid(x)}")
print(f"신경망에 의해 예측된 값 y : {y}")

plt.plot(input_data, teaching_data)
for i in range(len(result_array)):
    plt.plot(input_data, result_array[i])
plt.show()