def cal_perceptron_score(x1, x2, x3, w1, w2, w3, tetha) -> int:
    """
    각 입력값을 가중치에 따라 계산하여 기준치에 따라 결과를 리턴하는 함수입니다. \n
    x1, x2, x3 (int) : 각 조건을 의미하는 값입니다. \n
    w1, w2, w3 (float) : 각 조건을 생각하는 중요도(가중치) 입니다. 0~1사이의 확률값입니다. \n
    tetha (int) : 특정 행동을 실행하기 위한 기준치 입니다. \n
    """
    comp_value = x1*w1 + x2*w2* + x3*w3
    if comp_value > tetha:
        return 1
    else:
        return 0
    
result = cal_perceptron_score(10, 10, 20, 0.3, 0.2, 0.5, 60)
print(result)

def cal_perceptron_score_2(x1, x2, x3, w1, w2, w3, tetha):
    return 1 if x1*w1 + x2*w2* + x3*w3 > tetha else 0

result = cal_perceptron_score_2(10, 10, 20, 0.3, 0.2, 0.5, 60)
print(result)           