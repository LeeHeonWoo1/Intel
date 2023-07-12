# def cal_perceptron_score(x1, x2, x3, w1, w2, w3, tetha) -> int:
#     comp_value = x1*w1 + x2*w2 + x3*w3
#     if comp_value > tetha:
#         return 1
#     else:
#         return 0
    
# result = cal_perceptron_score(10, 10, 20, 0.3, 0.2, 0.5, 60)
# print(result)

def cal_perceptron_score_2(x1, x2, x3, x4, w1, w2, w3, w4, tetha):
    print(x1*w1 + x2*w2* + x3*w3 + x4*w4 > tetha)
    print(x1*w1 + x2*w2* + x3*w3 + x4*w4)
    return 1 if x1*w1 + x2*w2 + x3*w3 + x4*w4 > tetha else 0

result = cal_perceptron_score_2(80, 55, 90, 73, 0.35, 0.3, 0.2, 0.15, 40)
print(result)