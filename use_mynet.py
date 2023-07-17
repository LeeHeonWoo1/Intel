from MyNet import MyNet
import numpy as np
import pprint

def start(i):
    MET = float(input(""" 각 운동 종류에 따른 운동 계수(MET)는 다음과 같습니다.
1. 매우 가벼운 운동(볼링, 배구, 자전거) : 3.0 
2. 체조(중간 강도) : 3.5 
3. 경보(평지, 조금 빠르게) : 3.8
4. 속보(평지, 95~100m/m) : 4.0
5. 배드민턴 : 4.5 
6. 발레, 트위스트, 재즈, 탭댄스 : 4.8 
7. 웨이트 트레이닝(고강도, 파워리프팅) : 6.0 
8. 에어로빅 : 6.5 
9. 조깅, 축구, 테니스 : 7.0 
10. 등산(1~2kg 가량의 가방을 메고) : 7.5
-------------------------------------------------------------------------
위 운동 계수 중 하나를 선택하여 입력하거나 다른 운동 계수를 입력하세요 : """))
    print("-"*60)
    lr = float(input("학습률을 입력하세요 : "))
    print("-"*60)
    epochs = int(input("학습 횟수를 입력하세요 : "))
    print("-"*60)
    kg = int(input("체중을 입력하세요 : "))
    print("-"*60)
    time = int(input("운동할 시간을 분 단위로 입력해주세요 : "))
    print("-"*60)

    # 모델 객체 생성
    model = MyNet(lr, epochs, kg, time, MET)

    # 모델 훈련/예측 메소드 실행
    y = model.fit_and_predict(kg, time)
    print("="*60)
    print("체중 : {}kg, 운동 시간 : {}분, 실제 감량 체중 : [{}]kg".format(kg, time, model.loss_weight(kg, time)))
    print("체중 : {}kg, 운동 시간 : {}분, 예측 감량 체중 : [{}]kg".format(kg, time, np.round(y, 4)[0]))
    print("="*60)
    
    # 수정된 부분 : 각 지정한 횟수마다 학습이 끝나면, 각 정보를 담아서 dict에 저장
    train_result[f"result of {i} time trainning"] = {"learning_rate" : lr, "epochs": epochs, "predicted" : np.round(y, 4)[0], "origin_value" : model.loss_weight(kg, time)}

def show_results(): # 수정된 부분 : 각 지정한 횟수 마다 학습한 결과를 볼 수 있는 dict 출력
    pp = pprint.PrettyPrinter(indent = 4)
    pp.pprint(train_result)

def start2():
    pass

if __name__ == "__main__":
    train_result = {}
    i = 1
    while True:
        print("="*60)
        print("1. 감량 체중 예측 \n2. XOR 게이트 계산 \n3. 훈련 결과 보기 \n4. 종료")
        print("="*60)
        key = input("실행할 메뉴의 번호를 입력하세요 : ")
        if key == "1":
            start(i)
            i += 1
        elif key == "2":
            start2()
        elif key == "3":
            show_results()
        else:
            break