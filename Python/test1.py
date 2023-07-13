from discount_ticket import *
from cow_rating_module import rating

while True: # 특정 반복문 탈출 지점(break문)을 만나기 전까지는 계속 반복문 실행
    print("사용할 수 있는 서비스는 아래와 같습니다.", "1. 소 지방률 계산", "2. 영화관 특별 이벤트", "3. 종료하기", sep="\n") # sep이라는 옵션과 \n을 이용해 각 문자열을 enter + 출력 하는 효과.
    service_num = int(input("어떤 서비스를 사용하시겠습니까? : ")) # 고객이 이용하고자 하는 서비스 번호를 받아 각 조건문에 따라 코드 실행
    if service_num == 1: # 서비스가 1번 이라면
        print("소 등급 판별기 서비스를 선택하셨습니다. ") # 안내 문구 출력과 동시에
        rating() # 소 등급 판별 함수를 실행하고
        print("="*60) # 구분선을 출력한다. break문을 만나지 않았기 때문에 다시 코드의 최상단으로 올라간다.
    elif service_num == 2: # 서비스가 2번이라면
        print("영화관 특별 이벤트 탭을 선택하셨습니다. ") # 안내 문구 출력과 동시에
        person = int(input("이용하시는 고객님의 수를 입력해주세요 : ")) # 인원수를 입력받고,
        if person == 1: # 인원수가 1명이라면
            discriminator() # 단일 대상 프로그램을 실행하고, 
            print("="*60) # 함수가 종료되면 구분선을 사용한다.
        else: # 인원수가 다인원이라면
            discriminator2() # 다중 대상 프로그램을 실행한다.
            print("="*60) # 구분선 출력
    else: # 서비스가 3번이라면
        print("프로그램을 종료합니다. 이용해주셔서 감사합니다.") # 종료 안내문구 출력과 동시에
        break # 반복문을 탈출한다
            
        