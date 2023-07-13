def discriminator(): # 단일 고객 대상
    year = input("단일 고객 대상 프로그램입니다. 태어난 년도를 입력하세요 : ") # input함수를 이용해 태어난 년도를 입력받는다.
    birth_year = int(year) # input 함수의 경우, 기본적인 return 타입이 string(문자열)이기 때문에 int()라는 클래스를 이용해서 정수로 바꿔준다.
    ticket_price = 10000 # 티켓 가격을 ticket_price라는 변수명에 저장한다.
    animal_no = (birth_year + 8) % 12 # 공식에 의거, 값을 계산한다.
    animal = ('쥐', '소', '호랑이', '토끼', '용', '뱀', '말', '양', '원숭이', '닭', '개', '돼지') # 각 띠 정보가 담긴 튜플 생성
    
    lst_result = animal[animal_no] # 위에서 계산한 값을 토대로 animal이라는 튜플에 담긴 띠 값을 가져온다(인덱싱(indexing))
    if lst_result == '토끼': # 만약 계산한 결과가 "토끼"와 같은 값이라면
        # 문자열 formatting을 이용해서 해당 문자열을 출력하고,
        print("계묘년 기념 토끼띠 반값 할인 이벤트 대상자입니다. 할인된 티켓 가격은 {0}원 입니다.".format(int(ticket_price/2))) # 나눗셈 연산자를 이용해 계산한 결과는 실수로 출력되기에, 정수형으로 바꿔준다.
    else: # 아니라면
        print("티켓 가격은 {0}원 입니다.".format(ticket_price)) # 원래 티켓 가격을 출력한다,

def discriminator2(): # 단체 고객 대상
    year = input("단체 대상 전용 프로그램입니다. 모든 고객님의 출생 년도를 띄어쓰기를 기준으로 작성하세요 : ") # 모든 고객의 출생 년도를 띄어쓰기를 기준으로 입력받는다.
    birth_list = list(map(int, year.split(' '))) # map : 배열 내의 원소를 내가 원하는 자료형으로 바꿀 수 있는 클래스. list()라는 클래스로 감싸줘야 리스트 형태로 출력된다.
    animal = ('쥐', '소', '호랑이', '토끼', '용', '뱀', '말', '양', '원숭이', '닭', '개', '돼지') # 각 띠 정보다 담긴 튜플 생성
    
    rab_count = 0 # 토끼띠 고객이 몇 명 존재하는지 카운트 하기 위한 변수를 생성한다. 초기값은 0명.
    for element in birth_list: # birth_list라는 배열 내의 원소 하나 하나를 element라는 변수 이름으로 모두 참조하겠다. (반복문)
        animal_no = (element + 8) % 12 # 공식에 의거, 계산
        if animal[animal_no] == "토끼": # 계산한 결과가 "토끼"와 같은 문자열이라면, 
            rab_count += 1 # 토끼띠 고객 수를 늘린다.

    # 모든 반복문이 끝난 이후에, 
    
    # 변수들을 종합해서 아래의 문자열을 출력한다.
    print(f"단체 고객 {len(birth_list)}명 중 토끼띠이신 고객님은 {rab_count}명 이므로, 총 가격 {10000*len(birth_list)}원 중 {int(10000/2*rab_count)}원 할인되었습니다.")
