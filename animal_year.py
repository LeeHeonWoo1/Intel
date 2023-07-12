year = input("태어난 년도를 입력하세요 : ")
birth_year = int(year)
animal_no = (birth_year + 8)%12
animal = ('쥐', '소', '호랑이', '토끼', '용', '뱀', '말', '양', '원숭이', '닭', '개', '돼지')
print("당신의 띠는", animal[animal_no], sep=":", end="!!")