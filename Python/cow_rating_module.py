def rating():
    marble = int(input("소의 지방률이 몇% 인가요? : "))
    if marble >= 90:
        print("지방률 {0}%를 가진 소 이므로 {1}등급 입니다! b".format(marble, 1))
    elif marble >= 80 :
        print("지방률 {0}%를 가진 소 이므로 {1}등급 입니다. :)".format(marble, 2))
    elif marble >= 70 :
        print("지방률 {0}%를 가진 소 이므로 {1}등급 입니다.".format(marble, 3))
    else :
        print("지방률 {0}%를 가진 소 이므로 {1}등급 입니다..ㅠㅠ".format(marble, 4))
        
        
rating()