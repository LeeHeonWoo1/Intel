# Mediapipe에서 제공하는 Hands함수를 통해 손의 21개 랜드마크를 획득
# MCP: 밑쪽 관절, PIP: 중간관절, DIP: 끝쪽 관절, CMC: 엄지 뿌리관절
# 0:Wrist,             1:Thumb_CMC,          2:Thumb_MCP,   3:Thumb_IP,   4:Thumb_Tip
# 5:Index_Finger_MCP,  6:Index_Finger_PIP,   7:Index_Finger_DIP,   8:Index_Finger_Tip
# 9:Middle_Finger_MCP, 10:Middle_Finger_PIP, 11:Middle_Finger_DIP, 12:Middle_Finger_Tip
# 13:Ring_Finger_MCP,  14:Ring_Finger_PIP,   15:Ring_Finger_DIP,   16:Ring_Finger_Tip,
# 17:Pinky_MCP,        18:Pinky_PIP,         19:Pinky_DIP,         20:Pinky_Tip  

import cv2
import mediapipe as mp          # 손과 랜드마크 인식용
import time                     # 영상 프레임 속도 (FPS) 체크용

cap = cv2.VideoCapture(1)       ###### 영상 획득 기본 코드 ###### 0:첫번째 캠

mpHand = mp.solutions.hands     # Mediapipe를 통한 손 검출
hands = mpHand.Hands()          # Hands()함수 파라메터를 살펴볼 것. 
mpDraw = mp.solutions.drawing_utils  # 영상위에 그림을 그려주기 위한 모듈

pTime = 0
cTime = 0

while True:
    success, img = cap.read()   ###### 영상 획득 기본 코드 ######
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR을 RGB로 변환
    results = hands.process(imgRGB)  # RGB영상으로 바꾸고 hands모듈 수행

    if results.multi_hand_landmarks:    # 두개의 손의 랜드마크가 잡힌다면
        for handLms in results.multi_hand_landmarks:  # 랜드마크와 연결선을 표시
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape  # 전체 영상 이미지의 높이, 폭, 채널 획득
                cx, cy = int(lm.x*w), int(lm.y*h) # 영상에서 랜드마크의 위치점 획득
                print(id, cx, cy)  # 랜드마크 번호, x위치, y위치 표시
                
                if id == 0:  # id 0번(손목)의 위치에 핑크색 원을 표시
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)
                
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

    cTime = time.time()     # time함수를 써서 현재시간 체크
    fps = 1/(cTime-pTime)   # 현재 프레임과 전 프레임의 시간간격으로 fps계산 
    pTime = cTime           # 현재시간을 전시간으로 설정

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)    ###### 영상 획득 기본 코드 ######
    if cv2.waitKey(33) == ord('q'):     # 키보드의 q 키가 입력되면 정지
        break