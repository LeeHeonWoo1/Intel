import cv2
import os 
import datetime 
from imutils.video import WebcamVideoStream

host = "{}:4747/video".format("http://192.168.0.39")
cam = WebcamVideoStream(src=host).start()    # 비디오 스트림 시작. capture = cv2.VideoCapture(0) 부분에 해당 

while True:    # q키 입력으로 영상 종료
    frame = cam.read()    # 웹캠 영상을 읽어와 실시간으로 뿌림. ret, frame = capture.read() 에 해당  
    cv2.imshow('Original Video', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('1'): # 1키가 눌리면
        file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'
        path = r'D:\Intel\projects\NavigationPt\image\0_lego'
        cv2.imwrite(os.path.join(path , file), frame) # 경로와 파일명을 합쳐서 저장
        print(file, '_0_ saved')
    elif key == ord('2'): # 2키가 눌리면
        file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'
        path = r'D:\Intel\projects\NavigationPt\image\1_airpods'
        cv2.imwrite(os.path.join(path , file), frame) # 경로와 파일명을 합쳐서 저장
        print(file, '_1_ saved')
    elif key == ord('3'): # 3키가 눌리면
        file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'
        path = r'D:\Intel\projects\NavigationPt\image\2_charger'
        cv2.imwrite(os.path.join(path , file), frame) # 경로와 파일명을 합쳐서 저장
        print(file, '_2_ saved')

cv2.destroyAllWindows()