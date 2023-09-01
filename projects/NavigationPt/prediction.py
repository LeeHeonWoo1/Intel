import cv2 
from imutils.video import WebcamVideoStream
from keras import models, layers, optimizers
import winsound
import tensorflow as tf
import numpy as np
import playsound
from keras.models import load_model

def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image

def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation = "relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation = "relu"))
    model.add(layers.Dense(3, activation = "softmax"))

    optim = optimizers.Adam(0.001)
    model.compile(optimizer = optim, loss = "categorical_crossentropy", metrics = "acc")
    return model

model = get_model()
model.load_weights(r"D:\Intel\projects\NavigationPt\models\weights\add_lego_data_11-0.05.hdf5")

host = "{}:4747/video".format("http://192.168.0.39")
cam = WebcamVideoStream(src=host).start() 

test_path = r"D:\Intel\projects\NavigationPt\test_image"
input_shape = (50, 50, 3)
class_list = {0 : 'lego', 1 : 'airpods', 2 : 'charger'}
while True:               # q키 입력으로 영상 종료
    frame = cam.read()    # 웹캠 영상을 읽어와 실시간으로 뿌림. ret, frame = capture.read() 에 해당  
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    else:
        save = cv2.imwrite(test_path + "/temp.jpg", frame, params= None)
        img = read_image(test_path + "/temp.jpg")
        img = tf.image.resize(img, input_shape[:2])

        image = np.array(img)
        
        testImage = image[tf.newaxis, ...]
        pred = model.predict(testImage, verbose = 0)
        num = np.argmax(pred)
        result = class_list[num]
        
        img = cv2.putText(frame, result, (350, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("RESULTS", img)
        if num == 2:
            playsound.playsound(r"D:\Intel\projects\NavigationPt\mp3_files\warning.mp3")
        elif num == 0 or num == 1:
            winsound.Beep(200, 700)

cv2.destroyAllWindows()