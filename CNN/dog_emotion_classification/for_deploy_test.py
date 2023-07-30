import numpy as np
import tensorflow as tf
from for_deploy_train import get_model
from glob import glob
import matplotlib.pyplot as plt, matplotlib as mlp

mlp.rcParams["font.family"] = "Malgun Gothic"

def read_image(path):                                   
    gfile = tf.io.read_file(path)                       
    image = tf.io.decode_image(gfile, dtype=tf.float32) 
    return image

model = get_model(tf.keras.applications.VGG19)
    
# 컴파일 한 뒤
model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])

# 학습했던 가중치를 적용한다.
model.load_weights('가중치 파일 경로.hdf5')

# 라벨값을 dict로 생성하여 결과 표출 시 사용한다.
labels = {0 : "화가 난 것", 1 : "행복한 것", 2 : "평온한 것", 3 : "슬픈 것"}

# 검증파일 경로 지정
data_path = glob('validation 데이터 경로/*/*.jpg')
np.random.shuffle(data_path)
for test_no in range(3):                    
    path = data_path[test_no]
    
    img = read_image(path)
    img = tf.image.resize(img, (300, 300))  # input size의 가로 세로의 크기와 동일하게 맞춘다.

    image = np.array(img)                   # inshow를 사용하기 위해서 ndarray로 전환
    plt.imshow(image, 'gray')
    plt.title('**씨 지금 저희 강아지 어때요?')
    plt.show()

    image = image[:, :, :]                     # 원본 이미지의 크기를 모두 가져오고(300, 300, 3)
    test_image = image[tf.newaxis, ...]        # 0번째 인덱스 값에 차원을 추가하며 (1, 300, 300, 3)의 shape으로 형전환
    pred = model.predict(test_image)           # predict함수를 사용하여 테스트 이미지 값을 유추
    num = np.argmax(pred)                      # 확률이 가장 높은 인덱스를 가져와서
    print(f"아 지금은 {labels[num]} 같네요 !") # dict의 키값으로 사용(라벨 표출)