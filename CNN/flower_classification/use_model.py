from glob import glob
from train_only import create_model
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt, matplotlib as mlp, pandas as pd

def read_image(path):                                    # 이미지를 읽어오기 위한 함수 만듦
    gfile = tf.io.read_file(path)                        # 경로상의 하나의 이미지를 읽어 들여 gfile변수에 보관
    image = tf.io.decode_image(gfile, dtype=tf.float32)  # 읽어들인 이미지를 디코딩하여 이미지 배열로 만듦
    return image

# 학습 추이 그래프
def show_history(history_csv_path):               # history csv 경로를 입력받고
    mlp.rcParams["font.family"] = "Malgun Gothic" # 그래프 내부에 한글 깨짐 방지 처리
    mlp.rcParams["axes.unicode_minus"] = False # 그래프 축 상에 마이너스 부호 깨짐 방지
    
    history = pd.read_csv(history_csv_path)
    plt.figure(figsize=(10,5))

    # 1행 2열 중, 1열의 공간을 생성함
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history['accuracy'], label="train_accuracy") # 정확도에 대한 추이 그래프
    plt.plot(history["epoch"], history['val_accuracy'], label="valid_accuracy") # 검증 정확도에 대한 추이 그래프
    plt.title('모델 정확도') # 그래프의 타이틀 설정
    plt.xlabel('학습 횟수') # x축 이름 설정
    plt.ylabel("정확도") # y축 이름 설정
    plt.legend()

    # 1행 2열 중, 2열의 공간을 생성함
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history['loss'], label="train_loss_value") # 손실값에 대한 추이 그래프
    plt.plot(history["epoch"], history['val_loss'], label="valid_loss_value") # 검증 손실값에 대한 추이 그래프
    plt.title('모델 손실값')
    plt.xlabel('학습 횟수')
    plt.ylabel("손실값")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 모델의 구조를 형성하고
    model = create_model()
    
    # 컴파일 한 뒤
    model.compile(optimizer = tf.optimizers.Adam(0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # 학습했던 가중치를 적용한다.
    model.load_weights("./CNN/flower_classification/model/weights/weights.25-0.44.hdf5")
    
    # 학습 추이 시각화
    show_history("./CNN/flower_classification/model/history.csv")

    # 라벨값을 dict로 생성하여 결과 표출 시 사용한다.
    labels = {0:"cosmos", 1:"sunflower", 2:"tulip"}

    # 검증파일 경로 지정
    data_path = glob('./CNN/flower_classification/flower_dataset/val/*/*.jpg')
    np.random.shuffle(data_path)

    for test_no in range(3):                    # shuffle된 data_path의 0, 1, 2인덱스에 해당하는 이미지들만 확인한다.
        path = data_path[test_no]
        
        img = read_image(path)
        img = tf.image.resize(img, (120, 120))  # input size의 가로 세로의 크기와 동일하게 맞춘다.

        image = np.array(img)                   # inshow를 사용하기 위해서 ndarray로 전환
        plt.imshow(image, 'gray')
        plt.title('Check the Image and Predict Together!')
        plt.show()

        image = image[:, :, :]                  # 원본 이미지의 크기를 모두 가져오고(120, 120, 3)
        test_image = image[tf.newaxis, ...]     # 0번째 인덱스 값에 차원을 추가하며 (1, 120, 120, 3)의 shape으로 형전환
        pred = model.predict(test_image)        # predict함수를 사용하여 테스트 이미지 값을 유추
        num = np.argmax(pred)                   # 확률이 가장 높은 인덱스를 가져와서
        print("예측값: {}".format(labels[num])) # dict의 키값으로 사용(라벨 표출)