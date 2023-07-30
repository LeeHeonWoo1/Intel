from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import tensorflow as tf
import matplotlib as mlp, matplotlib.pyplot as plt
import pandas as pd
import splitfolders 
import os
import matplotlib.pyplot as plt, os, matplotlib as mlp
from glob import glob
from keras.preprocessing.image import ImageDataGenerator

# random seed 고정, ratio를 통해 train:test:validation의 비율을 결정. 8:1:1로 실행.
splitfolders.ratio('파일 분할 대상 경로', '파일 분할 결과 경로', seed=1337, ratio=(0.8, 0.1, 0.1))

# 폴더 이름을 라벨값과 함께 변경
t = ["train", "test", "val"]
for set_ in t:
    dir_list = os.listdir(f'파일 분할 대상 경로/{set_}/') # 
    for idx, dir in enumerate(dir_list):
        print(f"{set_}폴더의 {dir}를 {idx}_{dir}로 변경합니다.")
        os.rename(f"파일 분할 결과 경로/{set_}/{dir}", f"파일 분할 결과 경로/{set_}/{idx}_{dir}")
        
mlp.rcParams["font.family"] = "Malgun Gothic" # 한글 깨짐 방지

train_path = glob("훈련셋 파일 경로/*")
file_cnt = [len(os.listdir(path)) for path in train_path] # 파일 개수 리스트화
labels = ["angry", "happy", "relaxed", "sad"] # 라벨값

bar = plt.bar(labels, file_cnt, width=0.4, color = ["green", "blue", "pink", "purple"]) # barplot
plt.xlabel("라벨값 0~3")
plt.ylabel("파일 개수")

for rect in bar: # barplot 위에 숫자 표시하기
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 12)
    
plt.show()

# 하이퍼 파라미터 정의
num_epoch = 100             # 훈련 횟수
batch_size = 5              # 데이터 묶음 단위수
learning_rate = 0.001       # 학습률
dropout_rate = 0.2          # dropout 비율
input_shape = (150, 150, 3) # 입력 이미지 크기
num_class = 4               # 분류 클래스 수

# 이미지 증강기 정의
train_dir = '훈련셋 파일 경로'
test_dir = "테스트셋 파일 경로"

train_datagen = ImageDataGenerator(       # train dataset 용 이미지 전처리 과정
    rescale=1./255.,                      # Normalize를 위해 255로 나누어 줌
    width_shift_range=0.3,                # 폭(가로) 쪽으로 30%범위에서 랜덤하게 좌우 시프트 시킴
    zoom_range=0.2,                       # 20%범위에서 랜덤하게 크기를 늘리거나 줄임
    horizontal_flip=True                  # 수평축을 중심으로 이미지를 뒤집음
)

test_datagen = ImageDataGenerator(        # test dataset 용 이미지 전처리 과정
    rescale=1./255.                       # train의 DataGenerator와 같은 크기로 rescale해야 함
)

train_generator = train_datagen.flow_from_directory( 
    train_dir,                      # train dir
    target_size = input_shape[:2],  # 하이퍼 파라미터에서 정의한 변수의 가로 세로 길이만 가져온다.
    batch_size = batch_size,        # batch_size 지정
    color_mode = 'rgb',             # 컬러 파일인 경우, 'rgb' 또는 'rgba'로 설정
    class_mode='categorical'        # 4개의 클래스를 분류하는 문제이기에 categorical로 설정
)                                   

validation_generator = test_datagen.flow_from_directory(
    test_dir,                       # 위와 마찬가지로 트리 형태로 존재해야 함
    target_size=input_shape[:2],    
    batch_size=batch_size,          # 고해상도 사진의 경우 오류를 미연에 방지하기 위해 적은 값으로 설정한다. 4~8정도가 적당하다.
    color_mode='rgb',
    class_mode='categorical'        
) 

# 직접 빌드
def create_my_cnn():
    """
    직접 빌드하는 CNN 모델입니다. 각 파라미터를 적절히 조절하여 성능을 개선해보세요.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape = input_shape))
    model.add(layers.MaxPooling2D(2, 2)) # stride를 지정하지 않을 경우, 커널 사이즈로 자동 지정됨. = 2칸씩 건너뛰며 MaxPooling 수행
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2)) 
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten()) # 완전연결층의 입력층 부분
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_class, activation="softmax")) # 다중분류 문제이기에 softmax로 각 라벨에 대한 확률을 출력
                                                             # 이진 분류의 경우 출력층 노드 개수를 1개로 설정하고 sigmoid로 활성화 하여 중간값(0.5)을 기준으로 0, 1 클래스 분류
    return model

# 훈련
def fit_my_cnn():
    """
    직접 빌드한 모델을 컴파일하고 콜백함수와 함께 학습을 진행합니다.
    """
    model = create_my_cnn()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=['acc']) # 다중분류 문제이기에 손실함수는 categorical_crossentropy
                                                                                                                 # 이진 분류의 경우 binary_crossentropy를 사용한다.
    
    # 콜백함수 정의
    model_check = ModelCheckpoint(filepath = "가중치 파일 저장 경로/가중치 파일 이름.{epoch:02d}-{val_loss:.2f}.hdf5", monitor = "val_loss", save_best_only = True, mode = "min")
    reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.3, patience = 3, mode = "min") # 지표(val_loss) 개선이 미미하면 학습률에 factor를 곱해 학습률을 감소시킵니다.
    early_stop = EarlyStopping(monitor = "val_loss", patience = 5, mode = "min")
    logger = CSVLogger('저장할 csv파일 경로.csv')
    
    call_back_list = [model_check, reduce_lr, early_stop, logger]
    
    # 학습
    model.fit(train_generator, epochs = num_epoch, validation_data = validation_generator, callbacks = call_back_list)

def show_history(history_csv_path):              
    """
    csv파일 경로를 받아 학습 곡선을 시각화 합니다. 1행 2열로 구성되어 1열에는 학습 정확도, 검증 정확도가 출력되며 2열에는 학습 손실값, 검증 손실값에 대한 표를 시각화합니다.
    """
    mlp.rcParams["font.family"] = "Malgun Gothic" # 그래프 내부에 한글 깨짐 방지 처리
    mlp.rcParams["axes.unicode_minus"] = False    # 그래프 축 상에 마이너스 부호 깨짐 방지
    
    history = pd.read_csv(history_csv_path)       # 히스토리 csv파일을 읽어온다. dataframe으로 리턴한다.
    plt.figure(figsize=(10,5))

    # 1행 2열 중, 1열의 공간을 생성함
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history['acc'], label="train_accuracy")     # 학습 정확도에 대한 그래프
    plt.plot(history["epoch"], history['val_acc'], label="valid_accuracy") # 검증 정확도에 대한 그래프
    plt.title('모델 정확도') # 그래프의 타이틀 설정
    plt.xlabel('학습 횟수') # x축 이름 설정
    plt.ylabel("정확도") # y축 이름 설정
    plt.legend()

    # 1행 2열 중, 2열의 공간을 생성함
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history['loss'], label="train_loss_value")     # 손실값에 대한 그래프
    plt.plot(history["epoch"], history['val_loss'], label="valid_loss_value") # 검증 손실값에 대한 그래프
    plt.title('모델 손실값')
    plt.xlabel('학습 횟수')
    plt.ylabel("손실값")
    plt.legend()
    plt.show()
    
input_shape = (300, 300, 3) # 150, 224
batch_size = 16 # 4, 8

# 이미지 증강기를 갱신합니다. (다른 옵션, 함수 활용)
train_datagen = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.densenet.preprocess_input, # 0 ~ 1사이 값으로 조정하되, 각 채널은 ImageNet 데이터 세트에 대해 정규화
    width_shift_range=0.3,               
    zoom_range=0.2,                      
    horizontal_flip=True,
    fill_mode = 'nearest',
    shear_range = 0.2               
)

test_datagen = ImageDataGenerator(      
    preprocessing_function = tf.keras.applications.densenet.preprocess_input
)

train_generator = train_datagen.flow_from_directory( 
    train_dir,                 
    target_size = input_shape[:2],  
    batch_size = batch_size,      
    color_mode = 'rgb',           
    class_mode='categorical'        
)                                   

validation_generator = test_datagen.flow_from_directory(
    test_dir,                      
    target_size=input_shape[:2],    
    batch_size=batch_size,          
    color_mode='rgb',
    class_mode='categorical'        
)

# 전이학습 (VGG19 모델 활용)
def get_model(model):
    """
    이미지 분류에 적합하다고 알려진 VGG19 모델로 전이학습을 진행합니다.
    """
    params = {'input_shape' : input_shape, 
              'include_top': False,  # Affine 계층을 포함시키지 않고 직접 구현하여 연결한다.
              'weights':'imagenet', # imagenet의 가중치 사용
              'pooling':'max',
              'classes':4}
    
    pretrained_model = model(**params) # **는 dictionary(params)내부의 키:값 쌍을 함수에 keyword 인자(input_shape = input_shape)로 전달합니다.
    
    for pretrained_layer in pretrained_model.layers[:23]: # 본래 미세 조정의 의미는 Affine 계층의 하이퍼 파라미터를 재학습함과 더불어 사전 학습 모델의 최상단의 일부를 재학습 하는 것을 의미하나
        pretrained_layer.trainable = False # 딱히 의미가 없어 그냥 전부 동결시켰습니다.
    
    inputs = pretrained_model.input
    
    # 완전연결신경망 구축(functional api method)
    x = tf.keras.layers.Flatten()(pretrained_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    
    return model

def fit_vgg19():
    """
    전이 학습된 모델을 컴파일하고 훈련합니다.
    """
    model = get_model(tf.keras.applications.VGG19)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
    model_check = ModelCheckpoint(filepath = "가중치 저장 경로/가중치 파일 이름.{epoch:02d}-{val_loss:.2f}.hdf5", monitor = "val_loss", save_best_only = True, mode = "min")
    reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.3, patience = 3, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", patience = 5, mode = "min")
    logger = CSVLogger('csv파일 저장 경로.csv')
    
    model.fit(train_generator, epochs = num_epoch, steps_per_epoch = len(train_generator),
              validation_data = validation_generator, validation_steps = len(validation_generator),
              callbacks=[model_check, reduce_lr, early_stop, logger])

# 직접 빌드한 CNN모델 학습, 학습 곡선 시각화
# fit_my_cnn()
# show_history('./model/my_cnn_history.csv')

# 전이학습된 모델 학습, 학습 곡선 시각화
# fit_vgg19()
# show_history('./model/VGG19/VGG19_history.csv')