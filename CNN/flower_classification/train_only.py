from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import tensorflow as tf, os
import datetime

train_dir='D:/Intel/CNN/flower_classification/flower_dataset/train'
test_dir='D:/Intel/CNN/flower_classification/flower_dataset/test'

# define parameters
num_epoch=100                             # 훈련 에포크 회수
batch_size=5                              # 훈련용 이미지의 묶음
learning_rate=0.001                       # 학습률, 작을수록 학습 정확도 올라감
dropout_rate=0.3                          # 30%의 신경망 연결을 의도적으로 끊음. 과적합 방지용
input_shape=(120, 120, 3)                 # 입력데이터(이미지)의 크기, 원하는 크기를 입력하면 모든 이미지가 resize됨
num_class=3                               # 분류를 위한 정답의 갯수

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
    train_dir,                      # flow_from_directory이기에 train dataset 경로 입력. 이 때 트리 형태(내부에 클래스별로 분류해 둔 형태)로 있어야함.
    target_size = input_shape[:2],  # 하이퍼 파라미터에서 정의한 변수의 가로 세로 길이만 가져온다.
    batch_size = batch_size,        # batch_size 지정
    color_mode = 'rgb',             # 컬러 파일인 경우에는 'rgb' 또는 'rgba'로 설정해야 함
    class_mode='categorical'        # 2진 분류의 경우 binary로 설정
)                                   # Found 60000 images belonging to 10 classes.가 표시되며 이미지 증폭 완료

validation_generator = test_datagen.flow_from_directory(
    test_dir,                       # 위와 마찬가지로 트리 형태로 존재해야 함
    target_size=input_shape[:2],    
    batch_size=batch_size,          # 고해상도 사진의 경우 오류를 미연에 방지하기 위해 적은 값으로 설정한다. 4~8정도가 적당하다.
    color_mode='rgb',
    class_mode='categorical'        
)                              
     
# build DenseNet with pretrained model
def get_model(model):
    any = {
        'input_shape':(120, 120, 3), 
        'include_top':False, # Affine계층은 우리가 구성할 예정이기에 포함하지 않는다.
        'weights':'imagenet', 
        'pooling':'avg'
    }
    
    pretrained_model = model(**any)
    pretrained_model.trainable = False # 레이어를 동결 시켜서 훈련중 손실을 최소화 한다.
    
    inputs = pretrained_model.input
    
    # 완전연결신경망 구축(functional api method)
    x = tf.keras.layers.Dropout(rate=0.3)(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def fit_densenet():
    weight_path = glob("./CNN/flower_classification/model/weights/*.hdf5")
    if len(weight_path) != 0:
        for path in weight_path:
            os.remove(path)
    
    # log directory 설정
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # define callback functions
    els = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    mch = ModelCheckpoint(filepath="./CNN/flower_classification/model/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor="val_loss", save_best_only=True, mode="min")
    rdl = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, mode="min")
    logger = CSVLogger("./CNN/flower_classification/model/history.csv")
    t_board = TensorBoard(log_dir = log_dir, histogram_freq = 1)
    
    model = get_model(tf.keras.applications.DenseNet201)
    model.fit(train_generator,
        steps_per_epoch = len(train_generator),
        epochs = num_epoch,
        validation_data = validation_generator,
        validation_steps = len(validation_generator),
        callbacks=[mch, rdl, els, logger, t_board])
    
# build CNN in my own
def create_model():
    model=models.Sequential()                                                        # 모델 객체 생성  
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))  # 1차 합성곱 연산 시행
    # model.add(layers.MaxPooling2D((2,2)))                                            # (2, 2)크기로 max pooling 시행, 이 때 stride는 지정하지 않으면 pooling size로 지정된다.
    model.add(layers.Dropout(dropout_rate))                                          # dropout 층
    model.add(layers.Conv2D(32, (3,3), activation='relu'))                           # 2차 합성곱 연산 시행
    # model.add(layers.MaxPooling2D((2,2)))                                            # Max Pooling   
    model.add(layers.Dropout(dropout_rate))                                          # dropout
    # Affine 계층
    model.add(layers.Flatten())                                                      # 평탄화 작업층. 완전연결 신경망의 입력층 부분이 된다.
    model.add(layers.Dense(128, activation = 'relu'))                                 # 히든 레이어. 노드의 개수는 64개
    # model.add(layers.Dropout(dropout_rate))                                          
    model.add(layers.Dense(num_class, activation = 'softmax'))                       # 출력층의 노드 수는 3개이며, 각 데이터의 확률값으로 출력하기 위한 softmax(확률의 총합 = 1)
    
    return model # 계층을 추가한 모델을 반환한다.

def fit_model():
    model = create_model()
    model.compile(optimizer = tf.optimizers.Adam(learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    weight_path = glob("./CNN/flower_classification/model/weights/*.hdf5")
    if len(weight_path) != 0:
        for path in weight_path:
            os.remove(path)
    
    # 콜백함수(https://wikidocs.net/179491)
    # EarlyStopping : 한 epoch가 끝나면 EarlyStopping 이 호출되고 구성한 모델이 지금까지 찾은 최상의 값과 관련하여 개선되었는지 여부를 확인한다. 
    # 개선되지 않은 경우 '최상의 값 이후 개선되지 않은 횟수'의 횟수를 1만큼 증가시킨다. 실제로 개선된 경우 이 카운트를 초기화 하며 반복한다.
    
    # ModelCheckpoint : 부여하는 옵션에 따라 모델을 저장하는 콜백함수

    # ReduceLROnPlateau : 모델의 개선이 더딘 경우, learning rate를 자동으로 조절하며 모델의 개선을 유도한다.

    # 순서대로 조기종료 주시대상, 조기종료 전 모니터 할 횟수, 모드를 의미한다. 모드의 경우 monitor 지표가 감소해야 좋을 경우 min, 증가해야 좋을 경우 max를 사용한다. 
    # 모드의 경우 사용 기준은 아래에 나올 두 callback 함수 모두에도 동일하게 적용된다.
    els = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    
    # 순서대로 저장 경로, 저장 주시 대상, 가장 좋은 결과만 저장 선택 여부, 모드를 의미한다.
    mch = ModelCheckpoint(filepath="./CNN/flower_classification/model/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor="val_loss",
                        save_best_only=True, mode="min")
    # 순서대로 주시 대상, 학습 속도를 줄이는 척도, lr을 줄이기 전 모니터할 에포크 횟수, 모드를 의미한다.
    rdl = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, mode="min")
    
    # epoch횟수, 정확도, 손실값, 학습률, 검증 정확도, 검증 손실값을 각 컬럼으로 구성하여 csv파일로 생성한다.
    logger = CSVLogger("./CNN/flower_classification/model/history.csv")

    model.fit(
        train_generator, # generator를 넣음으로, batch 단위로 순회하면서 이미지를 변형
        steps_per_epoch = len(train_generator),
        epochs = num_epoch,
        validation_data = validation_generator,
        validation_steps = len(validation_generator),
        callbacks=[mch, rdl, els, logger]
    )

if __name__ == "__main__":
    fit_densenet()