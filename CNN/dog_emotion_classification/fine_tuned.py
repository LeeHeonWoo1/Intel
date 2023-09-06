"""
강아지 표정 분류를 전이학습을 이용하여 해결합니다.
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
NUM_CLASS = 4
INPUT_SHAPE = (300, 300, 3)
BATCH_SIZE = 8
TRAIN_DIR = "./CNN/dog_emotion_classification/dog_dataset_output/train"
TEST_DIR = "./CNN/dog_emotion_classification/dog_dataset_output/test"

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255.
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = INPUT_SHAPE[:2],
    batch_size = BATCH_SIZE,
    color_mode = 'rgb',
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical'
)

def get_model(model):
    """
    전이학습 모델을 구성합니다.
    """
    params = {
        'input_shape' : INPUT_SHAPE, 
        'include_top': False,  
        'weights':'imagenet',
        'pooling':'max',
        'classes': 4
        }

    # **는 dictionary(params)내부의 키:값 쌍을 함수에 keyword 인자(input_shape = input_shape)로 전달합니다.
    pretrained_model = model(**params)
    for pretrained_layer in pretrained_model.layers[-4:]: # 사전 학습된 모델의 계층들
        pretrained_layer.trainable = True # 사전 학습된 모델들의 가중치는 동결하되, 최상단의 일부만 동결을 해제하면서 미세조정을 진행

    inputs = pretrained_model.input

    flow = tf.keras.layers.Flatten()(pretrained_model.output)
    flow = tf.keras.layers.Dense(128, activation='relu')(flow)
    flow = tf.keras.layers.Dropout(rate=0.2)(flow)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])

    return model

def fit_vgg19():
    """
    전이학습된 모델을 기반으로 학습을 진행합니다.
    """
    model_check = ModelCheckpoint(
        filepath = "./CNN/dog_emotion_classification/model/VGG19_finetuned \
        /weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor = "val_loss",
        save_best_only = True,
        mode = "min"
    )
    reduce_lr = ReduceLROnPlateau(
        monitor = "val_loss",
        factor = 0.3,
        patience = 3,
        mode = "min"
    )
    early_stop = EarlyStopping(
        monitor = "val_loss",
        patience = 5,
        mode = "min"
    )
    logger = CSVLogger(
        './CNN/dog_emotion_classification/model/VGG19_finetuned/history.csv'
        )

    model = get_model(tf.keras.applications.VGG19)
    model.fit(
        train_generator,
        epochs = EPOCHS,
        steps_per_epoch = len(train_generator),
        validation_data = validation_generator,
        validation_steps = len(validation_generator),
        callbacks=[model_check, reduce_lr, early_stop, logger]
    )

fit_vgg19()
