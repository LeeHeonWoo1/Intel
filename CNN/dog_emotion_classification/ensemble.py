from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import tensorflow as tf

num_epoch = 100             # 훈련 횟수
learning_rate = 0.001       # 학습률
dropout_rate = 0.2          # dropout 비율
num_class = 4 
batch_size = 8 # 4, 8
input_shape = (300, 300, 3) # 150, 224

train_dir = "./CNN/dog_emotion_classification/dog_dataset_output/train"
test_dir = "./CNN/dog_emotion_classification/dog_dataset_output/test"

# 생성기 수정 시도
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

train_generator_vgg = train_datagen.flow_from_directory( 
    train_dir,                 
    target_size = input_shape[:2],  
    batch_size = batch_size,      
    color_mode = 'rgb',           
    class_mode='categorical'        
)          
                         
train_generator_densenet = train_datagen.flow_from_directory( 
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

# VGG19 -> DenseNet201 -> Affine
def get_model(model1, model2):
    
    params = {'input_shape' : input_shape, 
              'include_top': False,  # Affine 계층을 포함시키지 않고 직접 구현하여 연결한다.
              'weights':'imagenet', # imagenet의 가중치 사용
              'pooling':'max',
              'classes':4}
    
    vgg_19 = model1(**params) # **는 dictionary(params)내부의 키:값 쌍을 함수에 keyword 인자(input_shape = input_shape)로 전달합니다.
    for pretrained_layer in vgg_19.layers[-3:]: # 사전 학습된 모델의 계층들 
        pretrained_layer.trainable = True # 사전 학습된 모델들의 가중치는 동결하되, 최상단의 일부만 동결을 해제하면서 미세조정을 진행
        
    densenet_201 = model2(**params)
    for pretrained_layer in densenet_201.layers[-3:]: 
        pretrained_layer.trainable = True 
    
    vgg_output = vgg_19.output               # vgg의 output은 densenet의 input이 되고
    densenet_output = densenet_201.output    # densenet의 input은 affine계층의 input이 된다.
    merged_output = tf.keras.layers.concatenate([vgg_output, densenet_output]) # 두 모델을 결합하고
    # 완전연결신경망 구축(functional api method)
    x = tf.keras.layers.Flatten()(merged_output) # 결합된 모델을 Affine계층의 입력층으로 전달한다.
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    ensembeled_model = tf.keras.Model(inputs=[vgg_19.input, densenet_201.input], outputs = outputs)

    ensembeled_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics=['acc'])
    
    return ensembeled_model

def get_dataset(generator):
    dataset = tf.data.Dataset.from_tensor_slices(generator)
    return dataset

train_dataset_vgg = get_dataset(train_generator_vgg)
train_dataset_densenet = get_dataset(train_generator_densenet)
print(train_generator_densenet)
print(train_dataset_vgg)

def ensemble_preferation():
    model_check = ModelCheckpoint(filepath="./CNN/dog_emotion_classification/model/ensembled/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                  monitor="val_loss", save_best_only=True, mode="min")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    logger = CSVLogger('./CNN/dog_emotion_classification/model/ensembled/VGG19_history.csv')

    model = get_model(tf.keras.applications.VGG19, tf.keras.applications.DenseNet201)
    
    inputs = {"vgg_input": train_dataset_vgg, "densenet_input": train_dataset_densenet}
    
    model.fit(inputs, epochs=num_epoch,
              validation_data=validation_generator,
              callbacks=[model_check, reduce_lr, early_stop, logger])

# ensemble_preferation()
