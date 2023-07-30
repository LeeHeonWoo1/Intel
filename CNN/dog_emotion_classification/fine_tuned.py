from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import tensorflow as tf

num_epoch = 100           
learning_rate = 0.001     
dropout_rate = 0.2       
num_class = 4 
input_shape = (300, 300, 3)
batch_size = 8 

train_dir = "./CNN/dog_emotion_classification/dog_dataset_output/train"
test_dir = "./CNN/dog_emotion_classification/dog_dataset_output/test"

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

def get_model(model):
    
    params = {'input_shape' : input_shape, 
              'include_top': False,  
              'weights':'imagenet',
              'pooling':'max',
              'classes':4}
    
    pretrained_model = model(**params) # **는 dictionary(params)내부의 키:값 쌍을 함수에 keyword 인자(input_shape = input_shape)로 전달합니다.
    for pretrained_layer in pretrained_model.layers[-4:]: # 사전 학습된 모델의 계층들 
        pretrained_layer.trainable = True # 사전 학습된 모델들의 가중치는 동결하되, 최상단의 일부만 동결을 해제하면서 미세조정을 진행
    
    inputs = pretrained_model.input
    
    # 완전연결신경망 구축(functional api method)
    x = tf.keras.layers.Flatten()(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
    
    return model

def fit_vgg19():
    model_check = ModelCheckpoint(filepath = "./CNN/dog_emotion_classification/model/VGG19_finetuned/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                  monitor = "val_loss", save_best_only = True, mode = "min")
    reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.3, patience = 3, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", patience = 5, mode = "min")
    logger = CSVLogger('./CNN/dog_emotion_classification/model/VGG19_finetuned/history.csv')
    
    model = get_model(tf.keras.applications.VGG19)
    model.fit(train_generator, epochs = num_epoch, steps_per_epoch = len(train_generator),
              validation_data = validation_generator, validation_steps = len(validation_generator),
              callbacks=[model_check, reduce_lr, early_stop, logger])

fit_vgg19()