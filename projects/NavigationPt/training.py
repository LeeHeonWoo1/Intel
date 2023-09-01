from keras import models, layers, callbacks, optimizers
from keras.preprocessing.image import ImageDataGenerator

train_data_path = r"D:\Intel\projects\NavigationPt\image"

train_generator = ImageDataGenerator(
    rescale = 1./255.,
    validation_split = 0.2,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1
)

train_data = train_generator.flow_from_directory(
    train_data_path,
    target_size = (50, 50),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 4,
    subset = "training"
)

valid_data = train_generator.flow_from_directory(
    train_data_path,
    target_size = (50, 50),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 4,
    subset = "validation"
)

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
model.fit(train_data, 
          steps_per_epoch = len(train_data),
          epochs = 100,
          callbacks = [callbacks.ModelCheckpoint("./projects/NavigationPt/models/weights/val_spilit_0.2_{epoch:02d}-{val_loss:.2f}.hdf5",
                                                 monitor = "val_loss",
                                                 save_best_only = True,
                                                 mode = "min"),
                       callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience = 7)],
          validation_data = valid_data,
          validation_steps = len(valid_data)
        )