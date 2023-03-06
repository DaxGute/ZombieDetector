import tensorflow.keras.utils as utils
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.data as data



train = utils.image_dataset_from_directory(
    'data',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (1280, 720),
    seed = 37,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'data',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (1280, 720),
    seed = 37,
    validation_split = 0.3,
    subset = 'validation',
)

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
test = test.cache().prefetch(buffer_size = data.AUTOTUNE)

class Net():
    def __init__(self, input_shape,):
        self.model = Sequential()

        self.model.add(layers.ZeroPadding2D(
            padding = ((1,0), (2,3))
        )) #1285x721

        self.model.add(layers.Conv2D(
                8, # filters
                37, # kernel size
                strides = 12, # step size
                activation = 'relu',
                input_shape = input_shape, # need for first layer

        )) #105x58x8

        self.model.add(layers.ZeroPadding2D(
            padding = ((0,0), (1,0))
        )) #106x58x8

        self.model.add(layers.MaxPool2D(
            pool_size=2,
        )) #53x29x8

        self.model.add(layers.Conv2D(
            8, # filters
            3, # kernel size
            strides = 1, # step size
            activation = 'relu'
        )) #50x26x64

        self.model.add(layers.MaxPool2D(
            pool_size=2,
        )) #25x13x64

        self.model.add(layers.Flatten(
        ))

        self.model.add(layers.Dense(
            1024,
            activation = 'relu',
        ))

        self.model.add(layers.Dense(
            256,
            activation = 'relu',
        ))

        self.model.add(layers.Dense(
            64,
            activation = 'relu',
        ))

        self.model.add(layers.Dense(
            2, #two classes
            activation = 'softmax', # always use softmax last
        ))

        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )

    def __str__(self):
        self.model.summary()
        return ""

net = Net((1280,720,3))

print(net)

net.model.fit(
    train,
    batch_size = 32,
    epochs = 40,
    validation_data = test,
    validation_batch_size = 32,
)
net.model.save('faces_model_save')