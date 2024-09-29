import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 5

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data", shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)

class_names = dataset.class_names

def split_data(ds,train_split, test_split,val_split):
    length = len(ds)
    train = int(length*train_split)
    train_ds = ds.take(train)
    test = int(length*test_split)
    val = int(length*val_split)
    val_ds = ds.skip(train).take(val)
    test_ds = ds.skip(train).skip(val)
    return train_ds, test_ds, val_ds

train_ds, test_ds, val_ds =  split_data(dataset,0.8,0.1,0.1)

print(len(train_ds))
print(len(test_ds))
print(len(val_ds))

train_ds = train_ds.cache().shuffle(10000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(10000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(10000).prefetch(buffer_size = tf.data.AUTOTUNE)

rescale_resize = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomFlip(),
    layers.RandomTranslation(height_factor = 0.3, width_factor = 0.4)
])

img_shape = (256,256, 3)  # height, width, channels

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=img_shape),  # Input shape here
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax"),
])

model.summary()
model.compile(
    optimizer = "Adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ["accuracy"]
)

history =  model.fit(train_ds,epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1, validation_data = val_ds)
scores = model.evaluate(test_ds)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
