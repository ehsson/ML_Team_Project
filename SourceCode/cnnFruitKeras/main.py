
from keras import models
from keras.layers import *
import matplotlib.pyplot as plt
from dataset.fruit import load_fruit
import plot_show


print("start")
(x_train, t_train), (x_test, t_test) = load_fruit()
print("load end")
print(x_train.shape)
print(x_test.shape)

x_train = x_train.swapaxes(1, 3)
x_train = x_train.astype('float')
x_train = x_train / 255.


x_test = x_test.swapaxes(1, 3)
x_test = x_test.astype('float')
x_test = x_test / 255.

print(x_train.shape)
print(x_test.shape)

model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 104, 3), strides=(1, 1), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(x_train, t_train,
                    batch_size=200,
                    validation_data=(x_test, t_test),
                    epochs=10,
                    # steps_per_epoch=len(x_train),
                    # validation_steps=len(x_test)
                    )
loss, acc = model.evaluate(x_test, t_test, verbose=2)
print("\nLoss: {}, Acc: {}".format(loss, acc))

plot_show.plt_show_loss(history)
plt.show()

plot_show.plt_show_acc(history)
plt.show()