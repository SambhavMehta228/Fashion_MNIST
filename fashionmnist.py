import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
print(x_train_full.shape)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] #Name of Output CLasses
x_valid,x_train=x_train_full[:5000]/255.0,x_train_full[5000:]/255.0 #Creating Validating Set
y_valid,y_train=y_train_full[:5000],y_train_full[5000:]
sft_model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(300, activation="relu"),
keras.layers.Dense(100, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])
print(sft_model.summary())
sft_model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
history=sft_model.fit(x_train,y_train,epochs=50,validation_data=(x_valid,y_valid))  #the model is trained.
#making predictions
x_new=x_test[:3]
y_probability=sft_model.predict(x_new)
print(y_probability.round(2))
print(class_names[y_probability])
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()





