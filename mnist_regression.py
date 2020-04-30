import keras
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# mnistのデータの取得
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 二次元配列から一次元に変換
x_train = np.array(x_train).reshape(len(x_train), 784)
x_test = np.array(x_test).reshape(len(x_test), 784)
x_train = np.array(x_train).astype("float32")
x_test = np.array(x_test).astype("float32")

# 0〜1に正規化
x_train /= 255
x_test /= 255
y_train = np.array(y_train)
y_test = np.array(y_test)

# モデルの構築
model = Sequential()
model.add(Dense(256, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))

# 誤差関数は平均二乗誤差、最適化手法はrmsprop
model.compile(loss="mean_squared_error", optimizer="rmsprop")

# 学習
history = model.fit(x_train, y_train,
                    batch_size=32, epochs=100,
                    verbose=1, validation_split=0.2)

# 20個程度のテストデータを使って予測結果の確認
print("正解：予測")
for x, y in zip(x_test[0:20], y_test[0:20]):
    predicted_y = model.predict(np.array([x]))[0][0]
    print("{}:{}".format(y,predicted_y))
