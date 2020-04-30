from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 各層のパラメータ
nb_filters = 10           # 畳み込みフィルタ数
nb_conv = 3              # 畳み込みフィルタの縦横pixel数
nb_pool = 2               # プーリングを行う範囲の縦横pixel数　　
nb_classes = 10        # 分類するクラス数
nb_epoch = 50          # 最適化計算のループ回数

# 特徴量抽出
model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))   # 畳み込みフィルタ層
model.add(Activation("relu"))     # 最適化関数
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))  # プーリング層
model.add(Dropout(0.2))     # ドロップアウト層

# 特徴量に基づいた分類
model.add(Flatten())     # 全結合層入力のためのデータの一次元化（図1では省略している）
model.add(Dense(128))     # 全結合層
model.add(Activation("relu"))     # 最適化関数
model.add(Dropout(0.2))     # ドロップアウト層
model.add(Dense(nb_classes))     # 出力層（全結合層：ノードの数は分類クラス数）
model.add(Activation("softmax"))     # 出力層

#モデルのコンパイル
model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

# モデルの学習
early_stopping = EarlyStopping(patience=2, verbose=1)

model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=128, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# モデルの評価
classes = model.predict(X_test, batch_size=128, verbose=True)
print(classes[0])
