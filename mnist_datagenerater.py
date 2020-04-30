from keras.datasets import mnist
from keras.utils import np_utils
import numpy
import struct
import math

# 画像のパラメータ
train_filenum = 60000
test_filenum = 1
img_rows, img_cols = 28, 28
pixelnum = img_rows*img_cols
bytesize = 1
nb_classes = 10

# 配列の確保
X_train_binary = [0 for j in range(pixelnum*train_filenum)]
X_train_int = [0 for j in range(pixelnum*train_filenum)]
X_test_binary = [0 for j in range(pixelnum*test_filenum)]
X_test_int = [0 for j in range(pixelnum*test_filenum)]
y_train_int = [0 for j in range(train_filenum)]
y_test_int = [0 for j in range(test_filenum)]

# ダウンロードしたトレーニング画像を開く
f1 = open("cnn/train-images-idx3-ubyte", "rb")
f2 = open("cnn/train-labels-idx1-ubyte", "rb")

# 自分で手書き作製したテスト画像（今回は数字の8）を開く
f3 = open("cnn/t10k-images-idx3-ubyte", "rb")
f4 = open("cnn/t10k-labels-idx1-ubyte", "rb")

# 画像データ読み込み
f1.seek(16)
X_train_binary = f1.read(bytesize*img_rows*img_cols*train_filenum)
X_test_binary = f3.read(bytesize*img_rows*img_cols*test_filenum)
f1.close()
f3.close()

# ラベルデータ読み込み
f2.seek(8)
y_train_binary = f2.read(bytesize*train_filenum)
y_test_binary = f4.read(bytesize*test_filenum)
f2.close()
f4.close()

print(img_rows*img_cols*train_filenum,len(X_train_int),len(X_train_binary))
# バイナリを整数(文字列)に変換
for a in range(img_rows*img_cols*train_filenum):
    X_train_int[a] = struct.unpack("h", X_train_binary[a].to_bytes(2,'little'))

for a in range(img_rows*img_cols*test_filenum):
    X_test_int[a] = struct.unpack("h", X_test_binary[a].to_bytes(2,'little'))

for a in range(train_filenum):
    y_train_int[a] = struct.unpack("h", y_train_binary[a].to_bytes(2,'little'))

for a in range(test_filenum):
    y_test_int[a] = struct.unpack("h", y_test_binary[a].to_bytes(2,'little'))

# numpy配列に格納
X_train = numpy.array(X_train_int)
X_test  = numpy.array(X_test_int)
y_train = numpy.array(y_train_int)
y_test  = numpy.array(y_test_int)

# 画像を一次元配列に
X_train = X_train.reshape(train_filenum, 1, img_rows, img_cols)
X_test = X_test.reshape(test_filenum, 1, img_rows, img_cols)

# 画像を0.0~1.0の範囲に変換
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# クラスラベル y （数字の0~9）を、one-hotエンコーディング型式に変換
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
