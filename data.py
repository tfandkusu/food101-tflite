import json
import numpy as np
import cv2
import random
import tensorflow as tf
BATCH_SIZE = 50
TRAIN_SIZE = 101000
TEST_BATCH_SIZE = 10
CLASSES = 101

class Data:
    def __init__(self):
        #ラベル一覧
        self.labels = []
        # 訓練JSONを読み込み
        self.train = []
        n = 0
        with open("train.json") as f:
            jd = json.loads(f.read())
            for label in sorted(jd.keys()):
                self.labels.append(label)
                self.train.append(jd[label])
                n += 1
                if n >= CLASSES:
                    break
        # ラベルJSONを読み込み
        self.test = []
        with open("test.json") as f:
            jd = json.loads(f.read())
            for label in self.labels:
                self.test.append(jd[label])

    def generator(self):
        "訓練データのジェネレータ"
        while True:
            xs,ys = self.make_batch()
            yield(xs,ys)

    def generator_test(self):
        "テストデータのジェネレータ"
        index = 0
        xs = []
        ys = []
        images = self.test
        step = 0
        while True:
            for label in range(0,len(self.labels)):
                for i in range(0,len(images[label])):
                    path = "shrink/" + images[label][i] + ".jpg"
                    img = cv2.imread(path,cv2.IMREAD_COLOR)
                    x = img.astype(np.float32)
                    x /= 255
                    # yを作成
                    y = tf.keras.utils.to_categorical(label,len(self.labels))
                    # x,yを作成
                    xs.append(x)
                    ys.append(y)
                    index += 1
                    if index % TEST_BATCH_SIZE == 0 and index > 0:
                        yield(np.array(xs),np.array(ys))
                        step += 1
                        xs = []
                        ys = []

    def test_steps(self):
        steps =  len(self.labels)*250 // TEST_BATCH_SIZE
        return steps

    def make_batch(self):
        "訓練データのバッチを作成する"
        xs = []
        ys = []
        for i in range(BATCH_SIZE):
            x,y = self.make_xy()
            xs.append(x)
            ys.append(y)
        return np.array(xs),np.array(ys)

    def make_xy(self):
        "訓練データの入力x,出力yのペアを作成する"
        # どの料理にするかランダムに選ぶ
        label = random.randint(0,len(self.labels) - 1)
        # その中から画像をランダムに選ぶ
        images = self.train
        index = random.randint(0,len(images[label]) - 1)
        path = "images/" + images[label][index] + ".jpg"
        # 入力xを作成
        # 画像を読み込んで
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        # ランダムに反転、クロップを行い画像かさ増しする
        img = self.augment(img)
        # 学習用の大きさに縮小する
        img = cv2.resize(img,(224,224))
        # Float配列データとする
        x = img.astype(np.float32)
        x /= 255
        # yを作成
        y = tf.keras.utils.to_categorical(label,len(self.labels))
        return x,y

    def augment(self,img):
        # ランダムに反転
        if random.randint(0,1) == 0:
            img = cv2.flip(img,0)
        # ランダムにクロップ
        width = img.shape[1]
        height = img.shape[0]
        left = random.randint(0,width//4 - 1)
        top = random.randint(0,height//4 - 1)
        right = random.randint(3*width//4,width)
        bottom = random.randint(3*height//4,height)
        img = img[top:bottom,left:right]
        return img
