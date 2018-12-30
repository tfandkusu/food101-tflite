import tensorflow as tf
import model
import data

# 訓練データ作成担当
g = data.Data()
# GPUをすべて使わないオプション
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)
# モデルを作成
model = model.make(tflite=False)
# 最適化を定義
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",
    metrics=["categorical_accuracy"])
# コールバック
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        "各エポック終了時に重みを保存する"
        model.save("weight.hdf5")
cb = Callback()
# 途中から学習する場合
initial_epoch = 0
if initial_epoch >= 1:
    model.load_weights("weight.hdf5")
# 学習する
model.fit_generator(g.generator(),
    validation_data=g.generator_test(),
    validation_steps=g.test_steps(),
    callbacks = [cb],
    steps_per_epoch=data.TRAIN_SIZE/data.BATCH_SIZE,epochs=50,
    initial_epoch=initial_epoch)
