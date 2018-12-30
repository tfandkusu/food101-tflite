import tensorflow as tf
import model
import shutil

# Kerasのモデルを読み込む
model = model.make(tflite=True)
# ニューラルの重みを読み込む
model.load_weights("weight.hdf5")
# TensorFlowのセッションを取得
sess = tf.keras.backend.get_session()
# SavedModelを出力
shutil.rmtree("saved_model/",True)
tf.saved_model.simple_save(sess,"saved_model/",
    inputs={'input': model.inputs[0]},
    outputs={'output': model.outputs[0]})
