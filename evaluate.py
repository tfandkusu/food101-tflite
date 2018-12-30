import tensorflow as tf
import data
import model

g = data.Data()
# GPUをすべて使わない
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# モデルを復元
model = model.make(tflite = False)
adam = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam,loss="categorical_crossentropy",
    metrics=["categorical_accuracy"])
model.load_weights("weight.hdf5")
# テストデータの検証を行う
results = model.evaluate_generator(g.generator_test(),steps=g.test_steps(),verbose=1)
print("Accuracy %.2f %%" % (results[1]*100))
