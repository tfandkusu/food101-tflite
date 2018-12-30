# food101-tflite
[Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)データセットによる分類をTensorFlow Lite モデル形式で作成する。

## 環境構築

Python3を使用します。

ライブラリのインストール。
```
% pip3 --user tensorflow-gpu==1.12.0 opencv-python h5py
```
TensorFlow バージョン1.12で動作確認しています。バージョン1.10ではtflite_convertコマンドに失敗します。

## 使用手順

Food-101データセットをダウンロードし、ついでに検証用の縮小画像を作ります。
```
% ./download.sh
```
学習を行います。GTX 1050 Tiで12時間ほどかかります。
```
% python3 train.py
```
予測精度を確認します。だいたい60%ぐらいになります。
```
% python3 evaluate.py
```
SavedModelを作成します。
```
% python3 keras2saved_model.py
```
TensorFlow Lite モデルファイルを作成します。
```
% ./make_tflite_model.sh
```
graph.lite というファイル名で作成されます。

## TensorFlow書き換え

[mobilenet.py](https://github.com/tfkeras/food101-tflite/blob/master/mobilenet.py)、
[normalization.py](https://github.com/tfkeras/food101-tflite/blob/master/normalization.py)、
[imagenet_utils.py](https://github.com/tfkeras/food101-tflite/blob/master/imagenet_utils.py)
は[TensorFlow](https://github.com/tensorflow/tensorflow)ソースコードからコピーして、
TensorFlow Lite モデルに変換可能なように書き換えています。
