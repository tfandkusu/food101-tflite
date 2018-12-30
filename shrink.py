import os
import cv2

for root, dirs, files  in os.walk("images/"):
    for fn in files:
        path = root + "/" + fn
        if path.endswith(".jpg"):
            # 出力パス
            outpath = path.replace("images","shrink")
            # ディレクトリ作成
            dn = os.path.dirname(outpath)
            os.makedirs(dn, exist_ok=True)
            # 画像を読み込んで縮小する
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img,(224,224))
            cv2.imwrite(outpath,img,[int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(outpath)
