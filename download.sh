#!/bin/sh
wget -P /tmp/ http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvzf /tmp/food-101.tar.gz
mv food-101/images ./
mv food-101/meta/labels.txt ./
mv food-101/meta/train.json ./
mv food-101/meta/test.json ./
rm -r food-101
python3 shrink.py
