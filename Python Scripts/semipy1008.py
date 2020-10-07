# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
iris_dataset = load_iris()
#print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))#データセットの要素書き出し
#print(iris_dataset['DESCR'][:193] + "\n...")#データセットの説明をする
#print("Target names: {}".format(iris_dataset['target_names']))#分類する先,アイリスの種類を表示
#print("Feature name: \n{}".format(iris_dataset['feature_names']))#特徴量の名前
#print("Type of data: {}".format(type(iris_dataset['data'])))#dataのデータ形式を表示する
#print("Shape of data: {}".format(iris_dataset['data'].shape))#dataのサイズ確認
#print("Target:\n{}".format(iris_dataset['target']))#0,1,2で花の種類を表現0はsetosa,1はversicolor,2はvirginica
