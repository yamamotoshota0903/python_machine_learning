# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
import pandas as pd
import mglearn
iris_dataset = load_iris()
#print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))#データセットの要素書き出し
#print(iris_dataset['DESCR'][:193] + "\n...")#データセットの説明をする
#print("Target names: {}".format(iris_dataset['target_names']))#分類する先,アイリスの種類を表示
#print("Feature name: \n{}".format(iris_dataset['feature_names']))#特徴量の名前
#print("Type of data: {}".format(type(iris_dataset['data'])))#dataのデータ形式を表示する
#print("Shape of data: {}".format(iris_dataset['data'].shape))#dataのサイズ確認
#print("Target:\n{}".format(iris_dataset['target']))#0,1,2で花の種類を表現0はsetosa,1はversicolor,2はvirginica
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
#Xは入力, yは出力, trainは訓練用データ, testはテスト用データ, train_test_splitを使ってそれぞれにデータセットを切り分けた
#以下、データの散布図化
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8, cmap=mglearn.cm3)
#以下、k-最近傍法で機械学習モデルを構築
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
#以下、なんかoutに書いてあったけどよくわからんやつ
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
