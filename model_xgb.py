import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class ModelXGB():

    def __init__(self, fold_num, params):
        """
        コンストラクタ
        run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        params: ハイパーパラメータ
        """
        self.fold_num = fold_num
        self.params = params
        self.model = None
        self.model_name = "XGB"

    def train(self, tr_x, tr_y, val_x=None, val_y=None):
        """
        モデルの学習を行い、学習済のモデルを保存する
        ty_x, tr_y : trainデータの特徴量と目的変数
        va_x, va_y : validationデータの特徴量と目的変数
        """
        is_valid = val_x is not None # valdationデータが与えられているか
        # xgboost用のデータセットの作成
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if is_valid:
            dvalid = xgb.DMatrix(val_x, label=val_y)
        
        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop("num_round")

        # 学習
        if is_valid:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            watchlist = [(dtrain, "train"), (dvalid, "valid")]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
            early_stopping_rounds=early_stopping_rounds)
        else:
            watchlist = [(dtrain, "train")]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist)
    
    def predict(self, test_x):
        """
        テストセットの予測
        """
        dtest = xgb.DMatrix(test_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

    def save_model(self, model):
        """
        モデルを保存する " K-Foldを考えてfol_numごとの名前をつける
        """
        model_path = os.path.join("./model/", f"{self.fold_num}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(model, open(model_path, "wb"))
    
    def load_model(self):
        model_path = os.path.join("./model/", f"{self.model_name}_{self.fold_num}.model")
        self.model = pickle.load(open(model_path, "rb"))

class modelRandomForest:
    def __init__(self, fold_num, params):
        """
        コンストラクタ
        run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        params: ハイパーパラメータ
        """
        self.fold_num = fold_num
        self.params = params
        self.model = None
        self.model_name = "Random_Forest"

    def train(self, tr_x, tr_y, is_gird=False):
        """
        モデルの学習を行い、学習済のモデルを保存する
        ty_x, tr_y : trainデータの特徴量と目的変数
        va_x, va_y : validationデータの特徴量と目的変数
        """
        # params = {
        #     "max_depth" : [2,3,4,5,6,7],
        #     "min_sample_leaf" : [1,2,3,4,5,6,7,8,9,10],
        #     "min_sample_split" : [2,3,4,5]
        # }
        # モデルの定義
        forest = RandomForestClassifier(n_estimators=10, random_state=0)
        # グリッドサーチ
        grid_search = GridSearchCV(self.model, self.params, iid=True, cv=10, return_train_score=True)

        self.model = grid_search.best_estimator_     
    
    def predict(self, test_x):
        """
        テストセットの予測
        """
        return self.model.predcit(test_x)

    def save_model(self, model):
        """
        モデルを保存する " K-Foldを考えてfol_numごとの名前をつける
        """
        model_path = os.path.join("./model/", f"{self.fold_num}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(model, open(model_path, "wb"))
    
    def load_model(self):
        model_path = os.path.join("./model/", f"{self.model_name}_{self.fold_num}.model")
        self.model = pickle.load(open(model_path, "rb"))
