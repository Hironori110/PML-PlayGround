from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS():
    """
    逐次後退選択(Sequential Backward Selection)
    """

    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.2, random_state=42):
        self.scoring = scoring  # 特徴量を評価する指標
        self.estimator = clone(estimator)  # 推定機
        self.k_features = k_features  # 選択する特徴量の和
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        # 全特徴量の個数、列インデックス
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        # すべての特徴量によるスコア
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        # 指定した特徴量の数になるまで反復
        while dim > self.k_features:
            scores = []
            subsets = []

            # combinations: tuple 要素(n)に対して、r 個選択した組み合わせを返す
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            # 最良のスコアの列インデックスを格納
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 指定された列インデックスの特徴量を抽出して、モデルに適合
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)

        return score
