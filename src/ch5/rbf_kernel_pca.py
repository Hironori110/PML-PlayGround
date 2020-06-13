from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
    """
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
    gamma: float
        Tuning parameter of the RBF kernel
    n_components: int
        返される主成分の数

    Returns
    ------------
    X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
        射影済みデータセット
    lambdas: list
        固有値
    """
    # ペアごとの平方ユークリッド距離、データセット内でペアごとに計算
    sq_dists = pdist(X, 'sqeuclidean')
    # ペアごとの距離を正方行列に
    mat_sq_dists = squareform(sq_dists)
    # 対象カーネル行列の計算
    K = exp(-gamma * mat_sq_dists)

    # カーネル行列の中心化
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # 中心化されたカーネル行列から固有対を取得
    eigvals, eigvecs = eigh(K)
    # 大きい順にソート
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # 上位 k 個の固有ベクトルと固有値を収集
    X_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))
    lambdas = [eigvals[i] for i in range(n_components)]

    return X_pc, lambdas
