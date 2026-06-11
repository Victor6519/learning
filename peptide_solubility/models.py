"""
models.py — 机器学习模型: 训练、保存、加载、预测
"""

import os
import pickle
import numpy as np

MODEL_DIR = os.path.dirname(__file__)


class PeptideSolubilityModel:
    """
    多肽溶解度 ML 模型包装。
    默认使用 RandomForestRegressor (类似小分子 ESOL 项目)。
    """

    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler      # 标准化器
        self.feature_names = []
        self.trained = self.model is not None

    # ---- 训练 ----
    def train(self, X, y, feature_names=None, model_type='rf'):
        """
        训练模型。
        X: (n_samples, n_features)
        y: (n_samples,)  logS 标签
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 模型
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=3,
                random_state=42, n_jobs=-1
            )
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42
                )
            except ImportError:
                raise ImportError("XGBoost 未安装, 请用 pip install xgboost")

        self.model.fit(X_scaled, y)
        self.trained = True

        # CV 评估
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5,
                                     scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()

        train_pred = self.model.predict(X_scaled)
        train_mae = np.mean(np.abs(train_pred - y))

        return {'train_mae': train_mae, 'cv_mae': cv_mae, 'n_samples': len(y)}

    # ---- 预测 ----
    def predict(self, X):
        """返回 logS 预测值"""
        if not self.trained:
            raise RuntimeError("模型未训练或未加载")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    # ---- 持久化 ----
    def save(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, 'peptide_solubility_model.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'trained': self.trained,
            }, f)
        return path

    def load(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, 'peptide_solubility_model.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data.get('feature_names', [])
        self.trained = data.get('trained', True)
        return self


# ================================================================
#  全局模型实例 (懒加载)
# ================================================================

_model = None


def get_model():
    """获取或创建模型实例 (自动加载已训练模型)"""
    global _model
    if _model is None:
        _model = PeptideSolubilityModel()
        default_path = os.path.join(MODEL_DIR, 'peptide_solubility_model.pkl')
        if os.path.exists(default_path):
            _model.load(default_path)
    return _model
