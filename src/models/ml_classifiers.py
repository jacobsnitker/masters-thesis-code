"""
8 classical ML classifiers exactly as specified in the CLARE paper (Section VI-A).

Hyperparameters:
  GB:      n_estimators=300, loss='log_loss', max_depth=3
  LGBM:    n_estimators=2000, num_leaves=100, learning_rate=0.001
  LDA:     solver='lsqr'
  LR:      max_iter=400, C=1
  MLP:     hidden_layer_sizes=(100, 10), learning_rate='adaptive', max_iter=1000
  RF:      n_estimators=1000, min_samples_split=5, max_depth=5
  SVM:     C=10
  XGBoost: n_estimators=300, learning_rate=0.001, reg_alpha=0.0001
"""

from sklearn.ensemble       import GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model   import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm            import SVC
from lightgbm               import LGBMClassifier
from xgboost                import XGBClassifier


def get_classifiers(balanced: bool = False) -> dict:
    """
    balanced=True adds class imbalance correction where natively supported
    (LR, RF, SVM, LGBM). GB, LDA, and XGBoost receive sample_weight at fit
    time instead (handled in evaluation.py). MLP has no imbalance support.
    """
    cw = "balanced" if balanced else None
    return {
        "GB": GradientBoostingClassifier(
            n_estimators=300,
            loss="log_loss",
            max_depth=3,
            random_state=42,
        ),
        "LGBM": LGBMClassifier(
            n_estimators=2000,
            num_leaves=100,
            learning_rate=0.001,
            class_weight=cw,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        ),
        "LDA": LinearDiscriminantAnalysis(
            solver="lsqr",
        ),
        "LR": LogisticRegression(
            max_iter=400,
            C=1.0,
            class_weight=cw,
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 10),
            learning_rate="adaptive",
            max_iter=1000,
            random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=1000,
            min_samples_split=5,
            max_depth=5,
            class_weight=cw,
            random_state=42,
            n_jobs=1,
        ),
        "SVM": SVC(
            C=10.0,
            class_weight=cw,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.001,
            reg_alpha=0.0001,
            n_jobs=1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }
