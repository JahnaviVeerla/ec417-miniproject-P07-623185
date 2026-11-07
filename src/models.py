from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier


def make_dummy_model(strategy: str = "prior") -> DummyClassifier:
    return DummyClassifier(strategy=strategy)


def make_logistic_model(random_state: int = 0) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest_model(random_state: int = 0, **kwargs) -> RandomForestClassifier:
    defaults = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": random_state,
        "n_jobs": -1,
    }
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def make_hist_gradient_boosting_model(random_state: int = 0, **kwargs) -> HistGradientBoostingClassifier:
    defaults = {
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_depth": 8,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "random_state": random_state,
        "max_iter": 500,
        "early_stopping": True,
        "n_iter_no_change": 20,
        "validation_fraction": 0.2,
        "class_weight": "balanced",
    }
    defaults.update(kwargs)
    return HistGradientBoostingClassifier(**defaults)
