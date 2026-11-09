"""Platt (sigmoid) calibration, in its own module so pickled calibrators load anywhere."""
import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    """Sigmoid calibration on the logit of the raw probability (CalibratedClassifierCV(method='sigmoid')
    with a frozen, pre-fitted model). Two parameters -> low risk of over-fitting 2.6k validation rows."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6).fit(logit(p)[:, None], y)
        self.a, self.b = float(self.lr.coef_[0, 0]), float(self.lr.intercept_[0])
        return self

    def __call__(self, p):
        return 1 / (1 + np.exp(-(self.a * logit(p) + self.b)))
