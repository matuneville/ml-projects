import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, classifiers, vote='classlabel', weights=None):
        """
        Parameters
        :param classifiers: List of classifiers 
        :param vote: Get classlabel or probability of majority vote
        :param weights: List of weights in case needed 
        """
        self.classifiers = classifiers
        self.named_classifiers_ = {
            key: value for key, value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights
        
        
    def fit(self, X, y):
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote must be 'probability' or 'classlabel'" 
                             f"; got (vote={self.vote})")
        
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and weights must be equal'
                             f'; got {len(self.weights)} weights,'
                             f' {len(self.classifiers)} classifiers')
        
        self.lable_encoder_ = LabelEncoder()
        self.y_encoded_ = self.lable_encoder_.fit_transform(y)
        self.classes_ = self.lable_encoder_.classes_
        self.classifiers_ = []
        
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.y_encoded_)
            self.classifiers_.append(fitted_clf)
        
        return self
    
    
    def predict(self, X):
        if self.vote == 'probability':
            majority_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            # self.vote == 'classlabel'
            predictions = np.array([clf.predict(X) for clf in self.classifiers]).T
            majority_vote = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(X, self.weights[x])
                ),
                axis=1, arr=predictions
            )
        majority_vote = self.lable_encoder_.inverse_transform(majority_vote)
        return majority_vote
    
    
    def predict_proba(self, X):
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers])
        mean_weighted_probas = np.average(probas, weights=self.weights, axis=0)
        return mean_weighted_probas
    
    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers_.copy()
            for name, step in self.named_classifiers_.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out