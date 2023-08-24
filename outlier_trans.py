import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from streamlit_gallery.utils.helper_function import (
    delete_potential_outlier_list,
    delete_potential_outlier,
    outlier_thresholds
)


class OutlierTrans(BaseEstimator, TransformerMixin):

    def __init__(self, outlierlist):
        self.outlierlist = outlierlist

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return delete_potential_outlier_list(X, self.outlierlist)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features"""
        if input_features is None:
            raise ValueError(
                "input_features should be the names of the features from input DataFrame")
        return np.array(input_features)
