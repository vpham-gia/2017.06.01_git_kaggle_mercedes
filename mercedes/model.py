import pandas as pd
import numpy as np

import constants
np.random.seed(seed=constants.RANDOM_STATE)

class Predictor(object):
    """docstring for Predictor."""
    def __init__(self):
        pass

    def training(self):
        pass

    def cross_validate(self, features, target):
        pass

    def get_feat_imp_table(self, model, prefix):
        pass


class ModelStacker(object):
    """docstring for ModelStacker."""
    def __init__(self, df):
        self.df = df

    def set_fold_number(self, number_folds):
        random_numbers = np.random.uniform(low=0, high=1, size=self.df.shape[0])
        self.df['fold_number'] = pd.qcut(x=random_numbers, q=number_folds,
                                         labels=[(i + 1) for i in range(number_folds)])
        pass

    def train_test_split(self, fold_test_number):
        test = self.df[self.df['fold_number'] == fold_test_number]
        train = self.df[self.df['fold_number'] != fold_test_number]

        # train.drop('fold_number', axis=1, inplace=True)
        # test.drop('fold_number', axis=1, inplace=True)

        return train, test


if __name__ == "__main__":
    df = pd.DataFrame({'a': ['aa', 'ab', 'aa', 'ab'], 'b': [1, 2, 5, 45]})

    # df = pd.DataFrame({'a': [1, 1, 1, 1], 'b': [1, 1, 0, 0]})
    a = ModelStacker(df=df).set_fold_number(number_folds=4)
    train, test = ModelStacker(df=df).train_test_split(fold_test_number=2)

    train
