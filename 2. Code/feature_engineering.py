import pandas as pd

class FeatureEngineering(object):
    """docstring for FeatureEngineering."""
    def __init__(self, df):
        super(FeatureEngineering, self).__init__()
        self.df = df

    def project_categorical_variable_on_target_mean(self, cat_var, target):
        df_copy = self.df.copy(deep=True)

        df_categorical_target_mean = self.df.groupby(cat_var)[target].mean()

        cat_proj = '{}_projected_mean'.format(cat_var)
        df_copy[cat_proj] = df_copy[cat_var].map(df_categorical_target_mean)
        return df_copy

    def add_all_binary_var(self, list_binary_vars):
        pass


if __name__ == "__main__":
    test = pd.DataFrame({'a': ['aa', 'ab', 'aa', 'ab'], 'b': [1, 2, 3, 4]})

    fe = FeatureEngineering(df=test)
    toto = fe.project_categorical_variable_on_target_mean(cat_var='a', target='b')
