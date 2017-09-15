import pandas as pd

class FeatureEngineering(object):
    """docstring for FeatureEngineering."""
    def __init__(self, df):
        super(FeatureEngineering, self).__init__()
        self.df = df

    def project_cat_var_on_target_mean(
                    self, df_ref, cat_var, target, bool_replace):
        df_categorical_target_mean = df_ref.groupby(cat_var)[target].mean()

        cat_proj = '{}_proj_mean'.format(cat_var)
        self.df[cat_proj] = self.df[cat_var].map(df_categorical_target_mean)

        if bool_replace:
            self.df.drop(cat_var, axis=1, inplace=True)

        return self.df

    def project_all_categorical_on_target_mean(
                    self, df_ref, list_cat_vars, target, bool_replace):
        for var in list_cat_vars:
            df_with_cat_projected = self.project_cat_var_on_target_mean(
                                            df_ref=df_ref, cat_var=var,
                                            target=target,
                                            bool_replace=bool_replace)

        return df_with_cat_projected

    def project_cat_var_on_target_median(
                    self, df_ref, cat_var, target, bool_replace):
        df_categorical_target_median = df_ref.groupby(cat_var)[target].median()

        cat_proj = '{}_proj_median'.format(cat_var)
        self.df[cat_proj] = self.df[cat_var].map(df_categorical_target_median)

        if bool_replace:
            self.df.drop(cat_var, axis=1, inplace=True)

        return self.df

    def project_all_categorical_on_target_median(
                    self, df_ref, list_cat_vars, target, bool_replace):
        for var in list_cat_vars:
            df_with_cat_projected = self.project_cat_var_on_target_median(
                                            df_ref=df_ref, cat_var=var,
                                            target=target,
                                            bool_replace=bool_replace)

        return df_with_cat_projected

    def sum_all_binary_var(self, name, list_binary_vars):
        df_copy = self.df
        df_copy[name] = df_copy[list_binary_vars].sum(axis=1)

        return df_copy


if __name__ == "__main__":
    df = pd.DataFrame({'a': ['aa', 'ab', 'aa', 'ab'], 'b': [1, 2, 5, 45]})
    a = FeatureEngineering(df).project_cat_var_on_target_median(
                        df_ref=df, cat_var='a', target='b', bool_replace=False)

    df = pd.DataFrame({'a': [1, 1, 1, 1], 'b': [1, 1, 0, 0]})
    df.sum(axis=0).values
