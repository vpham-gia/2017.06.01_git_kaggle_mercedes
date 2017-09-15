import os
import pandas as pd
import stats_desc

import constants
from mercedes.cleaning import TableCleaner
from mercedes.feature_engineering import FeatureEngineering


data = pd.read_csv(os.path.join(constants.PATH_DATA, 'train.csv'))
test_set = pd.read_csv(os.path.join(constants.PATH_DATA, 'test.csv'))

full_df = pd.concat([data.drop('y', axis=1), test_set],
axis=0, ignore_index=True)

# list_same_cols_data = TableCleaner(df=data).get_all_same_columns()
# len(list_same_cols_data)

list_same_cols = TableCleaner(df=full_df).get_all_same_columns()
print(list_same_cols)
print(len(list_same_cols))

list_double = TableCleaner(df=full_df).\
                    get_variables_to_drop(list_of_tuples=list_same_cols)
# Same columns
list(set(list_double))

list_complement_cols = TableCleaner(df=full_df).get_all_complement_columns()
print(list_complement_cols)
print(len(list_complement_cols))

# Complements
[('X128', 'X130'), ('X156', 'X157'), ('X204', 'X205'), ('X232', 'X263'), ('X263', 'X279')]

fe_train = FeatureEngineering(df=data.copy(deep=True))
train_set_fe = fe_train.project_all_categorical_on_target_mean(
    df_ref=data, list_cat_vars=constants.CATEGORICAL_VARIABLES,
    target='y', bool_replace=True)
