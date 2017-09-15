import pandas as pd
import os

from mercedes.stats_desc.stats_desc import StatsDesc
import mercedes.constants
from mercedes.feature_engineering import FeatureEngineering

data = pd.read_csv('../../data/train.csv')

fe_train = FeatureEngineering(df=data.copy(deep=True))
train_set_fe = fe_train.project_all_categorical_on_target_mean(
    df_ref=data, list_cat_vars=mercedes.constants.CATEGORICAL_VARIABLES,
    target='y', bool_replace=False)

df_stats_desc = train_set_fe.drop(['ID'], axis=1)

var_type = ['numeric']
var_type.extend(['categorical' for x in data.columns if x.startswith('X')])
var_type.extend(['numeric' for x in train_set_fe.columns if x.endswith('_mean')])

sd = StatsDesc()
sd.save_eda(df=df_stats_desc, list_variables=list(df_stats_desc.columns),
            target='y', target_type='numeric',
            autodetect_types=False, type_variables=var_type)
