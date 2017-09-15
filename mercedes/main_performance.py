import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

import constants
from mercedes.feature_engineering import FeatureEngineering
from mercedes.model import ModelStacker

data = pd.read_csv(os.path.join(constants.PATH_DATA, 'train.csv'))

# Cleaning ---------------------------------------------------------------------
list_single_value_in_data = ['X11', 'X93', 'X107', 'X233', 'X235',
                             'X268', 'X289', 'X290', 'X293', 'X297',
                             'X330', 'X347']

list_double = ['X296', 'X227', 'X119', 'X302', 'X216', 'X253', 'X330', 'X213',
               'X113', 'X147', 'X146', 'X262', 'X134', 'X226', 'X37', 'X254',
               'X239', 'X102', 'X214', 'X360', 'X172', 'X76', 'X293', 'X382',
               'X324', 'X364', 'X222', 'X299', 'X385', 'X84', 'X244', 'X35',
               'X39', 'X279', 'X326', 'X199']

list_complement = []# list_complement = ['X130', 'X157', 'X205', 'X263', 'X279']
to_drop = list(set(list_double + list_complement))

data.drop(to_drop, axis=1, inplace=True)

binary_vars = [x for x in data.columns
                    if x not in constants.CATEGORICAL_VARIABLES + ['ID', 'y']]
# ------------------------------------------------------------------------------

# Stacker ----------------------------------------------------------------------
ms = ModelStacker(df=data)
ms.set_fold_number(number_folds=5)

results_all = pd.DataFrame()

for i in range(1, 6):
    train, validation = ms.train_test_split(fold_test_number=i)

    # train.drop('fold_number', axis=1, inplace=True)
    # validation.drop('fold_number', axis=1, inplace=True)
# ------------------------------------------------------------------------------

# Feature Engineering ----------------------------------------------------------
    fe_train = FeatureEngineering(df=train.copy(deep=True))
    train_set_fe = fe_train.sum_all_binary_var(list_binary_vars=binary_vars)
    train_set_fe = fe_train.project_all_categorical_on_target_mean(
        df_ref=train, list_cat_vars=constants.CATEGORICAL_VARIABLES,
        target='y', bool_replace=False)
    train_set_fe = fe_train.project_all_categorical_on_target_median(
        df_ref=train, list_cat_vars=constants.CATEGORICAL_VARIABLES,
        target='y', bool_replace=True)

    fe_val = FeatureEngineering(df=validation.copy(deep=True))
    validation_fe = fe_val.sum_all_binary_var(list_binary_vars=binary_vars)
    validation_fe = fe_val.project_all_categorical_on_target_mean(
        df_ref=train, list_cat_vars=constants.CATEGORICAL_VARIABLES,
        target='y', bool_replace=False)
    validation_fe = fe_val.project_all_categorical_on_target_median(
        df_ref=train, list_cat_vars=constants.CATEGORICAL_VARIABLES,
        target='y', bool_replace=True)
# ------------------------------------------------------------------------------

# Models -----------------------------------------------------------------------
    gbm_reg = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                              learning_rate=0.1, loss='ls', max_depth=2,
                              max_features='auto', max_leaf_nodes=None,
                              min_impurity_split=1e-07, min_samples_leaf=1,
                              min_samples_split=50, min_weight_fraction_leaf=0.0,
                              n_estimators=100, presort='auto', random_state=92,
                              subsample=1.0, verbose=0, warm_start=False)

    knn_reg = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=3, n_neighbors=10, p=1,
              weights='uniform')

    rf_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=50, n_jobs=3, oob_score=False, random_state=92,
               verbose=0, warm_start=False)

    et_reg = ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=4,
              max_features=200, max_leaf_nodes=None, min_impurity_split=1e-07,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=3,
              oob_score=False, random_state=92, verbose=0, warm_start=False)

    gbm_reg.fit(X=train_set_fe.drop(['ID', 'y', 'fold_number'], axis=1), y=train_set_fe['y'])
    rf_reg.fit(X=train_set_fe.drop(['ID', 'y', 'fold_number'], axis=1), y=train_set_fe['y'])
    et_reg.fit(X=train_set_fe.drop(['ID', 'y', 'fold_number'], axis=1), y=train_set_fe['y'])
    knn_reg.fit(X=train_set_fe[constants.IMPORTANT_BINARY], y=train_set_fe['y'])
# ------------------------------------------------------------------------------

# Predictions ------------------------------------------------------------------
    imputer = Imputer(missing_values='NaN', strategy='median')
    val_set_imputed = imputer.fit_transform(validation_fe.drop(['ID', 'y', 'fold_number'], axis=1))

    gbm_pred = gbm_reg.predict(val_set_imputed)
    rf_pred = rf_reg.predict(val_set_imputed)
    et_pred = et_reg.predict(val_set_imputed)
    knn_pred = knn_reg.predict(validation_fe[constants.IMPORTANT_BINARY])
# ------------------------------------------------------------------------------

# Analysis of predictions ------------------------------------------------------
    results = pd.DataFrame({'id': validation_fe['ID'], 'y': validation_fe['y'],
                            'gbm_pred': gbm_pred, 'rf_pred': rf_pred,
                            'et_pred': et_pred, 'knn_pred': knn_pred})

    results_all = pd.concat([results_all, results], axis=0)

    print('Score RF:', r2_score(results['y'], results['rf_pred']))
    print('Score GBM:', r2_score(results['y'], results['gbm_pred']))
    print('Score ET:', r2_score(results['y'], results['et_pred']))
    print('Score KNN:', r2_score(results['y'], results['knn_pred']))

# ------------------------------------------------------------------------------
# validation_fe[['ID', 'X0_proj_mean']]

results_all.head()

set(results_all['id']) - set(data['ID'])
set(data['ID']) - set(results_all['id'])
results_all.shape
data.shape

results = results_all.copy(deep=True)

for model in ['rf', 'gbm', 'et', 'knn']:
    results['{}_error'.format(model)] = round(100 * (results['{}_pred'.format(model)] - results['y'])/results['y'], 1)

results['min_error'] = results[['rf_error', 'gbm_error', 'et_error', 'knn_error']].min(axis=1)
results['max_error'] = results[['rf_error', 'gbm_error', 'et_error', 'knn_error']].max(axis=1)
results.sort_values('min_error', ascending=False)

import matplotlib.pyplot as plt
results['min_error'].plot(kind='box')
plt.show()


results['type'] = pd.cut(results['min_error'], bins=[-100, -5, 5, 20], labels=[-1, 0, 1])
results['type'].value_counts()

out = pd.merge(left=data, left_on='ID',
               right=results[['id', 'type']], right_on='id', how='left')
out.to_csv('analysis_full_dataset.csv')
