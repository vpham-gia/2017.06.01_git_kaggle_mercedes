import os
import time

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score

import constants
from mercedes.feature_engineering import FeatureEngineering
from mercedes.model import ModelStacker

data = pd.read_csv(os.path.join(constants.PATH_DATA, 'train.csv'))
test_set = pd.read_csv(os.path.join(constants.PATH_DATA, 'test.csv'))
submission = pd.read_csv(os.path.join(constants.PATH_DATA,
                                      'sample_submission.csv'))

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
test_set.drop(to_drop, axis=1, inplace=True)

binary_vars = [x for x in data.columns
                    if x not in constants.CATEGORICAL_VARIABLES + ['ID', 'y']]
# ------------------------------------------------------------------------------

# Stacker ----------------------------------------------------------------------
ms = ModelStacker(df=data)
ms.set_fold_number(number_folds=5)

predictions = pd.DataFrame()
for fold_num in range(1, 6):
    train, validation = ms.train_test_split(fold_test_number=fold_num)

    # train.drop('fold_number', axis=1, inplace=True)
    # validation.drop('fold_number', axis=1, inplace=True)

    # Feature Engineering ------------------------------------------------------
    fe_train = FeatureEngineering(df=train.copy(deep=True))
    train_set_fe = fe_train.sum_all_binary_var(name='sum_all',
                                               list_binary_vars=binary_vars)
    train_set_fe = fe_train.project_all_categorical_on_target_mean(
        df_ref=train, list_cat_vars=constants.CATEGORICAL_VARIABLES,
        target='y', bool_replace=False)
    train_set_fe = fe_train.project_all_categorical_on_target_median(
        df_ref=train, list_cat_vars=constants.CATEGORICAL_VARIABLES,
        target='y', bool_replace=True)

    fe_val = FeatureEngineering(df=validation.copy(deep=True))
    val_set_fe = fe_val.sum_all_binary_var(name='sum_all',
                                           list_binary_vars=binary_vars)
    val_set_fe = fe_val.project_all_categorical_on_target_mean(
        df_ref=train, list_cat_vars=constants.CATEGORICAL_VARIABLES,
        target='y', bool_replace=False)
    val_set_fe = fe_val.project_all_categorical_on_target_median(
        df_ref=train, list_cat_vars=constants.CATEGORICAL_VARIABLES,
        target='y', bool_replace=True)

    # --------------------------------------------------------------------------


    # # Modeling CV ------------------------------------------------------------
    # # params = [{'n_estimators': [500], 'max_depth': [None], 'max_features': ['auto']}]
    #
    # rf_params = [{'n_estimators': [50, 100, 500],
    #               'max_depth': [None, 2, 4, 6, 8, 10],
    #               'max_features': ['auto', 5, 15, 50, 100, 200]}]
    #
    # clf = GridSearchCV(ExtraTreesRegressor(n_jobs=3,
    #                                          random_state=constants.RANDOM_STATE),
    #                    rf_params, cv=5, n_jobs=3)
    # clf.fit(X=train_set_fe.drop(['ID', 'y'], axis=1), y=train_set_fe['y'])
    # clf.best_estimator_.fit(X=train_set_fe.drop(['ID', 'y'], axis=1),
    #                         y=train_set_fe['y'])
    #
    #
    # # gbm_params = [{'learning_rate': [0.05, 0.1, 0.2],
    # #                'n_estimators': [50, 100, 500],
    # #                'max_depth': [None, 2, 4, 6, 8, 10],
    # #                'min_samples_split': [2, 5, 10, 50],
    # #                'max_features': ['auto', 5, 15, 50, 100, 200]}]
    # #
    # # te = time.time()
    # # clf = GridSearchCV(GradientBoostingRegressor(random_state=constants.RANDOM_STATE),
    # #                    gbm_params, cv=5, n_jobs=3)
    # # clf.fit(X=train_set_fe.drop(['ID', 'y'], axis=1), y=train_set_fe['y'])
    # # clf.best_estimator_.fit(X=train_set_fe.drop(['ID', 'y'], axis=1),
    # #                         y=train_set_fe['y'])
    # # ts = time.time()
    # # print('Time elapsed: ', ts - te)
    # # print('Best estimator: ', clf.best_estimator_)
    #
    # # knn_params = [{'n_neighbors': [2, 5, 10, 50],
    # #                'weights': ['uniform', 'distance'],
    # #                'p': [1, 2]}]
    # #
    # # clf = GridSearchCV(KNeighborsRegressor(n_jobs=3), knn_params, cv=5, n_jobs=3)
    # # clf.fit(X=train_set_fe[binary_vars], y=train_set_fe['y'])
    # # clf.best_estimator_.fit(X=train_set_fe[binary_vars], y=train_set_fe['y'])
    # # y_pred = clf.best_estimator_.predict(test_set_fe[binary_vars])
    # # ------------------------------------------------------------------------------

    # Models -------------------------------------------------------------------
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

    gbm_reg.fit(X=train_set_fe.drop(['ID', 'y', 'fold_number'], axis=1),
                y=train_set_fe['y'])
    rf_reg.fit(X=train_set_fe.drop(['ID', 'y', 'fold_number'], axis=1),
               y=train_set_fe['y'])
    et_reg.fit(X=train_set_fe.drop(['ID', 'y', 'fold_number'], axis=1),
               y=train_set_fe['y'])
    knn_reg.fit(X=train_set_fe[binary_vars], y=train_set_fe['y'])
    # ------------------------------------------------------------------------------

    # Predictions ------------------------------------------------------------------
    imputer = Imputer(missing_values='NaN', strategy='median')
    val_set_imputed = imputer.fit_transform(val_set_fe.drop(['ID', 'y', 'fold_number'], axis=1))

    gbm_pred = gbm_reg.predict(val_set_imputed)
    rf_pred = rf_reg.predict(val_set_imputed)
    et_pred = et_reg.predict(val_set_imputed)
    knn_pred = knn_reg.predict(val_set_fe[binary_vars])
    # ------------------------------------------------------------------------------

    # Predictions on validation set ------------------------------------------------
    pred_results = pd.DataFrame({'ID': val_set_fe['ID'], 'y': val_set_fe['y'],
                                 'gbm_pred': gbm_pred, 'rf_pred': rf_pred,
                                 'et_pred': et_pred, 'knn_pred': knn_pred})

    predictions = pd.concat([predictions, pred_results], axis=0)

    # print('Score RF:', r2_score(pred_results['y'], pred_results['rf_pred']))
    # print('Score GBM:', r2_score(pred_results['y'], pred_results['gbm_pred']))
    # print('Score ET:', r2_score(pred_results['y'], pred_results['et_pred']))
    # print('Score KNN:', r2_score(pred_results['y'], pred_results['knn_pred']))

data = pd.read_csv(os.path.join(constants.PATH_DATA, 'train.csv'))
data.drop(to_drop, axis=1, inplace=True)
print(data.shape)

fe_data = FeatureEngineering(df=data)
data_set_fe = fe_data.sum_all_binary_var(name='sum_over_over',
                                list_binary_vars=constants.BIN_OVER_OVER)
data_set_fe = fe_data.sum_all_binary_var(name='sum_over_under',
                                list_binary_vars=constants.BIN_OVER_UNDER)
data_set_fe = fe_data.sum_all_binary_var(name='sum_under_over',
                                list_binary_vars=constants.BIN_UNDER_OVER)
data_set_fe = fe_data.sum_all_binary_var(name='sum_under_under',
                                list_binary_vars=constants.BIN_UNDER_UNDER)

df_for_stacking = pd.merge(left=data_set_fe[['ID', 'sum_over_over', 'sum_over_under', 'sum_under_over', 'sum_under_under']], left_on='ID',
                           right=predictions, right_on='ID', how='left')

print(df_for_stacking.shape)
# ------------------------------------------------------------------------------

# Stacked model ----------------------------------------------------------------
rf_reg.fit(X=df_for_stacking.drop(['ID', 'y'], axis=1), y=df_for_stacking['y'])
feat_imp = pd.DataFrame({'imp': rf_reg.feature_importances_},
                        index=df_for_stacking.drop(['ID', 'y'], axis=1).columns)
feat_imp['imp_percent'] = round(100 * feat_imp['imp'] / sum(feat_imp['imp']),
                                ndigits=1)
feat_imp.sort_values('imp', ascending=False)

data = pd.read_csv(os.path.join(constants.PATH_DATA, 'train.csv'))
data.drop(to_drop, axis=1, inplace=True)

fe_data = FeatureEngineering(df=data.copy(deep=True))
data_set_fe = fe_data.sum_all_binary_var(list_binary_vars=binary_vars)
data_set_fe = fe_data.project_all_categorical_on_target_mean(
    df_ref=data, list_cat_vars=constants.CATEGORICAL_VARIABLES,
    target='y', bool_replace=False)
data_set_fe = fe_data.project_all_categorical_on_target_median(
    df_ref=data, list_cat_vars=constants.CATEGORICAL_VARIABLES,
    target='y', bool_replace=True)

fe_test = FeatureEngineering(df=test_set.copy(deep=True))
test_set_fe = fe_test.sum_all_binary_var(list_binary_vars=binary_vars)
test_set_fe = fe_test.project_all_categorical_on_target_mean(
    df_ref=data, list_cat_vars=constants.CATEGORICAL_VARIABLES,
    target='y', bool_replace=False)
test_set_fe = fe_test.project_all_categorical_on_target_median(
    df_ref=data, list_cat_vars=constants.CATEGORICAL_VARIABLES,
    target='y', bool_replace=True)

gbm_reg.fit(X=data_set_fe.drop(['ID', 'y'], axis=1), y=data_set_fe['y'])
rf_reg.fit(X=data_set_fe.drop(['ID', 'y'], axis=1), y=data_set_fe['y'])
et_reg.fit(X=data_set_fe.drop(['ID', 'y'], axis=1), y=data_set_fe['y'])
knn_reg.fit(X=data_set_fe[binary_vars], y=data_set_fe['y'])

imputer = Imputer(missing_values='NaN', strategy='median')
X_test_imputed = imputer.fit_transform(test_set_fe.drop('ID', axis=1))

gbm_pred = gbm_reg.predict(X_test_imputed)
rf_pred = rf_reg.predict(X_test_imputed)
et_pred = et_reg.predict(X_test_imputed)
knn_pred = knn_reg.predict(test_set_fe[binary_vars])

df_for_lm = pd.DataFrame({'gbm_pred': gbm_pred, 'rf_pred': rf_pred,
                          'et_pred': et_pred, 'knn_pred': knn_pred})
# ------------------------------------------------------------------------------
y_pred = reg.predict(df_for_lm)

df_to_submit = pd.concat([test_set_fe['ID'], pd.DataFrame({'y': y_pred})],
                         axis=1)
# ------------------------------------------------------------------------------
df_to_submit.to_csv(os.path.join(constants.PATH_SUBMISSION,
                                 '2017.06.13_4_Stacking_1Fold_lm.csv'),
                    index=False)

# TODO: foldID and predictions on dataset - check binary vars for big errors - create meta var (sum of important)


# feat_imp = pd.DataFrame({'imp': clf.best_estimator_.feature_importances_},
#                         index=train_set_fe.drop(['ID', 'y'], axis=1).columns)
# feat_imp['imp_percent'] = round(100 * feat_imp['imp'] / sum(feat_imp['imp']),
#                                 ndigits=1)
# print(feat_imp.sort_values('imp', ascending=False))
