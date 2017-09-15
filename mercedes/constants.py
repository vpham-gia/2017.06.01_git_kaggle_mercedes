PATH_DATA = "../data/"
PATH_SUBMISSION = "../submissions/"

RANDOM_STATE = 92

CATEGORICAL_VARIABLES = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

IMPORTANT_BINARY_1 = ['X14', 'X45', 'X114', 'X116', 'X127', 'X143', 'X154',
                    'X155', 'X178', 'X195', 'X218', 'X223', 'X241', 'X250',
                    'X261', 'X264', 'X273', 'X283', 'X285', 'X301', 'X313',
                    'X314', 'X321', 'X334', 'X374', 'X376']

IMPORTANT_BINARY = ['X45', 'X51', 'X58', 'X96', 'X115', 'X118', 'X157', 'X218',
                    'X234', 'X247', 'X250', 'X261', 'X275', 'X300', 'X311',
                    'X314', 'X316', 'X321', 'X327', 'X334', 'X337',	'X350',
                    'X354', 'X355',	'X356', 'X374']

BIN_OVER_OVER = ['X115', 'X234', 'X250', 'X316', 'X321', 'X337', 'X356', 'X374']
BIN_OVER_UNDER = ['X45', 'X51', 'X96', 'X118', 'X218', 'X261', 'X300', 'X314', 'X327', 'X334', 'X350', 'X355']
BIN_UNDER_OVER = ['X45', 'X118', 'X157', 'X247', 'X261', 'X275', 'X311', 'X327', 'X354', 'X355', 'X356']
BIN_UNDER_UNDER = ['X115', 'X234', 'X316']
# # Results from CV
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
#           metric_params=None, n_jobs=3, n_neighbors=10, p=1,
#           weights='uniform')

# GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#                           learning_rate=0.1, loss='ls', max_depth=2,
#                           max_features='auto', max_leaf_nodes=None,
#                           min_impurity_split=1e-07, min_samples_leaf=1,
#                           min_samples_split=50, min_weight_fraction_leaf=0.0,
#                           n_estimators=100, presort='auto', random_state=92,
#                           subsample=1.0, verbose=0, warm_start=False)
