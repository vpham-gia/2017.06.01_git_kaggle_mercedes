from unittest import TestCase

import pandas as pd
from pandas.util.testing import assert_frame_equal

from mercedes.feature_engineering import FeatureEngineering


class TestFeatureEngineering(TestCase):
    def test_project_cat_var_on_target_mean(self):
        """Test projection of categorical variable on target mean """

        # Given
        df = pd.DataFrame({'a': ['aa', 'ab', 'aa', 'ab'], 'b': [1, 2, 3, 4]})

        # When
        df_transformed = FeatureEngineering(df).project_cat_var_on_target_mean(
                                                            df_ref=df,
                                                            cat_var='a',
                                                            target='b',
                                                            bool_replace=False)

        # Then
        expected_res = pd.DataFrame({'a': ['aa', 'ab', 'aa', 'ab'],
                                     'b': [1, 2, 3, 4],
                                     'a_projected_mean': [2, 3, 2, 3]})

        df_transformed.equals(expected_res)
