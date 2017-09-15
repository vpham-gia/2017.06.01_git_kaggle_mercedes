from unittest import TestCase

import pandas as pd
from pandas.util.testing import assert_frame_equal

from mercedes.cleaning import TableCleaner


class TestTableCleaner(TestCase):
    df = pd.DataFrame({'a': [1, 0, 0, 1],
                       'b': [0, 0, 0, 1],
                       'toto': [0, 0, 0, 0],
                       'c': [1, 0, 0, 1],
                       'd': [0, 1, 1, 0],
                       'e': [1, 0, 0, 1],
                       'f': [0, 0, 0, 1]})

    def test_get_same_columns_for_var_1(self):
        """Test detection of same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_same_columns_for_var(var='a')

        # Then
        expected_res = [('a', 'c'), ('a', 'e')]
        assert res == expected_res

    def test_get_same_columns_for_var_2(self):
        """Test detection of same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_same_columns_for_var(var='c')

        # Then
        expected_res = [('c', 'e')]
        assert res == expected_res

    def test_get_same_columns_for_var_3(self):
        """Test detection of same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_same_columns_for_var(var='b')

        # Then
        expected_res = [('b', 'f')]
        assert res == expected_res

    def test_get_same_columns_for_var_4(self):
        """Test detection of same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_same_columns_for_var(var='toto')

        # Then
        expected_res = []
        assert res == expected_res

    def test_get_comp_columns_for_var_1(self):
        """Test detection of same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_complement_columns_for_var(var='a')

        # Then
        expected_res = [('a', 'd')]
        assert res == expected_res

    def test_get_comp_columns_for_var_2(self):
        """Test detection of same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_complement_columns_for_var(var='c')

        # Then
        expected_res = [('c', 'd')]
        assert res == expected_res

    def test_get_comp_columns_for_var_3(self):
        """Test detection of same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_complement_columns_for_var(var='b')

        # Then
        expected_res = []
        assert res == expected_res

    def test_get_all_same_columns(self):
        """Test detection of all same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_all_same_columns()

        # Then
        expected_res = [('a', 'c'), ('a', 'e'), ('b', 'f'), ('c', 'e')]
        assert res == expected_res

    def test_get_all_complement_columns(self):
        """Test detection of all same columns in a dataframe """

        # When
        res = TableCleaner(self.df).get_all_complement_columns()

        # Then
        expected_res = [('a', 'd'), ('c', 'd'), ('d', 'e')]
        assert res == expected_res
