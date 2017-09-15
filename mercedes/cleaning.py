import pandas as pd

class TableCleaner(object):
    """docstring for TableCleaner."""
    def __init__(self, df):
        self.df = df

    def get_same_columns_for_var(self, var):
        list_of_double = list()
        var_pos = self.df.columns.get_loc(var)

        for ind in range(var_pos + 1, self.df.shape[1]):
            sumproduct = sum(self.df[var] == self.df[self.df.columns[ind]])

            if sumproduct == self.df.shape[0]:
                list_of_double.append((var, self.df.columns[ind]))

        return list_of_double

    def get_all_same_columns(self):
        list_couples_double = list()

        for col in self.df.columns:
            doubles = self.get_same_columns_for_var(var=col)
            list_couples_double.extend(doubles)

        return list_couples_double

    def get_variables_to_drop(self, list_of_tuples):
        return [x[1] for x in list_of_tuples]

    def get_complement_columns_for_var(self, var):
        list_of_complements = list()
        var_pos = self.df.columns.get_loc(var)

        try:
            complement = 1 - self.df[var]

            for ind in range(var_pos + 1, self.df.shape[1]):
                sumproduct = sum(complement == self.df[self.df.columns[ind]])

                if sumproduct == self.df.shape[0]:
                    list_of_complements.append((var, self.df.columns[ind]))
        except:
            pass

        return list_of_complements

    def get_all_complement_columns(self):
        list_complement = list()

        for col in self.df.columns:
            doubles = self.get_complement_columns_for_var(var=col)
            list_complement.extend(doubles)

        return list_complement






if __name__ == "__main__":
    expected_res = [('a', 'c'), ('a', 'e')]
    [x[1] for x in expected_res]
