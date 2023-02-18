import pandas as pd
import math

class Node:
    def __init__(self, feature: str, threshold: int, left: 'Node' = None, right: 'Node' = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def _get_count_outcomes(self, y: pd.Series, search_variable: int) -> int:
        counts = y.value_counts()

        if search_variable not in counts:
            return 0

        return counts[search_variable]

    def _get_entropy(self, bool_variable: float) -> float:
        not_chance = 1 - bool_variable

        if bool_variable == 0:
            entropy = -(not_chance * math.log2(not_chance))
        elif not_chance == 0:
            entropy = -(bool_variable * math.log2(bool_variable))
        else:
            entropy = -(bool_variable * math.log2(bool_variable) + not_chance * math.log2(not_chance))

        return entropy

    def _get_remainder_op(self, t_pos, t_neg, s_pos, s_neg) -> float:
        s_count = s_pos + s_neg
        count = t_pos + t_neg

        s_pos_distribution = s_pos / s_count

        s_entropy = self._get_entropy(s_pos_distribution)

        return (s_count / count) * s_entropy

    def _get_remainder_attribute(self, x: pd.Series, y: pd.Series, t_pos: int, t_neg: int) -> float:
        attribute_values = x.unique()

        total_remainder = 0

        for value in attribute_values:
            occurence_indexes = x[x == value].index.tolist()

            subset_y = y.iloc[occurence_indexes]

            s_neg = self._get_count_outcomes(subset_y, 0)
            s_pos = self._get_count_outcomes(subset_y, 1)

            remainder = self._get_remainder_op(t_pos, t_neg, s_pos, s_neg)

            total_remainder += remainder

        return total_remainder

    def _get_attribute_gain(self, x: pd.Series, y: pd.Series, t_pos: int, t_neg: int) -> float:
        count = t_pos + t_neg
        pos_distribution = t_pos / count

        total_entropy = self._get_entropy(pos_distribution)
        total_remainder = self._get_remainder_attribute(x, y, t_pos, t_neg)

        return total_entropy - total_remainder

    def create_tree(self, x: pd.DataFrame, y: pd.Series):

        print('dd')