import pandas as pd
import math

class Node:
    def __init__(self, feature: str, threshold: int):
        self.feature = feature
        self.threshold = threshold
        self.value = None
        self.left = None
        self.right = None

    def _minus(self, a: float, b: float) -> bool:
        return a.astype(float) < b

    def _bigger(self, a: float, b: float) -> bool:
        return a.astype(float) >= b

    def _get_count_outcomes(self, y: pd.Series, search_variable: int) -> int:
        counts = y.value_counts()

        if search_variable not in counts:
            return 0

        return counts[search_variable]

    def _get_entropy(self, bool_variable: float) -> float:
        if bool_variable in [0, 1]:
            return 0

        not_chance = 1 - bool_variable

        entropy = -(bool_variable * math.log2(bool_variable) + not_chance * math.log2(not_chance))

        return entropy

    def _get_positive_distribution(self, y: pd.Series) -> float:
        num_neg = self._get_count_outcomes(y, 0)
        num_pos = self._get_count_outcomes(y, 1)

        count = num_neg + num_pos

        if count == 0:
            return 0

        return num_pos / count

    def _get_subset_entropy(self, y: pd.Series) -> float:
        pos_distribution = self._get_positive_distribution(y)

        return self._get_entropy(pos_distribution)

    def _get_distribution_threshold(self, x: pd.Series, threshold: float, rule) -> float:
        subset_x = x[rule(x, threshold)]

        count_subset = len(subset_x)

        if count_subset == 0:
            return 0

        return count_subset / len(x)

    def _get_output_by_threshold(self, x, y, threshold, rule):
        ocurrences_idx = x[rule(x, threshold)].index.tolist()

        return y.iloc[ocurrences_idx].reset_index(drop = True)

    def _get_att_split_information_gain(self, x: pd.Series, y: pd.Series, threshold: int, parent_entropy: float) -> float:
        percentage_below_average = self._get_distribution_threshold(x, threshold, self._minus)

        if percentage_below_average in [0, 1]:
            return 0

        subset_y_below_threshold = self._get_output_by_threshold(x, y, threshold, self._minus)
        below_threshold_entropy = self._get_subset_entropy(subset_y_below_threshold)

        percentage_above_average = self._get_distribution_threshold(x, threshold, self._bigger)
        subset_y_above_threshold = self._get_output_by_threshold(x, y, threshold, self._bigger)
        above_threshold_entropy = self._get_subset_entropy(subset_y_above_threshold)

        return parent_entropy - (percentage_below_average * below_threshold_entropy + percentage_above_average * above_threshold_entropy)

    def _get_best_split(self, x: pd.DataFrame, y: pd.Series):
        parent_entropy = self._get_subset_entropy(y)

        if parent_entropy == 0:
            return None, 0, 0

        best_split = None
        best_threshold = 0
        information_gain = 0

        for attribute in x.columns:
            subset_x = x[attribute].astype(float)

            categories = subset_x.unique()
            
            if len(categories) == 2:
                categories = [0.5]
            else:
                categories = sorted(categories)[1:]

            for category in categories:
                attribute_information_gain = self._get_att_split_information_gain(subset_x, y, category, parent_entropy)

                if attribute_information_gain > information_gain:
                    best_split = attribute
                    best_threshold = category
                    information_gain = attribute_information_gain

        return (best_split, best_threshold, information_gain)

    def _get_value_node(self, y: pd.Series) -> int:
        pos_distribution = self._get_positive_distribution(y)

        return round(pos_distribution)

    def _get_x_y_by_threshold_rule(self, x: pd.DataFrame, y: pd.Series, feature: str, threshold: float, rule):
        subset_x = x[rule(x[feature], threshold)].reset_index(drop = True)
        subset_y = self._get_output_by_threshold(x[feature], y, threshold, rule)

        return subset_x, subset_y

    def create_tree(self, x: pd.DataFrame, y: pd.Series, max_depth: int, depth: int = 0) -> 'Node':
        best_split, threshold, information_gain = self._get_best_split(x, y)

        if depth > max_depth or best_split == None:
            self.value = self._get_value_node(y)
            return

        self.feature = best_split
        self.threshold = threshold

        self.left = Node(None, 0)
        x_below_threshold, y_below_threshold = self._get_x_y_by_threshold_rule(x, y, best_split, threshold, self._minus)
        self.left.create_tree(x_below_threshold, y_below_threshold, max_depth, depth + 1)

        self.right = Node(None, 0)
        x_above_threshold, y_above_threshold = self._get_x_y_by_threshold_rule(x, y, best_split, threshold, self._bigger)
        self.right.create_tree(x_above_threshold, y_above_threshold, max_depth, depth + 1)

    def make_prediction(self, x: pd.DataFrame):
        if self.value != None:
            return self.value

        if (x[self.feature].astype(float) < self.threshold).bool():
            return self.left.make_prediction(x)

        return self.right.make_prediction(x)