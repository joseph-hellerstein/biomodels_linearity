'''Scales and descales data for system discovery.'''

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

class Scaler(object):
    '''Scales and descales data for system discovery.'''
    def __init__(self, data_df: pd.DataFrame, is_null_scaler: bool = False
            )-> None:
        """
        Args:
            data_df (pd.DataFrame): data to be scaled
            is_null_scaler (bool, optional): whether to use a null scaler. Defaults to False.
        """
        self._column_names = data_df.columns
        self._matrix_arr = data_df.values
        self._num_row, self._num_col = self._matrix_arr.shape
        self._is_null_scaler = is_null_scaler
        # Scale
        self._constant_cols: frozenset[str] = frozenset()
        if not self._is_null_scaler:
            self._scale_arr = np.std(self._matrix_arr, axis=0, ddof=1)
            col_means = np.mean(self._matrix_arr, axis=0)
            constant: list[str] = []
            for icol in range(self._num_col):
                if np.isclose(self._scale_arr[icol], 0):
                    constant.append(self._column_names[icol])
                    self._scale_arr[icol] = 1.0 / col_means[icol] if col_means[icol] != 0 else 1.0
            self._constant_cols = frozenset(constant)
        else:
            self._scale_arr = np.ones(self._num_col)
        self._scaling_dct = dict(zip(self._column_names, self._scale_arr))

    def _parseFeaturePowers(self, feature_name: str) -> dict[str, int]:
        """Parse a PySINDy feature name into {species_name: power}.

        Handles: '1' → {}, 'X' → {'X': 1}, 'X^2' → {'X': 2}, 'X Y' → {'X': 1, 'Y': 1}.
        """
        if feature_name == "1":
            return {}
        power_dct: dict[str, int] = {}
        for factor in feature_name.split(" "):
            if not factor:
                continue
            if "^" in factor:
                name, exp = factor.split("^", 1)
                power_dct[name] = int(exp)
            else:
                power_dct[factor] = 1
        return power_dct

    def normalize(self, matrix_arr: np.ndarray) -> np.ndarray:
        """Performs normalization. Handles constant valued data
            by setting normalized values to 1.

        Args:
            matrix_arr (list[np.ndarray]): data to be normalized
                if matrix_arr == np.array([]), then use self._matrix_arr for normalization

        Returns:
            list[np.ndarray]: normalized data
        """
        if matrix_arr.size == 0:
            matrix_arr = self._matrix_arr
        col_names = list(self._column_names)
        constant_indices = [col_names.index(n) for n in self._constant_cols]
        safe_scale = self._scale_arr.copy()
        for icol in constant_indices:
            safe_scale[icol] = 1.0
        normalized_arr = matrix_arr / safe_scale
        for icol in constant_indices:
            if normalized_arr.ndim == 1:
                normalized_arr[icol] = 1.0
            else:
                normalized_arr[:, icol] = 1.0
        return normalized_arr
    
    def denormalize(self, normalized_arr: np.ndarray) -> np.ndarray:
        """Performs denormalization. Handles constant valued data
            by setting denormalized values to the mean.

        Args:
            normalized_arr (list[np.ndarray]): normalized data

        Returns:
            list[np.ndarray]: denormalized data
        """
        try:
            denormalized_arr = normalized_arr * self._scale_arr
            col_names = list(self._column_names)
            constant_indices = [col_names.index(n) for n in self._constant_cols]
            for icol in constant_indices:
                mean_val = np.mean(self._matrix_arr[:, icol])
                if denormalized_arr.ndim == 1:
                    denormalized_arr[icol] = mean_val
                else:
                    denormalized_arr[:, icol] = mean_val
        except Exception as e:
            import pdb; pdb.set_trace()  # noqa: E702
            raise e
        return denormalized_arr

#    def deprecatedDenormalize(self, normalized_arr: np.ndarray) -> np.ndarray:
#        """Performs denormalization. Handles constant valued data
#            by setting denormalized values to the mean.
#
#        Args:
#            normalized_arr (list[np.ndarray]): normalized data
#
#        Returns:
#            list[np.ndarray]: denormalized data
#        """
#        denormalized_arr = normalized_arr * self._scale_arr
#        for icol, std in enumerate(self._scale_arr):
#            if np.isclose(std, 0):
#                if denormalized_arr.ndim == 1:
#                    denormalized_arr[icol] = np.mean(self._matrix_arr[:, icol])
#                else:
#                    denormalized_arr[:, icol] = np.mean(self._matrix_arr[:, icol])
#        return denormalized_arr
    
    def normalizeThreshold(self, state_variable: str, feature_str: str,
            physical_threshold: float) -> float:
        """Converts a physical-space coefficient threshold to normalized space.

        Returns t_norm such that |c_norm| < t_norm iff |c_physical| < physical_threshold.

        Args:
            state_variable (str): ODE output species name
            feature_str (str): feature string, e.g., "S1", "S2^2", "S1 S2"
            physical_threshold (float): threshold in physical units

        Returns:
            float: equivalent threshold in normalized coefficient space
        """
        if self._is_null_scaler:
            return physical_threshold
        powers = self._parseFeaturePowers(feature_str)
        factor_adj = 1.0
        for column_name, power in powers.items():
            if column_name in self._scaling_dct:
                factor_adj *= float(self._scaling_dct[column_name]) ** power
        return physical_threshold * factor_adj / self._scaling_dct[state_variable]

    def denormalizeCoordinate(self, state_variable, factor_str,
            normalized_value) -> float:
        """Denormalizes a single coordinate value, where a coordinate is a
        product of state variables that raised to a power. The denomalization depends
        on both the coordinate the the state variable.  For example, for a
        coordinate S1^2 S2, the denormalization of S1 depends on both S1 and S2.

        Args:
            state_variable (str): state variable name
            factor_str (str): factor string, e.g., "S1", "S2^2", "S1 S2"
            normalized_value (float): normalized coordinate value

        Returns:
            float: denormalized coordinate value
        """
        if self._is_null_scaler:
            return normalized_value
        #
        powers = self._parseFeaturePowers(factor_str)
        factor_adj = 1.0
        for column_name, power in powers.items():
            if column_name in self._scaling_dct:
                factor_adj *= float(self._scaling_dct[column_name]) ** power
        # Adjust the value
        denormalized_value = normalized_value * self._scaling_dct[state_variable] / factor_adj
        return denormalized_value