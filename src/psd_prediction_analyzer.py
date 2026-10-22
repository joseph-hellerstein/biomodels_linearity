'''Analyzes predictions  from a Piecewise System Discovery object for a BioModel.'''

import src.constants as cn  # type: ignore
from src.psd_prediction_file_iterator import PSDPredictionFileIterator  # type: ignore
from src.model import Model  # type: ignore
from src.piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Union

DEFAULT_MAX_CHANGEPOINT = 5000
DEFAULT_MIN_ACCURACY = 0.8
DEFAULT_ACCURACY_STATISTIC = cn.COL_MEAN


class PSDPredictionAnalyzer:

    def __init__(self, model_id: Union[str, int],
            min_accuracy : float = DEFAULT_MIN_ACCURACY,
            accuracy_statistic: str = DEFAULT_ACCURACY_STATISTIC,
            is_guard_filename: bool = True,
            timecourse_df: pd.DataFrame | None = None,
            data_dir: str = cn.PSD_DATA_DIR) -> None:
        """Construct a PSDPredictionAnalyzer object.

        Args:
            model_id (Union[str, int]): The identifier of the model to analyze, either the number or its name.
            min_accuracy (float): The minimum accuracy threshold for predictions to be considered valid.
            accuracy_statistic (str): The statistic to use for accuracy calculations.
            is_guard_filename (bool): Whether to use the guard filename for the data files.
            timecourse_df (pd.DataFrame | None): The timecourse data for the model.
            data_dir (str): The directory containing the data files.
        """
        self.model_num, self.model_name = Model.getModelNumberAndName(model_id)
        self.data_dir = data_dir
        self.min_accuracy = min_accuracy
        self.accuracy_statistic = accuracy_statistic
        self.is_guard_filename = is_guard_filename
        self.timecourse_df = timecourse_df
        self.model_df = self.getModelDF() # DataFrame with PSD predictions for the model
        self.best_model_ser = self._getModelBestSer()  # Best PSD prediction

    def _filterDataframe(self, df: pd.DataFrame, dct: dict) -> pd.DataFrame:
        """Filters a DataFrame based on the provided dictionary of conditions.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            dct (dict): A dictionary of column-value pairs to filter by.

        Returns:
            pd.DataFrame: A filtered DataFrame.
        """
        try:
            mask = df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL
            mask &= df[cn.COL_MAX_CHANGEPOINT] == DEFAULT_MAX_CHANGEPOINT
            mask &= df[cn.COL_SYSTEM_ID] == self.model_name
            result_df = df[mask]
            return result_df
        except Exception as e:
            raise ValueError(f"Error filtering DataFrame for {dct}: {e}")

    def getModelDF(self) -> pd.DataFrame:
        """Gets a DataFrame with PSD predictions for the model.

        Returns:
            pd.DataFrame: A DataFrame containing the model data.
        """
        filtered_df = PSDPredictionFileIterator.query(is_guard_filename=self.is_guard_filename,
                dataframe_filter=self._filterDataframe, data_dir=self.data_dir, is_pkl= True)
        #
        return filtered_df

    def _getModelBestSer(self) -> "pd.Series | None":
        """Gets the best series for the model based on the optimization criteria:
            Min AUC subject to the accuracy constraints.

        Returns:
            The row in self.model_df with the minimum AUC among rows whose
            accuracy_statistic >= min_accuracy, or None if no row qualifies.
        """
        if self.model_df.empty:
            return None
        mask = self.model_df[self.accuracy_statistic] >= self.min_accuracy
        df_filtered = self.model_df.loc[mask].copy()
        if df_filtered.empty:
            return None
        best_idx = df_filtered[cn.COL_AUC].idxmin()
        ser = self.model_df.loc[best_idx]  # type: ignore[assignment]
        return ser

    def makeBestPSD(self) -> PiecewiseSystemDiscovery:
        """Creates a Model object for the best PSD prediction.

        Returns:
            Model: A Model object representing the best PSD prediction.
        """
        if self.timecourse_df is None:
            timecourse_df = TimecourseIterator.getTimecourse(self.model_name).timecourse_df
        else:
            timecourse_df = self.timecourse_df
        if self.best_model_ser is None:
            raise ValueError(f"No valid PSD predictions found for model {self.model_name} with min_accuracy {self.min_accuracy}.")
        psd = PiecewiseSystemDiscovery(timecourse_df, 
                changepoints=self.best_model_ser.changepoints,
                is_changepoint_removal=False,
                model_id=self.model_name)
        psd.fit()
        return psd

    def analyzeAUCConsistency(self) -> pd.DataFrame:
        """Analyze the consistency of the AUC (Area Under the Curve) for a given system.

        Returns:
            pd.DataFrame: A DataFrame containing the analysis results, including the AUC values and
            their consistency metrics.
        """
        grouped_df = self.model_df.groupby(cn.COL_MAX_FRACTIONAL_REDUCTION).agg({
            cn.COL_AUC: [cn.COL_MEAN, cn.COL_STD, cn.COL_COUNT],
        })
        grouped_df = grouped_df.replace({0: np.nan})
        grouped_df[cn.COL_CV] = grouped_df[(cn.COL_AUC, cn.COL_STD)] / grouped_df[(cn.COL_AUC, cn.COL_MEAN)]

        return pd.DataFrame(grouped_df)

    @classmethod
    def plotBestLinearFits(cls,
            model_ids: list[Union[str, int]],
            min_accuracy: float = DEFAULT_MIN_ACCURACY,
            accuracy_statistic: str = DEFAULT_ACCURACY_STATISTIC,
            is_guard_filename: bool = True,
            data_dir: str = cn.PSD_DATA_DIR) -> None:
        """
        Plots the best linear fits for a list of model IDs.

        Args:
            model_ids (list[Union[str, int]]): A list of model identifiers (either numbers or names).
            min_accuracy (float): The minimum accuracy threshold for predictions to be considered valid.
            accuracy_statistic (str): The statistic to use for accuracy calculations.
            is_guard_filename (bool): Whether to use the guard filename for the data files.
            data_dir (str): The directory containing the data files.    
        """
        for model_id in model_ids:
            analyzer = cls(model_id, min_accuracy=min_accuracy,
                    accuracy_statistic=accuracy_statistic,
                    is_guard_filename=is_guard_filename,
                    data_dir=data_dir)
            psd = analyzer.makeBestPSD()
            psd.plotPiecewise(is_changepoint_plot=True, legend=False, num_true_point=60, is_nochangepoint_plot=False)