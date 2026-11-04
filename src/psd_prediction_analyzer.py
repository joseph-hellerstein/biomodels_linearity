'''Analyzes predictions  from a Piecewise System Discovery object for a BioModel.'''

import src.constants as cn  # type: ignore
from src.psd_prediction_file_iterator import PSDPredictionFileIterator  # type: ignore
from src.model import Model  # type: ignore
from src.piecewise_system_discovery import PiecewiseSystemDiscovery, ExtendedPlotOptions  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

import os
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Union, List, Any, cast, Optional

OUTPUT_PATH_PAT = os.path.join(cn.DATA_DIR, "psd_prediction_analyzer__minaccuracy_%s__accuracystat_%s__.csv")


class PSDPredictionAnalyzer:

    def __init__(self, model_id: Union[str, int],
            min_accuracy : float = cn.DEF_MIN_ACCURACY,
            accuracy_stat: str = cn.DEF_ACCURACY_STAT,
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
        self.accuracy_statistic = accuracy_stat
        self.is_guard_filename = is_guard_filename

        self.timecourse_df = timecourse_df
        self.model_df = self.getModelDF() # DataFrame with PSD predictions for the model
        self.best_model_ser = self._getModelBestSer()  # Best PSD prediction

    def _checkInitialized(self) -> None:
        """Checks if the PSDPredictionAnalyzer object is properly initialized.
        """
        if self.model_df is None or self.best_model_ser is None:
            raise ValueError("PSDPredictionAnalyzer is not properly initialized.")

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
            mask &= df[cn.COL_MAX_CHANGEPOINT] == cn.DEF_MAX_CHANGEPOINT
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
        #filtered_df[cn.COL_CHANGEPOINTS] = filtered_df[cn.COL_CHANGEPOINTS].apply(json.loads)
        #filtered_df[cn.COL_BOUNDARIES] = filtered_df[cn.COL_BOUNDARIES].apply(json.loads)
        changepoint_list: List[Any] = []
        boundaries_list: List[Any] = []
        for _, row in filtered_df.iterrows():
            if isinstance(row[cn.COL_CHANGEPOINTS], str):
                row[cn.COL_CHANGEPOINTS] = eval(row[cn.COL_CHANGEPOINTS])
            changepoint_list.append(row[cn.COL_CHANGEPOINTS])
            if isinstance(row[cn.COL_BOUNDARIES], str):
                row[cn.COL_BOUNDARIES] = eval(row[cn.COL_BOUNDARIES])
            boundaries_list.append(row[cn.COL_BOUNDARIES])
        #
        filtered_df[cn.COL_CHANGEPOINTS] = changepoint_list
        filtered_df[cn.COL_BOUNDARIES] = boundaries_list
        return filtered_df.reset_index()

    def _getModelBestSer(self) -> pd.Series | None:
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
        self._checkInitialized()
        if self.timecourse_df is None:
            timecourse_df = TimecourseIterator.getTimecourse(self.model_name).timecourse_df
        else:
            timecourse_df = self.timecourse_df
        if cn.COL_COEFFICIENT_THRESHOLD not in self.best_model_ser:  # type: ignore
            coefficient_threshold = cn.DEF_COEFFICIENT_THRESHOLD
        else:
            coefficient_threshold = self.best_model_ser[cn.COL_COEFFICIENT_THRESHOLD] # type: ignore
        if self.best_model_ser is None:
            raise ValueError(f"No valid PSD predictions found for model {self.model_name} with min_accuracy {self.min_accuracy}.")
        psd = PiecewiseSystemDiscovery(timecourse_df, 
                changepoints=self.best_model_ser.changepoints,
                coefficient_threshold=coefficient_threshold,
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
        self._checkInitialized()
        grouped_df = self.model_df.groupby(cn.COL_MAX_FRACTIONAL_REDUCTION).agg({
            cn.COL_AUC: [cn.COL_MEAN, cn.COL_STD, cn.COL_COUNT],
        })
        grouped_df = grouped_df.replace({0: np.nan})
        grouped_df[cn.COL_CV] = grouped_df[(cn.COL_AUC, cn.COL_STD)] / grouped_df[(cn.COL_AUC, cn.COL_MEAN)]

        return pd.DataFrame(grouped_df)

    @classmethod
    def makeBestPSDModelDF(cls, min_accuracy: float = cn.DEF_MIN_ACCURACY,
            accuracy_stat: str = cn.DEF_ACCURACY_STAT,
            output_path_pat: str = OUTPUT_PATH_PAT,
            ) -> pd.DataFrame:
        """Create the best PSD models for all models meeting the minimum accuracy threshold.

        Args:
            min_accuaracy (float): The minimum accuracy threshold for predictions to be considered valid.
            output_path_pat (str): The path pattern for saving the output CSV file containing the best PSD models.
                or a plain file path if no formatting is needed.
            accuracy_stat (str): The accuracy statistic to use for determining the best models.

        Returns:
            pd.DataFrame: A DataFrame containing the best PSD models and their associated metadata.
        """
        try:
            output_path = output_path_pat % (str(min_accuracy), accuracy_stat)
        except TypeError:
            output_path = output_path_pat
        #
        if os.path.exists(output_path):
            print(f"Output file already exists at {output_path}. Loading it instead of generating a new one.")
            df = pd.read_csv(output_path)
            return df.reset_index()
        #
        t_iter = TimecourseIterator()
        sers: List[pd.Series] = []
        best_model_df = pd.DataFrame()
        for item in t_iter:
            analyzer = PSDPredictionAnalyzer(item.model_name, min_accuracy=min_accuracy,
                    accuracy_stat=accuracy_stat)
            sers.append(cast(pd.Series, analyzer.best_model_ser))
            best_model_df = pd.concat(sers, axis=1).T
            best_model_df.to_csv(output_path, index=False)
        # Save the results to the specified output path
        best_model_df = best_model_df.reset_index()
        return best_model_df

    @classmethod
    def plotBestLinearFits(cls,
            model_id: Union[str, int],
            min_accuracy: float = cn.DEF_MIN_ACCURACY,
            accuracy_statistic: str = cn.DEF_ACCURACY_STAT,
            is_guard_filename: bool = True,
            ax: Optional[plt.Axes] = None,  # type: ignore
            data_dir: str = cn.PSD_DATA_DIR,
            **kwargs) -> ExtendedPlotOptions:
        """
        Plots the best linear fits for a list of model IDs.

        Args:
            model_id (Union[str, int]): model identifier (either numbers or names).
            min_accuracy (float): The minimum accuracy threshold for predictions to be considered valid.
            accuracy_statistic (str): The statistic to use for accuracy calculations.
            is_guard_filename (bool): Whether to use the guard filename for the data files.
            ax (Optional[plt.Axes]): The axes to plot on.
            data_dir (str): The directory containing the data files.    
            kwargs: Additional keyword arguments to pass to the plotting function.
        """
        analyzer = cls(model_id, min_accuracy=min_accuracy,
                accuracy_stat=accuracy_statistic,
                is_guard_filename=is_guard_filename,
                data_dir=data_dir)
        if analyzer.best_model_ser is None:
            raise ValueError(f"No valid PSD predictions found for model {analyzer.model_name} with min_accuracy {min_accuracy}.")
        psd = analyzer.makeBestPSD()
        if "num_true_point" not in kwargs:
            kwargs["num_true_point"] = 60
        return psd.plotPiecewise(is_changepoint_plot=True, legend=False, ax=ax, is_nochangepoint_plot=False,
                **kwargs)

    @classmethod
    def makeAUCCDFPlots(cls, min_accuracy: float = cn.DEF_MIN_ACCURACY) -> None:
        """Creates 3 CDF plots of AUC values for accuracy statistics min, median, max.

        Args:
            min_accuracy (float): The minimum accuracy threshold for predictions to be considered valid.
        """
        for accuracy_statistic in [cn.COL_MIN, cn.COL_P50, cn.COL_MAX]:
            best_model_df = cls.makeBestPSDModelDF(min_accuracy=min_accuracy,
                    accuracy_stat=accuracy_statistic)
            auc_values = best_model_df[cn.COL_AUC].to_numpy()
            auc_values = np.sort(auc_values)
            fraction_model = np.arange(1, len(auc_values) + 1) / len(auc_values)
            plt.plot(auc_values, fraction_model)
        #
        plt.xlabel("AUC")
        plt.ylabel("fraction of models")
        plt.legend(["Min", "Median", "Max"])

    @classmethod
    def makeLongestSegmentCDFPlot(cls, min_accuracy: float = cn.DEF_MIN_ACCURACY,
            accuracy_stat: str = cn.DEF_ACCURACY_STAT) -> pd.Series:
        """Creates a CDF plot of the longest segment lengths for the best PSD models.

        Args:
            min_accuracy (float): The minimum accuracy threshold for predictions to be considered valid.
            accuracy_stat (str): The statistic to use for accuracy calculations.

        Returns:
            pd.Series: A Series containing the longest segment lengths and their associated fractions.
        """
        best_model_df = cls.makeBestPSDModelDF(min_accuracy=min_accuracy, accuracy_stat=accuracy_stat)
        # Calculate the longest segment from best models
        longest_segments: list = []
        for _, row in best_model_df.iterrows():
            boundaries = row[cn.COL_BOUNDARIES]
            longest_segment = max(np.diff(boundaries))
            longest_segments.append(longest_segment)
        ser = pd.Series(longest_segments, name="Longest Segment Length", index=best_model_df[cn.COL_SYSTEM_ID]).sort_values()
        ser = ser / best_model_df.iloc[0][cn.COL_BOUNDARIES].max()  # Normalize by the maximum boundary value
        # Construct the plot
        fraction_model = np.arange(1, len(longest_segments) + 1) / len(longest_segments)
        plt.plot(longest_segments, fraction_model)
        plt.grid(True)
        plt.xlabel("longest segment (normalized) length")
        plt.ylabel("fraction of models")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        #
        return ser