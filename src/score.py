"""Calculates the Accuracy score for predictions."""

"""
Accuracy score is a pointwise score for each species and timepoint. At each timepoint, we calculate
the Absolute Percentage Error (APE) between the predicted and true values that is clipped to the range [0, 1].
APE = clip(abs(prediction - true) / true), 0, 1) with sentinel -1 for invalid (zero or non-finite) values.
Accuracy = 1 - APE, with sentinel -1 for invalid (zero or non-finite) values.
"""

from src.plot_options import PlotOptions  # type: ignore
import src.constants as cn  # type: ignore
from src.statistic_calculator import StatisticCalculator # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import List, Optional, Union  # type: ignore
import warnings  # type: ignore

MIN_SIGNIFICANT_VALUE = 1e-5  # Minimum value considered significant for true values; below this, accuracy is undefined.


class Score:
    """Scores prediction timecourses against true timecourses using zero-floor Accuracy.

    Results are accumulated via StatisticAccumulator and persisted to CSV.
    """
    SERIALIZATION_PATH = "score.csv"  # Default path for persistence, can be overridden in subclasses.

    def __init__(self, serialization_path: Optional[str] = None, is_persist: bool = True,
            col_percentile: str = cn.COL_P10) -> None:
        """
        Parameters
        ----------
        serialization_path : str
            Path to a CSV file for persistence.
        is_persist : bool
            Whether to persist the DataFrame to the CSV file.
        col_percentile : str
            The percentile to compute (e.g., 'p10', 'p50', 'p90'). Default is 'p10'.
        """
        self._is_persist = is_persist
        if serialization_path is None:
            serialization_path = self.SERIALIZATION_PATH
        self._serialization_path = serialization_path
        self._col_percentile = col_percentile
        #
        self.score_df = pd.DataFrame()

    @classmethod
    def deserialize(cls, serialization_path: Optional[str]=None) -> 'Score':
        """Loads a previously serialized score DataFrame from CSV.
        Uses a defulat of cn.COL_P10 for the percentile column, but this can be changed later if needed.

        Parameters
        ----------
        serialization_path : str
            Path to the CSV file.

        Returns
        -------
        'Score'
            The deserialized score object.
        """
        score = cls(serialization_path=serialization_path, is_persist=False)
        if serialization_path is not None:  
            score.score_df = pd.read_csv(serialization_path)
        score._col_percentile = cn.COL_P10  # Default to p10; can be changed later if needed.
        return score

    @property
    def serialization_path(self) -> str:
        return self._serialization_path

    @staticmethod 
    def calculateTimeseriesAccuracies(true_df: pd.DataFrame,
            prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes a [0, 1] clipped accuracy score for each species and timepoint. The accuracy is defined as
        Accuracy = 1 - clip(|prediction − true| / |true|, 0, 1).
        If the true value is undefined or zero, Accuracy is set to -1 as a sentinel
        indicating an invalid measurement (these values are excluded from aggregation).

        Parameters
        ----------
        true_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_df : pd.DataFrame
            Prediction timecourse with the same structure as true_df.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the same structure as the input dataframes, containing the Accuracy values.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ape_df = (prediction_df - true_df) / true_df
            ape_df = ape_df.abs()
            # Clip valid values to [0, 1] range first.
            ape_df = ape_df.clip(lower=0, upper=1)
            # Then mark undefined/zero-true values as -1 sentinel AFTER clipping.
            # This must happen after clip so that the -1 sentinel is not converted
            # to 0 by clip(lower=0), which would mask bad predictions from aggregation.
            invalid_mask = (np.isclose(true_df, 0, atol=MIN_SIGNIFICANT_VALUE)) | ~np.isfinite(ape_df)
            accuracy_df = 1 - ape_df
            accuracy_df = accuracy_df.where(~invalid_mask, other=-1.0)
        max_value = accuracy_df.max().max()
        if max_value > 1.0:
            raise ValueError(f"Accuracy values exceeded 1.0; max found: {max_value}")
        return accuracy_df

    @classmethod
    def calculateAccuracyPercentile(cls, true_df: pd.DataFrame, prediction_df: pd.DataFrame,
            col_percentile: str=cn.COL_P10) -> pd.Series:
        """
        Calculates the percentile of Accuracy values for each species across all timepoints.

        Parameters
        ----------
        true_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_df : pd.DataFrame
            Prediction timecourse with the same structure as true_df.
        col_percentile : str
            The percentile to compute (e.g., 'p10', 'p50', 'p90'). Default is 'p10'.

        Returns
        -------
        pd.Series
            A Series containing the Accuracy values for each timepoint and species.
        """
        accuracy_df = cls.calculateTimeseriesAccuracies(true_df, prediction_df)
        calculator = StatisticCalculator()
        for species_name in accuracy_df.columns:
            calculator.add(species_name, accuracy_df[species_name].to_numpy())
        ser = calculator.dataframe[col_percentile]
        ser.index = accuracy_df.columns
        return ser
    
    @classmethod
    def calculateModelAccuracy(cls, true_df: pd.DataFrame, prediction_df: pd.DataFrame,
            col_percentile: str=cn.COL_P10) -> float:
        """
        Calculates the model-level accuracy as the p-th percentile of per-species p-th-percentile accuracies.

        This double-percentile aggregation yields a conservative summary: first, for each species we
        compute the p-th percentile across all timepoints (capturing its best-performing fraction), and
        then we take the p-th percentile across species (capturing the performance level exceeded by that
        fraction of species).

        Parameters
        ----------
        true_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_df : pd.DataFrame
            Prediction timecourse with the same structure as true_df.
        col_percentile : str
            The percentile to compute (e.g., 'p10', 'p50', 'p90'). Default is 'p10'.

        Returns
        -------
        float
            The p-th percentile of per-species p-th-percentile accuracy values.
        """
        accuracy_ser = cls.calculateAccuracyPercentile(true_df, prediction_df, col_percentile)
        percentile = StatisticCalculator.extractPercentile(col_percentile)
        return accuracy_ser.quantile(q=percentile / 100)

    def add(self,
            true_timecourse_df: pd.DataFrame,
            prediction_timecourse_df: pd.DataFrame,
            system_id: str = "",
            ) -> pd.DataFrame:
        """Computes scores and accumulates statistics for model-level and per-species aggregations.

        Parameters
        ----------
        system_id : str
            Descriptive ID stored in each aggregation row.
        true_timecourse_df : pd.DataFrame
            True timecourse with timepoints as index and species as columns.
        prediction_timecourse_df : pd.DataFrame
            Prediction timecourse with the same structure.

        Returns
        -------
        pd.DataFrame
            The full score DataFrame after this addition.
        Raises
        ------
        ValueError
            If ``true_timecourse_df`` and ``prediction_timecourse_df`` have mismatched indexes or columns.
        """
        common_arr = np.array(set(true_timecourse_df.index).intersection(set(prediction_timecourse_df.index)))
        true_timecourse_df = true_timecourse_df.loc[common_arr]
        prediction_timecourse_df = prediction_timecourse_df.loc[common_arr]
        # Validate matching structure up front so misaligned inputs fail loudly rather than producing silent NaNs.
        if not true_timecourse_df.index.equals(prediction_timecourse_df.index):
            raise ValueError(
                f"Input DataFrames must share identical timepoint indexes; "
                f"true has {len(true_timecourse_df.index)} points, prediction has {len(prediction_timecourse_df.index)}.")
        if list(true_timecourse_df.columns) != list(prediction_timecourse_df.columns):
            raise ValueError(
                f"Input DataFrames must share identical species columns in the same order; "
                f"true = {list(true_timecourse_df.columns)}, prediction = {list(prediction_timecourse_df.columns)}.")
        statistic_calculator = StatisticCalculator()
        # Calculate model-level aggregation (across all species and timepoints)
        model_accuracy_ser = self.calculateAccuracyPercentile(true_timecourse_df, prediction_timecourse_df,
                col_percentile=self._col_percentile)
        statistic_calculator.add(cn.COL_AGGREGATION_TYPE_MODEL, model_accuracy_ser.to_numpy())
        # Species level aggregations (one per species column, across all timepoints)
        accuracy_df: pd.DataFrame = self.calculateTimeseriesAccuracies(true_timecourse_df, prediction_timecourse_df)
        species_names = list(accuracy_df.columns)
        for species_name in species_names:
            statistic_calculator.add(species_name, accuracy_df[species_name].to_numpy())
        # Add the system ID
        score_df = statistic_calculator.dataframe.copy()
        score_df[cn.COL_SYSTEM_ID] = system_id
        score_df = score_df.rename(columns={cn.COL_LABEL: cn.COL_AGGREGATION_TYPE})
        columns = [c for c in score_df.columns if c not in {cn.COL_SYSTEM_ID, cn.COL_AGGREGATION_TYPE}]
        score_df = score_df[[cn.COL_SYSTEM_ID, cn.COL_AGGREGATION_TYPE] + columns]
        self.score_df = pd.concat([self.score_df, score_df], ignore_index=True)
        # Serialize the accumulated statistics
        if self._is_persist:
            self.score_df.to_csv(self.serialization_path, index=False)
        return self.score_df

    def plotCDF(self, 
            metric_name: Union[str, List[str]],
            is_plot_model: bool = True,
            is_plot_species: bool = True,
            is_plot: bool = True,
            **plt_kwargs) -> PlotOptions:
        """
        Plots the CDF

        Args:
            metric_name (str | List[str]): Name(s) of the metric(s) to plot.
            is_plot_model (bool, optional): Whether to plot model-level aggregation. Defaults to True.
            is_plot_species (bool, optional): Whether to plot species-level aggregation. Defaults to True.
            is_plot (bool, optional): Whether to display the plot. Defaults to True.
            **plt_kwargs: Additional keyword arguments for PlotOptions.
        """
        if isinstance(metric_name, str):
            metric_names = [metric_name]
        else:
            metric_names = metric_name
        # Get the data
        df = self.score_df
        # Do the plot
        kwargs = dict(plt_kwargs)
        title = ""
        if is_plot_model and is_plot_species:
            title = "CDFs"
        elif is_plot_model:
            title = "CDF for models"
        elif is_plot_species:
            title = "CDF for species"
        else:
            raise ValueError("At least one of is_plot_model or is_plot_species must be True.")
        kwargs.setdefault("title", title)
        kwargs.setdefault("xlabel", "accuracy")
        kwargs.setdefault("ylabel", "fraction")
        #
        plotted_any = False
        missing_metrics = []
        plot_options = PlotOptions(**kwargs)
        legend = []
        if is_plot_model:
            for mname in metric_names:
                model_df = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
                if not model_df.empty and mname in model_df.columns:
                    suffix = "(model)" if is_plot_species else ""
                    legend.append(f"{mname} {suffix}")
                    value_arr = np.array(model_df[mname].values)
                    #doPlot(value_arr)
                    self.plotCDFArray(value_arr, is_plot=is_plot)
                    plotted_any = True
                else:
                    missing_metrics.append(f"model: {mname}")
        if is_plot_species:
                for mname in metric_names:
                    species_df = df[df[cn.COL_AGGREGATION_TYPE] != cn.COL_AGGREGATION_TYPE_MODEL]
                    if not species_df.empty and mname in species_df.columns:
                        suffix = "(species)" if is_plot_model else ""
                        legend.append(f"{mname} {suffix}")
                        value_arr = np.array(species_df[mname].values)
                        self.plotCDFArray(value_arr, is_plot=is_plot)
                        #doPlot(value_arr)
                        plotted_any = True
                    else:
                        missing_metrics.append(f"species: {mname}")
        if len(missing_metrics) > 0:
            warnings.warn(f"No data found for metrics {missing_metrics}; nothing was plotted.", UserWarning)
        if not plotted_any:
            warnings.warn(f"No data found for metric {metric_name}; nothing was plotted.", UserWarning)
        kwargs.setdefault("legend", legend)
        plot_options.legend = legend
        plot_options.apply()
        #
        return plot_options

    @staticmethod
    def plotCDFArray(value_arr: np.ndarray, ax = None, is_plot: bool = True):
        """
        Plots the CDF for a given array of values.

        Args:
            value_arr (np.ndarray): Array of values to plot.
            is_plot (bool): If True, the plot will be displayed; if False, it will be closed.
        """
        if ax is None:
            ax = plt.gca()
        sorted_arr = np.sort(value_arr)
        length = len(sorted_arr)
        yv = np.array(range(length))/length
        ax.grid(True)
        ax.plot(sorted_arr, yv)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("accuracy")
        ax.set_ylabel("fraction")
        if not is_plot:
            plt.close()
        return ax