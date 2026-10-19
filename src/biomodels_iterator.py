"""
Module for iterating over BioModels SBML model directories.
"""
import src.constants as cn  # type: ignore

import os
import pandas as pd  # type: ignore
from typing import Iterator, List, Optional, Tuple


def getBiomodelsEndtimes(endtimes_csv_path: str = cn.CALCULATED_ENTIMES_PATH,
        is_include_endtime_source: bool = False) -> dict:
    """
    Load a mapping of BioModels IDs to end times from a CSV file. Adjusts
    the end times based on the source of the end time (e.g., steadystate or max_median_cv) using predefined fractions.

    Parameters
    ----------
    endtimes_csv_path : str
        Path to the CSV file with columns cn.COL_MODEL_NAME and cn.COL_ENDTIME.
    is_include_endtime_source : bool
        Whether to include the end time source in the returned dictionary.

    Returns
    -------
    dict
        Mapping of BioModels IDs (str) to end times (float). Empty dict if
        the file is absent or missing required columns.
    """
    result_dct: dict = {}
    if os.path.exists(endtimes_csv_path):
        df = pd.read_csv(endtimes_csv_path)
        if not cn.COL_ENDTIME in df.columns or not cn.COL_MODEL_NAME in df.columns:
            raise ValueError(f"CSV file {endtimes_csv_path} must contain columns '{cn.COL_MODEL_NAME}' and '{cn.COL_ENDTIME}'")
        if len(df) == 0:
            return result_dct
        if cn.COL_ENDTIME_SOURCE not in df.columns:
            df[cn.COL_ENDTIME_SOURCE] = cn.ENDTIME_SOURCE_USER_SPECIFIED
        sel = df[cn.COL_ENDTIME_SOURCE] == cn.ENDTIME_SOURCE_STEADYSTATE
        df.loc[sel, cn.COL_ENDTIME] = df.loc[sel, cn.COL_ENDTIME] * cn.ENDTIME_FRACTION_STEADYSTATE
        sel = df[cn.COL_ENDTIME_SOURCE] == cn.ENDTIME_SOURCE_MAX_MEDIAN_CV
        df.loc[sel, cn.COL_ENDTIME] = df.loc[sel, cn.COL_ENDTIME] * cn.ENDTIME_FRACTION_MAXMEDIAN
        if cn.COL_MODEL_NAME in df.columns and cn.COL_ENDTIME in df.columns:
            result_dct = dict(zip(df[cn.COL_MODEL_NAME], df[cn.COL_ENDTIME]))
        if is_include_endtime_source and cn.COL_ENDTIME_SOURCE in df.columns:
            source_dct = dict(zip(df[cn.COL_MODEL_NAME], df[cn.COL_ENDTIME_SOURCE]))
            result_dct = {k: (v, source_dct[k]) for k, v in result_dct.items()}
    return result_dct

############################################
class BiomodelsItem:
    """Represents a single BioModel with its associated file paths."""

    def __init__(self,
            model_name: str,
            sbml_paths: List[str],
            sedml_paths: List[str],
            existing_df: pd.DataFrame = pd.DataFrame(),
            end_time: Optional[float] = None,
            endtime_source: Optional[str] = None) -> None:
        """
        Initialize a BiomodelsItem.

        Parameters
        ----------
        model_name : str
            The BioModel identifier (e.g. 'BIOMD0000000001').
        sbml_paths : List[str]
            Absolute paths to SBML (.xml) files in the model directory,
            excluding manifest.xml.
        sedml_paths : List[str]
            Absolute paths to SED-ML (.sedml) files in the model directory.
        existing_df : pd.DataFrame
            The DataFrame containing existing processed models.
        end_time : Optional[float]
            The simulation end time for this model, or None if unknown.
        endtime_source : Optional[str]
            The source of the end time, or None if unknown.
        """
        self.model_name = model_name
        self.sbml_paths = sbml_paths
        self.sedml_paths = sedml_paths
        self.end_time = end_time
        self.endtime_source = endtime_source
        self.existing_df = pd.DataFrame()
        self.model_num = self.getModelNumber()
        if existing_df is not None:
            self.existing_df = existing_df

    def __repr__(self) -> str:
        return (
            f"BiomodelsItem(model_name={self.model_name!r}, "
            f"sbml_paths={self.sbml_paths!r}, "
            f"sedml_paths={self.sedml_paths!r}, "
            f"end_time={self.end_time!r}, "
            f"model_num={self.model_num}, "
            f"existing_df={self.existing_df!r}"
        )
    
    def getModelNumber(self) -> int:
        """Extracts the numeric part of a model name like 'BIOMD0000000001'."""
        try:
            return int(self.model_name.replace("BIOMD", ""))
        except ValueError:
            return -1  # Return -1 for unexpected model name formats


############################################
class BiomodelsIterator:
    """Iterates over all BioModel directories in the BioModels repository."""

    def __init__(self,
                biomodels_dir: str = cn.BIOMODELS_DIR,
                excluded_models: List[str] = [],
                existing_csv_path: Optional[str] = None,
                is_report: bool = True,
                first_model_num: int = 0,
                last_model_num: int = int(1e9),
                endtimes_csv_path: Optional[str] = None
                ) -> None:
        """
        Initialize a BiomodelsIterator.

        Parameters
        ----------
        biomodels_dir : str
            Path to the directory containing BioModel subdirectories.
            Defaults to cn.BIOMODELS_DIR.
        excluded_models : List[str]
            List of model names to exclude from iteration.
        is_report : bool
            Whether to print progress reports.
        existing_csv_path : Optional[str]
            Path to an existing CSV file containing processed models. If provided,
            models listed in this file will be added to the excluded_models list.
            The column cn.COL_MODEL_NAME will be used to identify processed models.
        first_model_num : int
            The first model number to include (inclusive).
        last_model_num : int
            The last model number to include (inclusive).
        endtimes_csv_path : Optional[str]
            Path to the endtimes CSV file. If None, uses cn.CALCULATED_ENTIMES_PATH.
        """
        self.biomodels_dir = biomodels_dir
        self.excluded_models = excluded_models
        self._is_report = is_report
        self._existing_csv_path = existing_csv_path
        self._existing_df, self._processed_models = self._getProcessedModelsFromCSV()
        self.first_model_num = first_model_num
        self.last_model_num = last_model_num
        if endtimes_csv_path is not None:
            self._endtime_dct = getBiomodelsEndtimes(endtimes_csv_path=endtimes_csv_path,
                    is_include_endtime_source=True)
        else:
            # When no path provided, load without source info so all models pass the filter
            self._endtime_dct = getBiomodelsEndtimes(is_include_endtime_source=True)

    def _getProcessedModelsFromCSV(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Get a list of processed model names from an existing CSV file.

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            A tuple containing the DataFrame read from the CSV file and a list of model names that have already been processed.
            If no existing CSV path is provided or the file does not exist, returns an empty DataFrame and an empty list.
        """
        if self._existing_csv_path is None or not os.path.exists(self._existing_csv_path):
            return pd.DataFrame(), []
        df = pd.read_csv(self._existing_csv_path)
        if not cn.COL_MODEL_NAME in df.columns:
            raise ValueError(f"Expected column '{cn.COL_MODEL_NAME}' not found in existing CSV file: {self._existing_csv_path}")
        processed_models = df[cn.COL_MODEL_NAME].tolist()
        return df, processed_models

    @classmethod
    def _findFilesWithExtension(cls, model_dir: str, extension: str) -> List[str]:
        """
        Find files with a given extension in a model directory, excluding manifest.xml.

        Parameters
        ----------
        model_dir : str
            Absolute path to the model directory.
        extension : str
            File extension to search for (e.g. '.xml', '.sedml').

        Returns
        -------
        List[str]
            Sorted list of absolute file paths matching the extension.
        """
        paths = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(extension) and f != "manifest.xml"
        ]
        return sorted(paths)
    
    def _msg(self, text: str) -> None:
        if self._is_report:
            print(text)

    @classmethod
    def getBiomodelInfo(cls, model_dir: str) -> BiomodelsItem:
        model_name = os.path.basename(model_dir)
        sbml_paths = cls._findFilesWithExtension(model_dir, ".xml")
        sedml_paths = cls._findFilesWithExtension(model_dir, ".sedml")
        return BiomodelsItem(
            model_name=model_name,
            sbml_paths=sbml_paths,
            sedml_paths=sedml_paths,
            existing_df=pd.DataFrame()
        )
    
    @classmethod
    def extractModelNum(cls, model_name: str) -> int:
        """Extracts the numeric part of a model name like 'BIOMD0000000001'."""
        try:
            return int(model_name.replace("BIOMD", ""))
        except ValueError:
            return -1  # Return -1 for unexpected model name formats

    def __iter__(self) -> Iterator[BiomodelsItem]:
        """
        Yield a BiomodelsItem for each BioModel directory if its endtime is from SED-ML.

        Yields
        ------
        BiomodelsItem
            Item containing the model name and paths to its SBML and SED-ML files.
        """
        model_names = sorted(
            d for d in os.listdir(self.biomodels_dir)
            if os.path.isdir(os.path.join(self.biomodels_dir, d)) 
            and "BIOMD" in d
        )
        endtime_dct = self._endtime_dct
        is_reported_too_low = False
        is_reported_too_high = False
        for model_name in model_names:
            model_num = self.extractModelNum(model_name)
            # Skip models whose source is explicitly NOT sedml, but yield those absent from dict
            if (model_name in endtime_dct):
                entry = endtime_dct[model_name]
                if isinstance(entry, tuple) and len(entry) == 2:
                    if entry[1] != cn.ENDTIME_SOURCE_SEDML:
                        continue
            if model_num < self.first_model_num or model_num > self.last_model_num:
                if not is_reported_too_low and model_num < self.first_model_num:
                    self._msg(f"Model {model_name} has number {model_num} which is below the first model number {self.first_model_num}")
                    is_reported_too_low = True
                if not is_reported_too_high and model_num > self.last_model_num:
                    self._msg(f"Model {model_name} has number {model_num} which is above the last model number {self.last_model_num}")
                    is_reported_too_high = True
                continue
            model_dir = os.path.join(self.biomodels_dir, model_name)
            if model_name in self._processed_models:
                self._msg(f"Skipping processed model: {model_name}")
                continue
            if model_name in self.excluded_models:
                self._msg(f"Skipping excluded model: {model_name}")
                continue
            self._msg(f"Processing model: {model_name}")
            item = self.getBiomodelInfo(model_dir)
            item.existing_df = self._existing_df
            # Extract end time value (handle both plain values and tuples)
            entry = self._endtime_dct.get(model_name)
            if isinstance(entry, tuple):
                item.end_time = entry[0]
                item.endtime_source = entry[1]
            else:
                item.end_time = entry
                item.endtime_source = None
            yield item