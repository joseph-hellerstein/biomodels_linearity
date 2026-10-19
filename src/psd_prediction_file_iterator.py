"""Iterator over  piecewise predictions CSV and pickle files with coded metadata in name."""

import os
import pickle
import pandas as pd # type: ignore
from typing import Iterator, Optional, Callable, Union, cast, Dict, Any

from tomlkit import key

import src.constants as cn  # type: ignore
from src import util  # type: ignore
from src.model import Model
import src.util as util

import pandas as pd  # type: ignore
    
REQUIRED_META_DATA = ["manycp", "threshold", "removal", "numpoint", "repeat"]


class PSDPredictionItem:
    """Represents a single piecewise predictions file with its metadata and data."""

    def __init__(
        self,
        filename: str,
        filepath: str,
        max_fractional_reduction: float,
        coefficient_threshold: float,
        is_changepoint_removal: bool,
        max_changepoint: int,
        num_point: int,
        repeat: Optional[int],
        manycp: bool,
        df: pd.DataFrame
    ) -> None:
        """
        Initialize a PSDPredictionsItem.

        Args:
            filename: Name of the prediction file.
            filepath: Path to the prediction file.
            max_fractional_reduction: Maximum fractional reduction parameter from filename.
            coefficient_threshold: Coefficient threshold parameter from filename.
            is_changepoint_removal: Whether changepoint removal was enabled.
            max_changepoint: Maximum number of change points.
            num_point: Number of time points.
            repeat: Repeat index (if present in filename).
            manycp: Whether many changepoints mode was used.
            df: DataFrame containing the prediction data.
        """
        self.max_fractional_reduction = max_fractional_reduction
        self.coefficient_threshold = coefficient_threshold
        self.is_changepoint_removal = is_changepoint_removal
        self.max_changepoint = max_changepoint
        self.num_point = num_point
        self.repeat = repeat
        self.manycp = manycp
        self.df = df
        self.filepath = filepath
        self.file_dir = os.path.dirname(filepath)
        self.filename = os.path.splitext(filename)[0]
        self.cvs_filepath = self.filename + ".csv"
        self.pkl_filepath = self.filename + ".pkl"

    @property
    def pkl_path(self) -> str:
        """Return the full path to the pickle file."""
        return os.path.join(self.file_dir, self.pkl_filepath)

    @property
    def csv_path(self) -> str:
        """Return the full path to the CSV file."""
        return os.path.join(self.file_dir, self.cvs_filepath)


class PSDPredictionFileIterator:
    """Iterates piecewise predictions CSV and pickle files in cn.PSD_DATA_DIR."""

    def __init__(self, is_pkl: bool = False,
            data_dir: str = cn.PSD_DATA_DIR,
            is_guard_filename: bool = False) -> None:
        """
        Initialize the iterator.

        Args:
            is_pkl: If True, iterate over pickle files. If False, iterate over CSV files.
            data_dir: Directory to iterate over.:w
            is_guard_filename: If True, guard against filenames that don't match the expected
                pattern. If False, skip the guard and allow any filename to be processed.
        """
        self.is_pkl = is_pkl
        self.data_dir = data_dir
        self.is_guard_filenames = is_guard_filename

    def _load_dataframe(self, filepath: str) -> pd.DataFrame:
        """
        Load a DataFrame from either a CSV or pickle file.

        Args:
            filepath: Path to the file.

        Returns:
            Loaded DataFrame.
        """
        if self.is_pkl:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            return pd.read_csv(filepath)

    def _validateFileMetatdata(self, filename: str) -> bool:
        """
        Validate that the filename contains all required metadata keys.

        Args:
            filename: The filename to validate.

        Returns:
            True if the filename contains all required metadata keys, False otherwise.
        """
        metadata = util.parseFilename(filename)
        if metadata is None:
            return False
        if not metadata.get(cn.COL_NUM_POINT) in [1000, 10000, 100000]:
            return False
        return True

    def __iter__(self) -> Iterator[PSDPredictionItem]:
        """
        Iterate over piecewise predictions files.

        Yields:
            PSDPredictionItem: Contains filename, metadata values, and DataFrame.
        """
        if not os.path.isdir(self.data_dir):
            return
        

        # Find all relevant files
        ffiles = sorted([f for f in os.listdir(self.data_dir)])
        ffiles = [f for f in ffiles if "piecewise_predictions" in f]
        ffiles = [f for f in ffiles if "__numpoint_" in f]
        ffiles = [f for f in ffiles if not "dummy" in f]
        ffiles = [f for f in ffiles if (self.is_pkl and f.endswith(".pkl"))
                or ((not self.is_pkl) and f.endswith(".csv"))]

        for filename in ffiles:
            if self.is_guard_filenames and not self._validateFileMetatdata(filename):
                continue
            # Get the full path to the file
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.isfile(filepath):
                continue
            
            # Parse metadata from filename
            metadata = util.parseFilename(filename)
            if metadata is None:
                continue
            
            try:
                if self.is_pkl:
                    with open(filepath, "rb") as f:
                        df = pickle.load(f)
                else:
                    df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
            
            # Extract metadata values for the constructor
            yield PSDPredictionItem(
                filename=filename,
                filepath=filepath,
                max_fractional_reduction=metadata.get(cn.COL_MAX_FRACTIONAL_REDUCTION, None),  # type: ignore
                coefficient_threshold=metadata.get(cn.COL_COEFFICIENT_THRESHOLD, None),  # type: ignore
                is_changepoint_removal=bool(metadata.get(cn.COL_IS_CHANGEPOINT_REMOVAL, None)),  # type: ignore
                max_changepoint=metadata.get(cn.COL_MAX_CHANGEPOINT, None),  # type: ignore
                num_point=metadata.get(cn.COL_NUM_POINT, None),  # type: ignore
                repeat=metadata.get(cn.COL_REPEAT, None),  # type: ignore
                manycp=bool(metadata.get(cn.COL_MANYCP, None)),  # type: ignore
                df=df,
            )

    @classmethod
    def query(cls,
            file_filter: Optional[Callable[[dict], bool]] = None,
            dataframe_filter: Optional[Callable[[pd.DataFrame, dict], pd.DataFrame]] = None,
            is_pkl: bool = True,
            is_model: bool = True,
            data_dir: str = cn.PSD_DATA_DIR,
            is_guard_filename: bool = True,
            ) -> pd.DataFrame:
        """
        Query the piecewise predictions files based on filters for the files and their DataFrames.
        Adds a column to the result for the filename of the source file for each row.

        Args:
            file_filter (Callable[[dict], bool]): Filter files processed
                signature:  args: filename ; returns: True if it should be included in the query.
                The dictionary contains the parsed metadata from the filename.
            data_frame_filter (Callable[[pd.DataFrame, dict], pd.DataFrame]): filter of file
                signature: args: A DataFrame; returns: A filtered DataFrame.
            is_pkl (bool): If True, query pickle files; if False, query CSV files.
            is_model (bool): If True, only include files with model-level aggregation.
            data_dir (str): Directory to search for files.
            is_guard_filename (bool): If True, guard against filenames that don't match the expected
                pattern. If False, skip the guard and allow any filename to be processed.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated results of the filtered results
        """
        # Initializations
        if file_filter is None:
            # Default filter includes all files
            file_filter = lambda filename: True
        if dataframe_filter is None:
            # Default filter includes all DataFrames
            dataframe_filter = lambda df, _: df
        # Iterate across files
        dfs = []
        psd_iterator = PSDPredictionFileIterator(is_pkl=is_pkl, data_dir=data_dir,
                is_guard_filename=is_guard_filename)
        for item in psd_iterator:
            dct = util.codedstrToDict(item.filename)
            if not file_filter(dct):
                continue
            df = item.df.copy()
            if is_model:
                # Filter for model-level aggregation
                df = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
            filtered_df = dataframe_filter(df, dct)
            filtered_df[cn.COL_FILENAME] = item.filename  # Add the source filename
            dfs.append(filtered_df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs)