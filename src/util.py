'''Utilities'''

from traitlets import Any

import src.constants as cn

import ast  # type: ignore
import os
import pickle
import pandas as pd  # type: ignore
from pathlib import Path
from typing import Optional, List, Union, cast
import re


PATTERN = "piecewise_predictions__numpoint*.csv"
_NP_INT64_PATTERN = re.compile(r'np\.int64\(([^)]*)\)')


def makeCSVPaths(max_fractional_reduction: float, repeat: Optional[int] = None) -> List[str]:
    """Provide a list of CSV paths corresponding to the arguments.

    Args:
        max_fractional_reduction (float): _description_
        repeat (int, optional): Which repetition file to use. If None, then all repetitions are used. Defaults to None.

    Raises:
        FileNotFoundError: If no file is found

    Returns:
        List[Path]
    """
    file_selection1_str = dictToCodedstr(
        {"maxreduction": max_fractional_reduction},
        convert_strs=["maxreduction"],
    )
    if repeat is not None:
        file_selection2_str = dictToCodedstr(
            {"repeat": repeat},
        )
    else:
        file_selection2_str = None
    sections = [file_selection1_str]
    if isinstance(file_selection2_str, str):
        sections.append(file_selection2_str)
    csv_files: List[str] = [f for f in os.listdir(cn.PSD_DATA_DIR) if f.endswith(".csv")]
    for file_selection_str in sections:
        csv_files = [f for f in csv_files if file_selection_str in f]
    results = [os.path.join(cn.PSD_DATA_DIR, f) for f in csv_files]
    return results

def parseFilename(filename: str) -> dict:
    """Parse a piecewise predictions filename into its metadata components.

    Args:
        filename (str): Filename to parse.

    Returns:
        dict: Dictionary containing the parsed metadata.
    """
    KEWORD = "piecewise_predictions__"
    if not filename.startswith(KEWORD):
        raise ValueError(f"Filename {filename} does not start with expected prefix '{KEWORD}'.")
    codedstr = filename.split("piecewise_predictions__")[-1]
    return codedstrToDict(codedstr)

def getPSDPredictionDF(
            max_fractional_reduction: Optional[float] = None,
            repeat: Optional[int] = None,
            csv_files: Optional[List[Union[Path, str]]] = None,
            is_model: bool = True,
            ) -> pd.DataFrame:
    """Get a DataFrame from piecewise prediction CSVs.
    This can be done in two ways: either by providing a list of CSV files,
        or by filtering by max_fractional_reduction and repeat. One of the arguments
        csv_files or max_fractional_reduction must be provided.

    Args:
        max_fractional_reduction (float): _description_
        csv_files (Optional[List[Union[Path, str]]], optional): List of CSV files to read. If None, then max_fractional_reduction and repeat are used to filter the files.
            Defaults to None.
        repeat (int, optional): Which repetition file to use. If None, then all repetitions are used. Defaults to None.
        is_model (bool, optional): Whether to filter for model-level aggregations. Defaults to True.

    Raises:
        FileNotFoundError: If no file is found

    Returns:
        pd.DataFrame
    """
    if (csv_files is None) and (max_fractional_reduction is None):
        raise ValueError("Either csv_files or max_fractional_reduction must be provided.")
    if (csv_files is not None) and (max_fractional_reduction is not None):
        raise ValueError("Ambiguous: both csv_files and max_fractional_reduction are provided. Please provide only one.")
    #
    if csv_files is None:
        csv_files = []
        csv_files = makeCSVPaths(max_fractional_reduction=max_fractional_reduction,  # type: ignore
                repeat=repeat)
    if len(csv_files) == 0: # type: ignore
        raise FileNotFoundError(f"No CSV files found for max_fractional_reduction={max_fractional_reduction} and repeat={repeat}.") 
    # Get the data
    frames: list[pd.DataFrame] = []
    for path in cast(List[Union[Path, str]], csv_files):
        splits = str(path).split("/")
        filename = splits[-1].split(".csv")[0]
        csv_dir = os.path.dirname(str(path))
        pkl_path = os.path.join(csv_dir, f"{filename}.pkl")
        dct = parseFilename(filename)
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                df = pickle.load(f)
        else:
            df = pd.read_csv(path)
            df[cn.COL_CHANGEPOINTS] = df.apply(lambda row: 
                    _parseChangepoints(row[cn.COL_CHANGEPOINTS]), axis=1)
            try:
                df[cn.COL_BOUNDARIES] = df.apply(lambda row: [0]
                    + row[cn.COL_CHANGEPOINTS] + [dct[cn.COL_NUM_POINT]-1], axis=1)
            except Exception as e:
                import pdb; pdb.set_trace()
            with open(pkl_path, "wb") as f:
                pickle.dump(df, f)
        df[cn.COL_CSV_FILE] = str(path)
        frames.append(df)
    merged_df = pd.concat(frames, ignore_index=True)
    if is_model:
        merged_df = merged_df[merged_df["aggregation_type"] == "model"].reset_index(drop=True)
    # Check result
    if max_fractional_reduction is not None:
        sel = merged_df["max_fractional_reduction"] == float(max_fractional_reduction)
        if not sel.all():
            raise ValueError(
                f"Some rows in the merged DataFrame do not match max_fractional_reduction={max_fractional_reduction}"
            )
    #
    return merged_df

def codedstrToDict(codedstr: str, is_passthru: bool = False) -> dict:
    """Convert a coded string to a dictionary.

    Args:
        codedstr (str): Coded string in the format "key1_value1__key2_value2"
        is_passthru (bool): If True, the function will not raise an error if a key is not found in the metadata dictionary.

    Returns:
        dict: Dictionary with keys and values extracted from the coded string. Keys are the
        internal names of columns as specified in cn.FILE_METADATA_DCT.
    """
    ##
    def get(key: str) -> str:
        """Helper function to get a value from the metadata dictionary. Does passthru if is_passthru is True."""
        result = cn.COLUMN_METADATA_KEY_2_FILE_METADATA_KEY_DCT.get(key, None)
        if result is None:
            if is_passthru:
                return key
            else:
                raise ValueError(f"Key '{key}' not found in cn.COLUMN_METADATA_KEY_2_FILE_METADATA_KEY_DCT.")
        return result
    ##
    result = {}
    new_codedstr = codedstr.replace("piecewise_predictions__", "")
    for part in new_codedstr.split("__"):
        if "_" in part:
            key, value = part.split("_", 1)
            is_pkl = True if ".pkl" in value else False
            is_csv = True if ".csv" in value else False
            if is_pkl or is_csv:
                value = value[:-4]  # Remove the file extension
            if not is_passthru:
                key = get(key)
            try:
                result[key] = eval(value)
            except:
                result[key] = value
        else:
            raise ValueError(f"Segment '{part}' does not contain an underscore and cannot be parsed.")
    return result

def deprecatedCodedstrToDict(codedstr: str) -> dict:
    """Convert a coded string to a dictionary.

    Args:
        codedstr (str): Coded string in the format "key1_value1__key2_value2"

    Returns:
        dict: Dictionary with keys and values extracted from the coded string. Keys are the
        internal names of columns.
    """
    result = {}
    new_codedstr = codedstr.replace("piecewise_predictions__", "")
    for part in new_codedstr.split("__"):
        if "_" in part:
            key, value = part.split("_", 1)
            try:
                result[key] = eval(value)
            except:
                result[key] = value
    return result

def dictToCodedstr(dct: dict, convert_strs : Optional[List[str]] = None) -> str:
    """Convert a dictionary to a coded string. Optionally converts specified keys to scientific notation.
    Handles the differences in keys between the metadata dictionary and the internal column names.

    Args:
        dct (dict): Dictionary to convert.
        convert_strs (Optional[List[str]]): List of keys to convert to scientific notation.

    Returns:
        str: Coded string in the format "key1_value1__key2_value2"
    """
    dct = dict(dct) # Do not mutate the input dictionary
    ##
    def get(key: str) -> str:
        """Helper function to get a value from the metadata dictionary."""
        return cn.FILE_METADATA_KEY_2_COLUMN_METADATA_KEY_DCT.get(key, key)  # Use the original key if not found in the metadata dictionary
    ##
    if convert_strs is not None:
        for key in convert_strs:
            if key in dct:
                value = dct[key]
                new_value = f"{value:.1e}".replace(".0e-0", "e-")  # Convert to scientific notation and remove unnecessary .0
                dct[key] = new_value
    dct = {get(k): v for k, v in dct.items()}  # Convert keys to metadata dictionary keys
    stg = "__".join(f"{get(k)}_{v}" for k, v in dct.items())
    return stg

def deprecatedDictToCodedstr(dct: dict, convert_strs : Optional[List[str]] = None) -> str:
    """Convert a dictionary to a coded string. Optionally converts specified keys to scientific notation.

    Args:
        dct (dict): Dictionary to convert.
        convert_strs (Optional[List[str]]): List of keys to convert to scientific notation.

    Returns:
        str: Coded string in the format "key1_value1__key2_value2"
    """
    dct = dict(dct) # Do not mutate the input dictionary
    if convert_strs is not None:
        for key in convert_strs:
            if key in dct:
                value = dct[key]
                if key in convert_strs:
                    new_value = f"{value:.1e}".replace(".0e-0", "e-")  # Convert to scientific notation and remove unnecessary .0
                    dct[key] = new_value
    stg = "__".join(f"{k}_{v}" for k, v in dct.items())
    return stg

def _parseChangepoints(value) -> List[int]:
    """Parse a changepoints cell value into a list of numbers.

    Handles both Python lists (already parsed from pickle/JSON) and string
    representations like "[0.0, 2.5]" that come from CSV round-trips via pandas.

    Args:
        value: A Python list or a string representation of one.

    Returns:
        List of int/float values suitable for SegmentAnalyzer.__init__.

    Raises:
        ValueError: If the value cannot be converted to a list of numbers.
    """
    if isinstance(value, str):
        candidates = [value]
        normalized = _NP_INT64_PATTERN.sub(r'\1', value)
        if normalized != value:
            candidates.append(normalized)
        last_err: Optional[Exception] = None
        for s in candidates:
            try:
                return cast(List[int], ast.literal_eval(s))
            except (ValueError, SyntaxError) as e:  # noqa: PERF203
                last_err = e
        raise ValueError(
            f"Cannot parse changepoints string '{value}': {last_err!r}"
        ) from last_err
    if isinstance(value, list):
        if all(isinstance(x, int) for x in value):
            return value
    raise ValueError(
        f"Expected changepoints to be a list or string, got "
        f"{type(value).__name__}: {value!r}")