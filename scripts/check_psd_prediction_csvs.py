#!/usr/bin/env python3
'''Check integrity  of PSD prediction CSV files by verifying that data inside each file is consistent with its filename.

Filenames encode per-file metadata as key-value pairs separated by double underscores, e.g.:

    piecewise_predictions__numpoint_100_a__maxreduction_1e-3.csv
    piecewise_predictions__numpoint_50_b__threshold_0.001__removal_0__maxreduction_0.01.csv

This script parses the encoded portion of each filename and confirms that every row in the CSV
matches those values (within a small floating-point tolerance for numeric columns). Files that
carry no known metadata on their names -- such as aggregated ``piecewise_predictions.csv`` -- are
skipped with a warning rather than failing.
'''

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.constants as cn  # noqa: E402
from util import codedstrToDict, getPSDPredictionDF  # noqa: E402


# Mapping from filename-encoded key to the data column it should match. Only keys with a
# corresponding column are validated; everything else is skipped silently so that files using
# additional metadata keys (e.g. ``manycp``, ``repeat``) don't fail this check until mappings for
# them have been added here.
_FILENAME_KEY_TO_COLUMN: dict[str, str] = {
    "maxreduction": cn.COL_MAX_FRACTIONAL_REDUCTION,  # "max_fractional_reduction"
    "threshold": cn.COL_COEFFICIENT_THRESHOLD,        # "coefficient_threshold"
    "removal": cn.COL_IS_CHANGEPOINT_REMOVAL,          # "is_changepoint_removal"
}


_NUMERIC_TOLERANCE = 1e-9


def _parse_encoded_metadata(filename_without_ext: str) -> dict | None:
    """Parse the encoded metadata portion of a PSD prediction CSV filename.

    Filenames look like::

        piecewise_predictions__numpoint_100_a__maxreduction_1e-3.csv
        piecewise_predictions__numpoint_50_b__threshold_0.001__removal_0__maxreduction_0.01.csv

    Every generated file carries ``maxreduction`` as the first encoded key, so we anchor the parse
    at the first occurrence of ``__maxreduction_`` and run ``codedstrToDict`` on everything after
    that marker (which is itself in ``key_value__key_value...`` format).

    Returns a dict of parsed key/value pairs, or None if no known encoded keys are present.
    """
    idx = filename_without_ext.find("__numpoint_")
    if idx < 0:
        return None
    # ``idx + 2`` skips the leading "__" so we keep "maxreduction" attached to its value, which is
    # what ``codedstrToDict`` expects.
    encoded_part = filename_without_ext[idx + 2:]
    try:
        return codedstrToDict(encoded_part)
    except Exception as exc:  # pragma: no cover - defensive; eval() errors in codedstrToDict are rare
        raise ValueError(
            f"Failed to parse encoded portion '{encoded_part}' of "
            f"filename '{filename_without_ext}': {exc}"
        ) from exc



def _check_file(filepath: Path) -> list[str]:
    """Validate one CSV file. Returns a list of error strings; empty == OK."""
    ##
    def remove(metadata: dict, key: str) -> None:
        """Remove a key from the metadata dict  if it exists, otherwise do nothing."""
        if key in metadata:
            del metadata[key]
    ##

    errors: list[str] = []
    stem = filepath.stem
    metadata = _parse_encoded_metadata(stem)

    if metadata is None:
        return [f"No known encoded keys found in filename '{stem}'; skipping."]
    # Exclude keys that are not validated against the CSV columns. These are either not present in the
    # CSV or are not relevant to the validation.
    remove(metadata, cn.COL_NUM_POINT)
    remove(metadata, cn.COL_MANYCP)
    remove(metadata, cn.COL_REPEAT)

    try:
        df = getPSDPredictionDF(csv_files=[filepath])  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive only
        errors.append(f"Failed to read CSV at {filepath}: {exc}")
        return errors

    for col_name, expected in metadata.items():
        if col_name is None:
            continue

        if col_name not in df.columns:  # pragma: no cover - defensive only
            errors.append(
                f"{filepath.name}: column '{col_name}'"
                f"is missing from the CSV."
            )
            continue

        unique = df[col_name].unique()

        if isinstance(expected, bool):
            expected_set = {int(expected)}
            actual_set = set(int(v) for v in unique)
            if actual_set != expected_set:
                errors.append(
                    f"{filepath.name}: column '{col_name}' is inconsistent with filename "
                    f"key '{cn.COLUMN_METADATA_KEY_2_FILE_METADATA_KEY_DCT[col_name]}'. Expected {expected_set}, got {actual_set}."
                )
        else:
            try:
                expected_float = float(expected)  # handles both str and numeric inputs
            except (TypeError, ValueError):
                if set(unique.tolist()) != {expected}:
                    errors.append(
                        f"{filepath.name}: column '{col_name}' is inconsistent with filename "
                        f"key '{col_name}'. Expected {expected!r}, got {unique.tolist()}."
                    )
                continue
            # Numeric: allow a small floating-point tolerance for scientific-notation round-trips.
            if not all(abs(float(v) - expected_float) < _NUMERIC_TOLERANCE for v in unique):
                errors.append(
                    f"{filepath.name}: column '{col_name}' is inconsistent with filename "
                    f"key '{col_name}'. Expected {expected}, got {unique.tolist()}."
                )

    return errors


def main() -> int:
    data_dir = cn.PSD_DATA_DIR
    if not os.path.isdir(data_dir):  # pragma: no cover - defensive only
        print(f"Data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    csv_files = sorted(Path(data_dir).glob("piecewise_predictions__numpoint*.csv"))
    total = len(csv_files)
    ok_count = 0
    skip_count = 0
    fail_count = 0

    for filepath in csv_files:
        print(f"Checking {filepath.name}...")
        results = _check_file(filepath)
        if not results:
            ok_count += 1
            continue
        first_line = results[0]
        if "No known encoded keys found" in first_line:
            skip_count += 1
            print(f"WARN {filepath.name}")
            for msg in results:
                print(f"  - {msg}")
            continue
        fail_count += 1
        print(f"FAIL {filepath.name}")
        for msg in results:
            print(f"  - {msg}")

    summary = (
        f"\n{ok_count}/{total} files passed; "
        f"{skip_count} skipped (no encoded keys); {fail_count} failed."
    )
    print(summary)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
