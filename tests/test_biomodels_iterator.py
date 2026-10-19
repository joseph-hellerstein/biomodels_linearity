"""
Tests for BiomodelsItem and BiomodelsIterator.
"""
import csv
import os
import tempfile
import unittest
from typing import List

import pandas as pd # type: ignore
import src.constants as cn
from biomodels_iterator import BiomodelsItem, BiomodelsIterator, getBiomodelsEndtimes  # type: ignore

IGNORE_TESTS = False
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)
TEST_MODEL = "BIOMD0000000005"
TEST_MODEL_NUM = 5


def _make_model_dir(parent: str, model_name: str, xml_files: List[str],
                    sedml_files: List[str], add_manifest: bool = True) -> str:
    """Create a fake model directory under *parent* and return its path."""
    model_dir = os.path.join(parent, model_name)
    os.makedirs(model_dir, exist_ok=True)
    for fname in xml_files:
        open(os.path.join(model_dir, fname), "w").close()
    for fname in sedml_files:
        open(os.path.join(model_dir, fname), "w").close()
    if add_manifest:
        open(os.path.join(model_dir, "manifest.xml"), "w").close()
    return model_dir


def _write_endtimes_csv(parent: str, entries: List[tuple]) -> str:
    """Write an endtimes CSV with sedml source for each (model_name, end_time) entry.

    Returns the path to the written CSV file.
    """
    csv_path = os.path.join(parent, "endtimes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([cn.COL_MODEL_NAME, cn.COL_ENDTIME, cn.COL_ENDTIME_SOURCE])
        for model_name, end_time in entries:
            writer.writerow([model_name, str(end_time), cn.ENDTIME_SOURCE_SEDML])
    return csv_path


class TestBiomodelsItem(unittest.TestCase):
    """Tests for BiomodelsItem."""

    def test_attributes_stored(self) -> None:
        """Constructor stores model_name, sbml_paths, and sedml_paths."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem(
            model_name="BIOMD0000000005",
            sbml_paths=["/a/b.xml"],
            sedml_paths=["/a/b.sedml"],
        )
        self.assertEqual(item.model_name, "BIOMD0000000005")
        self.assertEqual(item.sbml_paths, ["/a/b.xml"])
        self.assertEqual(item.sedml_paths, ["/a/b.sedml"])

    def test_empty_paths(self) -> None:
        """Constructor accepts empty path lists."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem(model_name="BIOMD0000000002", sbml_paths=[], sedml_paths=[])
        self.assertEqual(item.sbml_paths, [])
        self.assertEqual(item.sedml_paths, [])

    def test_repr_contains_model_name(self) -> None:
        """repr includes the model name."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem(model_name="BIOMD0000000003", sbml_paths=[], sedml_paths=[])
        self.assertIn("BIOMD0000000003", repr(item))

    def test_repr_contains_paths(self) -> None:
        """repr includes sbml_paths and sedml_paths."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem(
            model_name="BIOMD0000000004",
            sbml_paths=["/x.xml"],
            sedml_paths=["/x.sedml"],
        )
        self.assertIn("/x.xml", repr(item))
        self.assertIn("/x.sedml", repr(item))

    def test_existing_df_stored(self) -> None:
        """Constructor stores existing_df when provided."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame({cn.COL_MODEL_NAME: ["BIOMD0000000005"]})
        item = BiomodelsItem(
            model_name="BIOMD0000000005",
            sbml_paths=[],
            sedml_paths=[],
            existing_df=df,
        )
        self.assertIsNotNone(item.existing_df)
        self.assertEqual(len(item.existing_df), 1)

    def test_existing_df_defaults_to_none(self) -> None:
        """existing_df defaults to None when not provided."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem(model_name="BIOMD0000000005", sbml_paths=[], sedml_paths=[])
        self.assertEqual(len(item.existing_df), 0)


class TestBiomodelsItemModelNum(unittest.TestCase):
    """Tests for BiomodelsItem.model_num and getModelNumber."""

    def test_standard_model_name(self) -> None:
        """model_num is the integer suffix of a standard BIOMD name."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem("BIOMD0000000005", [], [])
        self.assertEqual(item.model_num, 5)

    def test_large_model_number(self) -> None:
        """model_num is correct for a larger model number."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem("BIOMD0000001234", [], [])
        self.assertEqual(item.model_num, 1234)

    def test_invalid_name_returns_minus_one(self) -> None:
        """Non-BIOMD name returns model_num of -1."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem("NOT_A_MODEL", [], [])
        self.assertEqual(item.model_num, -1)

    def test_biomd_with_no_digits_returns_minus_one(self) -> None:
        """'BIOMD' with no trailing digits returns model_num of -1."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem("BIOMD", [], [])
        self.assertEqual(item.model_num, -1)

    def test_iterator_item_model_num_matches_name(self) -> None:
        """Items yielded by the iterator have model_num matching their model_name."""
        if IGNORE_TESTS:
            return
        tmpdir = tempfile.mkdtemp()
        try:
            _make_model_dir(tmpdir, "BIOMD0000000007", xml_files=["model.xml"], sedml_files=[])
            csv_path = _write_endtimes_csv(tmpdir, [("BIOMD0000000007", 10.0)])
            it = BiomodelsIterator(biomodels_dir=tmpdir, is_report=False, endtimes_csv_path=csv_path)
            items = list(it)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].model_num, 7)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGetProcessedModelsFromCSV(unittest.TestCase):
    """Tests for BiomodelsIterator._getProcessedModelsFromCSV."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _iterator(self, csv_path: str) -> BiomodelsIterator:
        return BiomodelsIterator(biomodels_dir=self._tmpdir, existing_csv_path=csv_path,
                                 is_report=False)

    def test_no_path_returns_empty(self) -> None:
        """No existing_csv_path returns empty DataFrame and empty list."""
        if IGNORE_TESTS:
            return
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False)
        df, models = it._getProcessedModelsFromCSV()
        self.assertTrue(df.empty)
        self.assertEqual(models, [])

    def test_nonexistent_path_returns_empty(self) -> None:
        """Non-existent file path returns empty DataFrame and empty list."""
        if IGNORE_TESTS:
            return
        it = BiomodelsIterator(biomodels_dir=self._tmpdir,
                               existing_csv_path="/nonexistent/path.csv", is_report=False)
        df, models = it._getProcessedModelsFromCSV()
        self.assertTrue(df.empty)
        self.assertEqual(models, [])

    def test_valid_csv_returns_model_names(self) -> None:
        """Valid CSV with model_name column returns the expected model names."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "existing.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME])
            writer.writerow(["BIOMD0000000005"])
            writer.writerow(["BIOMD0000000002"])
        it = self._iterator(csv_path)
        _, models = it._getProcessedModelsFromCSV()
        self.assertIn("BIOMD0000000005", models)
        self.assertIn("BIOMD0000000002", models)

    def test_valid_csv_returns_dataframe(self) -> None:
        """Valid CSV returns a non-empty DataFrame."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "existing.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME])
            writer.writerow(["BIOMD0000000005"])
        it = self._iterator(csv_path)
        df, _ = it._getProcessedModelsFromCSV()
        self.assertFalse(df.empty)
        self.assertIn(cn.COL_MODEL_NAME, df.columns)

    def test_missing_column_raises_value_error(self) -> None:
        """CSV missing the model_name column raises ValueError on construction."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "bad.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_column"])
            writer.writerow(["BIOMD0000000005"])
        with self.assertRaises(ValueError):
            BiomodelsIterator(biomodels_dir=self._tmpdir,
                              existing_csv_path=csv_path, is_report=False)


class TestFindFilesWithExtension(unittest.TestCase):
    """Tests for BiomodelsIterator._findFilesWithExtension using a temp directory."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _iterator(self) -> BiomodelsIterator:
        return BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=IGNORE_TESTS)

    def test_returns_list(self) -> None:
        """Returns a list for a directory with matching files."""
        if IGNORE_TESTS:
            return
        model_dir = _make_model_dir(self._tmpdir, "BIOMD0000000005",
                                    xml_files=["model.xml"], sedml_files=[])
        result = self._iterator()._findFilesWithExtension(model_dir, ".xml")
        self.assertIsInstance(result, list)

    def test_finds_xml_files(self) -> None:
        """Returns non-empty list when XML files exist (excluding manifest.xml)."""
        if IGNORE_TESTS:
            return
        model_dir = _make_model_dir(self._tmpdir, "BIOMD0000000005",
                                    xml_files=["model.xml"], sedml_files=[])
        result = self._iterator()._findFilesWithExtension(model_dir, ".xml")
        self.assertEqual(len(result), 1)

    def test_excludes_manifest_xml(self) -> None:
        """manifest.xml is excluded even though it has the .xml extension."""
        if IGNORE_TESTS:
            return
        model_dir = _make_model_dir(self._tmpdir, "BIOMD0000000005",
                                    xml_files=[], sedml_files=[], add_manifest=True)
        result = self._iterator()._findFilesWithExtension(model_dir, ".xml")
        self.assertEqual(result, [])

    def test_excludes_manifest_xml_when_other_xml_present(self) -> None:
        """manifest.xml is excluded even when other .xml files are present."""
        if IGNORE_TESTS:
            return
        model_dir = _make_model_dir(self._tmpdir, "BIOMD0000000005",
                                    xml_files=["model.xml"], sedml_files=[], add_manifest=True)
        result = self._iterator()._findFilesWithExtension(model_dir, ".xml")
        self.assertFalse(any("manifest.xml" in p for p in result))

    def test_returns_absolute_paths(self) -> None:
        """Returned paths are absolute."""
        if IGNORE_TESTS:
            return
        model_dir = _make_model_dir(self._tmpdir, "BIOMD0000000005",
                                    xml_files=["model.xml"], sedml_files=[])
        result = self._iterator()._findFilesWithExtension(model_dir, ".xml")
        for path in result:
            self.assertTrue(os.path.isabs(path))

    def test_returns_sorted_paths(self) -> None:
        """Returned paths are in sorted order."""
        if IGNORE_TESTS:
            return
        model_dir = _make_model_dir(self._tmpdir, "BIOMD0000000005",
                                    xml_files=["b.xml", "a.xml", "c.xml"], sedml_files=[])
        result = self._iterator()._findFilesWithExtension(model_dir, ".xml")
        self.assertEqual(result, sorted(result))

    def test_returns_empty_for_no_match(self) -> None:
        """Returns empty list when no files match the extension."""
        if IGNORE_TESTS:
            return
        model_dir = _make_model_dir(self._tmpdir, "BIOMD0000000005",
                                    xml_files=[], sedml_files=["model.sedml"])
        result = self._iterator()._findFilesWithExtension(model_dir, ".xml")
        self.assertEqual(result, [])

    def test_finds_sedml_files(self) -> None:
        """Finds .sedml files when requested."""
        if IGNORE_TESTS:
            return
        model_dir = _make_model_dir(self._tmpdir, "BIOMD0000000005",
                                    xml_files=[], sedml_files=["model.sedml"])
        result = self._iterator()._findFilesWithExtension(model_dir, ".sedml")
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].endswith(".sedml"))


class TestBiomodelsIteratorIter(unittest.TestCase):
    """Tests for BiomodelsIterator.__iter__ using a temp directory.

    Note: __iter__ now filters by ENDTIME_SOURCE_SEDML, so tests that expect
    models to be yielded must provide an endtimes CSV with sedml source entries.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _iterator(self) -> BiomodelsIterator:
        return BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=IGNORE_TESTS)

    def test_yields_biomodels_items(self) -> None:
        """Iterating yields BiomodelsItem instances."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["BIOMD0000000005_url.xml"], sedml_files=[])
        csv_path = _write_endtimes_csv(self._tmpdir, [("BIOMD0000000005", 10.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertTrue(all(isinstance(item, BiomodelsItem) for item in items))

    def test_yields_one_item_per_biomd_dir(self) -> None:
        """One BiomodelsItem is yielded per BIOMD directory."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["BIOMD0000000005_url.xml"], sedml_files=[])
        _make_model_dir(self._tmpdir, "BIOMD0000000002",
                        xml_files=["BIOMD0000000002_url.xml"], sedml_files=[])
        csv_path = _write_endtimes_csv(self._tmpdir, [
            ("BIOMD0000000005", 10.0), ("BIOMD0000000002", 20.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertEqual(len(items), 2)

    def test_skips_non_biomd_directories(self) -> None:
        """Directories whose name does not contain 'BIOMD' are skipped."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["model.xml"], sedml_files=[])
        _make_model_dir(self._tmpdir, "other_dir",
                        xml_files=["model.xml"], sedml_files=[])
        csv_path = _write_endtimes_csv(self._tmpdir, [("BIOMD0000000005", 10.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].model_name, "BIOMD0000000005")

    def test_items_sorted_by_model_name(self) -> None:
        """Yielded items are in sorted order by model name."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000003",
                        xml_files=["model.xml"], sedml_files=[])
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["model.xml"], sedml_files=[])
        _make_model_dir(self._tmpdir, "BIOMD0000000002",
                        xml_files=["model.xml"], sedml_files=[])
        csv_path = _write_endtimes_csv(self._tmpdir, [
            ("BIOMD0000000005", 10.0), ("BIOMD0000000002", 20.0), ("BIOMD0000000003", 30.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        names = [item.model_name for item in items]
        self.assertEqual(names, sorted(names))

    def test_item_model_name_matches_directory(self) -> None:
        """BiomodelsItem.model_name matches the directory name."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["model.xml"], sedml_files=[])
        csv_path = _write_endtimes_csv(self._tmpdir, [("BIOMD0000000005", 10.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertEqual(items[0].model_name, "BIOMD0000000005")

    def test_item_sbml_paths_populated(self) -> None:
        """BiomodelsItem.sbml_paths contains the expected XML file."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["BIOMD0000000005_url.xml"], sedml_files=[])
        csv_path = _write_endtimes_csv(self._tmpdir, [("BIOMD0000000005", 10.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertEqual(len(items[0].sbml_paths), 1)
        self.assertTrue(items[0].sbml_paths[0].endswith("BIOMD0000000005_url.xml"))

    def test_item_sedml_paths_populated(self) -> None:
        """BiomodelsItem.sedml_paths contains the expected SED-ML file."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=[], sedml_files=["model.sedml"])
        csv_path = _write_endtimes_csv(self._tmpdir, [("BIOMD0000000005", 10.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertEqual(len(items[0].sedml_paths), 1)
        self.assertTrue(items[0].sedml_paths[0].endswith("model.sedml"))

    def test_empty_directory_yields_no_items(self) -> None:
        """An empty biomodels_dir yields no items."""
        if IGNORE_TESTS:
            return
        items = list(BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False))
        self.assertEqual(items, [])

    def test_manifest_excluded_from_sbml_paths(self) -> None:
        """manifest.xml is not included in sbml_paths."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["model.xml"], sedml_files=[], add_manifest=True)
        csv_path = _write_endtimes_csv(self._tmpdir, [("BIOMD0000000005", 10.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertFalse(any("manifest.xml" in p for p in items[0].sbml_paths))

    def test_skips_excluded_models(self) -> None:
        """Models listed in excluded_models are not yielded."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["model.xml"], sedml_files=[])
        _make_model_dir(self._tmpdir, "BIOMD0000000002",
                        xml_files=["model.xml"], sedml_files=[])
        csv_path = _write_endtimes_csv(self._tmpdir, [
            ("BIOMD0000000005", 10.0), ("BIOMD0000000002", 20.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir,
                               excluded_models=["BIOMD0000000005"], is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        names = [item.model_name for item in items]
        self.assertNotIn("BIOMD0000000005", names)
        self.assertIn("BIOMD0000000002", names)

    def test_skips_models_from_existing_csv(self) -> None:
        """Models listed in existing_csv_path are not yielded."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["model.xml"], sedml_files=[])
        _make_model_dir(self._tmpdir, "BIOMD0000000002",
                        xml_files=["model.xml"], sedml_files=[])
        csv_path = os.path.join(self._tmpdir, "existing.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME])
            writer.writerow(["BIOMD0000000005"])
        endtimes_csv = _write_endtimes_csv(self._tmpdir, [
            ("BIOMD0000000005", 10.0), ("BIOMD0000000002", 20.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir,
                               existing_csv_path=csv_path, is_report=False, endtimes_csv_path=endtimes_csv)
        items = list(it)
        names = [item.model_name for item in items]
        self.assertNotIn("BIOMD0000000005", names)
        self.assertIn("BIOMD0000000002", names)

    def test_item_existing_df_set_from_csv(self) -> None:
        """BiomodelsItem.existing_df is populated from existing_csv_path."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000002",
                        xml_files=["model.xml"], sedml_files=[])
        csv_path = os.path.join(self._tmpdir, "existing.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME])
            writer.writerow(["BIOMD0000000005"])
        endtimes_csv = _write_endtimes_csv(self._tmpdir, [("BIOMD0000000002", 20.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir,
                               existing_csv_path=csv_path, is_report=False, endtimes_csv_path=endtimes_csv)
        items = list(it)
        self.assertEqual(len(items), 1)
        self.assertFalse(items[0].existing_df.empty)
        self.assertIn(cn.COL_MODEL_NAME, items[0].existing_df.columns)

    def test_item_existing_df_none_without_csv(self) -> None:
        """BiomodelsItem.existing_df is an empty DataFrame when no CSV is provided."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["model.xml"], sedml_files=[])
        csv_path = _write_endtimes_csv(self._tmpdir, [("BIOMD0000000005", 10.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertTrue(items[0].existing_df.empty)

    def test_skips_models_without_sedml_source(self) -> None:
        """Models without ENDTIME_SOURCE_SEDML in the endtimes CSV are skipped."""
        if IGNORE_TESTS:
            return
        _make_model_dir(self._tmpdir, "BIOMD0000000005",
                        xml_files=["model.xml"], sedml_files=[])
        # Write a CSV with steadystate source (not sedml) and pass it to the iterator
        csv_path = os.path.join(self._tmpdir, "endtimes.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME, cn.COL_ENDTIME, cn.COL_ENDTIME_SOURCE])
            writer.writerow(["BIOMD0000000005", "25.0", cn.ENDTIME_SOURCE_STEADYSTATE])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path)
        items = list(it)
        self.assertEqual(len(items), 0)


class TestExtractModelNum(unittest.TestCase):
    """Tests for BiomodelsIterator.extractModelNum."""

    def test_standard_model_name(self) -> None:
        """Extracts the integer from a standard BIOMD name."""
        if IGNORE_TESTS:
            return
        self.assertEqual(BiomodelsIterator.extractModelNum("BIOMD0000000005"), TEST_MODEL_NUM)

    def test_large_model_number(self) -> None:
        """Extracts a larger model number correctly."""
        if IGNORE_TESTS:
            return
        self.assertEqual(BiomodelsIterator.extractModelNum("BIOMD0000001234"), 1234)

    def test_invalid_name_returns_minus_one(self) -> None:
        """Non-numeric suffix returns -1."""
        if IGNORE_TESTS:
            return
        self.assertEqual(BiomodelsIterator.extractModelNum("NOT_A_MODEL"), -1)

    def test_empty_string_returns_minus_one(self) -> None:
        """Empty string returns -1."""
        if IGNORE_TESTS:
            return
        self.assertEqual(BiomodelsIterator.extractModelNum(""), -1)

    def test_biomd_with_no_digits_returns_minus_one(self) -> None:
        """'BIOMD' with no trailing digits returns -1."""
        if IGNORE_TESTS:
            return
        self.assertEqual(BiomodelsIterator.extractModelNum("BIOMD"), -1)


class TestBiomodelsIteratorRange(unittest.TestCase):
    """Tests for first_model_num / last_model_num range filtering in __iter__."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        for name in ["BIOMD0000000001", "BIOMD0000000002", "BIOMD0000000003"]:
            _make_model_dir(self._tmpdir, name, xml_files=["model.xml"], sedml_files=[])

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _names(self, **kwargs) -> List[str]:
        csv_path = _write_endtimes_csv(self._tmpdir, [
            ("BIOMD0000000001", 10.0), ("BIOMD0000000002", 20.0), ("BIOMD0000000003", 30.0)])
        it = BiomodelsIterator(biomodels_dir=self._tmpdir, is_report=False, endtimes_csv_path=csv_path, **kwargs)
        return [item.model_name for item in it]

    def test_default_range_yields_all_models(self) -> None:
        """Default range (0 to 1e9) yields all models."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self._names(), ["BIOMD0000000001", "BIOMD0000000002", "BIOMD0000000003"])

    def test_first_model_num_excludes_lower_models(self) -> None:
        """Models with number below first_model_num are not yielded."""
        if IGNORE_TESTS:
            return
        names = self._names(first_model_num=2)
        self.assertNotIn("BIOMD0000000001", names)
        self.assertIn("BIOMD0000000002", names)
        self.assertIn("BIOMD0000000003", names)

    def test_last_model_num_excludes_higher_models(self) -> None:
        """Models with number above last_model_num are not yielded."""
        if IGNORE_TESTS:
            return
        names = self._names(last_model_num=2)
        self.assertIn("BIOMD0000000001", names)
        self.assertIn("BIOMD0000000002", names)
        self.assertNotIn("BIOMD0000000003", names)

    def test_range_is_inclusive_on_both_ends(self) -> None:
        """first_model_num and last_model_num are both inclusive."""
        if IGNORE_TESTS:
            return
        names = self._names(first_model_num=2, last_model_num=2)
        self.assertEqual(names, ["BIOMD0000000002"])

    def test_narrow_range_yields_subset(self) -> None:
        """A range narrower than the full set yields only the matching models."""
        if IGNORE_TESTS:
            return
        names = self._names(first_model_num=1, last_model_num=2)
        self.assertEqual(names, ["BIOMD0000000001", "BIOMD0000000002"])

    def test_empty_range_yields_nothing(self) -> None:
        """A range that matches no models yields an empty list."""
        if IGNORE_TESTS:
            return
        names = self._names(first_model_num=10, last_model_num=20)
        self.assertEqual(names, [])


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestBiomodelsIteratorReal(unittest.TestCase):
    """Integration tests for BiomodelsIterator using the real BioModels directory."""

    def setUp(self) -> None:
        self._iterator = BiomodelsIterator(is_report=IGNORE_TESTS)

    def test_yields_biomodels_items(self) -> None:
        """All yielded objects are BiomodelsItem instances."""
        if IGNORE_TESTS:
            return
        items = list(self._iterator)
        self.assertTrue(all(isinstance(item, BiomodelsItem) for item in items))

    def test_yields_many_models(self) -> None:
        """At least one model is yielded from the real directory."""
        if IGNORE_TESTS:
            return
        items = list(self._iterator)
        self.assertGreater(len(items), 0)

    def test_known_model_present(self) -> None:
        """BIOMD0000000005 appears in the results."""
        if IGNORE_TESTS:
            return
        names = {item.model_name for item in self._iterator}
        self.assertIn(TEST_MODEL, names)

    def test_known_model_has_sbml_paths(self) -> None:
        """BIOMD0000000005 has at least one SBML file path."""
        if IGNORE_TESTS:
            return
        for item in self._iterator:
            if item.model_name == TEST_MODEL:
                self.assertGreater(len(item.sbml_paths), 0)
                return
        self.fail(f"{TEST_MODEL} not found in iterator results")

    def test_sbml_paths_are_existing_files(self) -> None:
        """All sbml_paths for BIOMD0000000005 point to existing files."""
        if IGNORE_TESTS:
            return
        for item in self._iterator:
            if item.model_name == TEST_MODEL:
                for path in item.sbml_paths:
                    self.assertTrue(os.path.isfile(path), f"Not a file: {path}")
                return
        self.fail(f"{TEST_MODEL} not found in iterator results")

    def test_manifest_excluded_from_real_sbml_paths(self) -> None:
        """manifest.xml is not included in sbml_paths for any model."""
        if IGNORE_TESTS:
            return
        for item in self._iterator:
            for path in item.sbml_paths:
                self.assertNotEqual(os.path.basename(path), "manifest.xml")

    def test_items_sorted_by_model_name(self) -> None:
        """All yielded items are in sorted order by model name."""
        if IGNORE_TESTS:
            return
        names = [item.model_name for item in self._iterator]
        self.assertEqual(names, sorted(names))


class TestGetBiomodelsEndtimes(unittest.TestCase):
    """Tests for getBiomodelsEndtimes."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_dict(self) -> None:
        """Returns a dict."""
        if IGNORE_TESTS:
            return
        result = getBiomodelsEndtimes("/nonexistent/path.csv")
        self.assertIsInstance(result, dict)

    def test_missing_file_returns_empty_dict(self) -> None:
        """Returns empty dict when the CSV file does not exist."""
        if IGNORE_TESTS:
            return
        result = getBiomodelsEndtimes("/nonexistent/path.csv")
        self.assertEqual(result, {})

    def test_valid_csv_returns_mapping(self) -> None:
        """Valid CSV with model_name and end_time columns returns correct mapping."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "endtimes.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME, cn.COL_ENDTIME])
            writer.writerow(["BIOMD0000000005", "25.0"])
            writer.writerow(["BIOMD0000000002", "100.0"])
        result = getBiomodelsEndtimes(csv_path)
        self.assertAlmostEqual(result["BIOMD0000000005"], 25.0)
        self.assertAlmostEqual(result["BIOMD0000000002"], 100.0)

    def test_missing_model_name_column_returns_empty_dict(self) -> None:
        """Returns empty dict when CSV is missing the model_name column."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "bad.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_col", cn.COL_ENDTIME])
            writer.writerow(["BIOMD0000000005", "25.0"])
        with self.assertRaises(ValueError):
            getBiomodelsEndtimes(csv_path)

    def test_missing_end_time_column_returns_empty_dict(self) -> None:
        """Returns empty dict when CSV is missing the end_time column."""
        if IGNORE_TESTS:
            return
        csv_path = os.path.join(self._tmpdir, "bad.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([cn.COL_MODEL_NAME, "wrong_col"])
            writer.writerow(["BIOMD0000000005", "25.0"])
        with self.assertRaises(ValueError):
            getBiomodelsEndtimes(csv_path)


class TestBiomodelsItemEndTime(unittest.TestCase):
    """Tests for the end_time field on BiomodelsItem."""

    def test_end_time_defaults_to_none(self) -> None:
        """end_time is None when not provided."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem("BIOMD0000000005", [], [])
        self.assertIsNone(item.end_time)

    def test_end_time_stored(self) -> None:
        """end_time passed to constructor is accessible."""
        if IGNORE_TESTS:
            return
        item = BiomodelsItem("BIOMD0000000005", [], [], end_time=42.0)
        self.assertAlmostEqual(item.end_time, 42.0)

    def test_iterator_stamps_end_time_from_csv(self) -> None:
        """Iterator sets end_time on each item when the endtimes CSV is present."""
        if IGNORE_TESTS:
            return
        tmpdir = tempfile.mkdtemp()
        try:
            # Build a fake BioModels directory with one model
            model_name = "BIOMD0000000005"
            model_dir = os.path.join(tmpdir, model_name)
            os.makedirs(model_dir)
            open(os.path.join(model_dir, f"{model_name}_url.xml"), "w").close()

            # Write a matching endtimes CSV with sedml source
            csv_path = _write_endtimes_csv(tmpdir, [(model_name, 99.5)])

            iterator = BiomodelsIterator(
                biomodels_dir=tmpdir,
                is_report=False,
                endtimes_csv_path=csv_path,
            )
            items = list(iterator)
            self.assertEqual(len(items), 1)
            self.assertAlmostEqual(items[0].end_time, 99.5)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_iterator_sets_end_time_none_when_not_in_csv(self) -> None:
        """Iterator sets end_time to None for models absent from the endtimes CSV."""
        if IGNORE_TESTS:
            return
        tmpdir = tempfile.mkdtemp()
        try:
            model_name = "BIOMD0000000002"
            model_dir = os.path.join(tmpdir, model_name)
            os.makedirs(model_dir)
            open(os.path.join(model_dir, f"{model_name}_url.xml"), "w").close()

            # Write an empty endtimes CSV (no models listed) so the iterator uses it
            csv_path = _write_endtimes_csv(tmpdir, [])
            iterator = BiomodelsIterator(
                biomodels_dir=tmpdir,
                is_report=False,
                endtimes_csv_path=csv_path,
            )
            items = list(iterator)
            self.assertEqual(len(items), 1)
            self.assertIsNone(items[0].end_time)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()