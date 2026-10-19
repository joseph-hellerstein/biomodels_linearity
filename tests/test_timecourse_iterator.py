"""Tests  TimecourseIterator and TimecourseIteratorItem."""

import os
import pickle
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd  # type: ignore

import src.constants as cn
from src.model import Model  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator, TimecourseIteratorItem  # type: ignore

IGNORE_TESTS = False

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""
NUM_POINT = 11
NUM_SPECIES = 2
MODEL_NAME = "test_model"
FIRST_REAL_MODEL = "BIOMD0000000005"
HAS_REAL_ZIP = os.path.isfile(TimecourseIterator.getZipPath(num_point=cn.NUM_POINT))


def _makeModel(model_name: str = MODEL_NAME) -> Model:
    return Model(ANTIMONY_MODEL, model_name=model_name)


def _makeTimecourse(model_name: str = MODEL_NAME) -> Timecourse:
    rng = np.random.default_rng(42)
    timepoint_arr = np.linspace(0.0, 10.0, NUM_POINT)
    timecourse_df = pd.DataFrame(
        rng.standard_normal((NUM_POINT, NUM_SPECIES)),
        index=timepoint_arr,
        columns=["S1", "S2"],
    )
    timecourse_df.index.name = "time"
    return Timecourse(
        model=_makeModel(model_name=model_name),
        timecourse_df=timecourse_df,
    )


def _makeZipWithTimecourses(zip_path: str, model_names: list) -> None:
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for name in model_names:
            tc = _makeTimecourse(model_name=name)
            dct = {
                "model": tc.model,
                "start_time": tc.start_time,
                "end_time": tc.end_time,
                "num_point": tc.num_point,
                "timecourse_df": tc._timecourse_df,
            }
            zf.writestr(f"{name}_timecourse.pkl", pickle.dumps(dct))


class TestTimecourseIteratorItem(unittest.TestCase):
    """Tests for TimecourseIteratorItem."""

    def test_stores_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        item = TimecourseIteratorItem(model_name=MODEL_NAME, timecourse=tc)
        self.assertEqual(item.model_name, MODEL_NAME)

    def test_stores_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        item = TimecourseIteratorItem(model_name=MODEL_NAME, timecourse=tc)
        self.assertIs(item.timecourse, tc)


class TestTimecourseIteratorInit(unittest.TestCase):
    """Tests for TimecourseIterator.__init__."""

    def test_custom_zip_path(self) -> None:
        if IGNORE_TESTS:
            return
        it = TimecourseIterator(zip_path="/some/path.zip")
        self.assertEqual(it.zip_path, "/some/path.zip")

    def test_model_collection_stored(self) -> None:
        if IGNORE_TESTS:
            return
        collection = ["model_A", "model_B"]
        it = TimecourseIterator(zip_path="/some/path.zip",  # type: ignore
                model_collection=collection)  # type: ignore
        self.assertEqual(it.model_name_collection, collection)

    def test_model_collection_defaults_to_none(self) -> None:
        if IGNORE_TESTS:
            return
        it = TimecourseIterator(zip_path="/some/path.zip")
        self.assertIsNone(it.model_name_collection)


class TestTimecourseIteratorIter(unittest.TestCase):
    """Tests for TimecourseIterator.__iter__ using a synthetic zip."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._zip_path = os.path.join(self._tmpdir.name, "test.zip")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _makeIterator(self, model_names: list) -> TimecourseIterator:
        _makeZipWithTimecourses(self._zip_path, model_names)
        return TimecourseIterator(zip_path=self._zip_path)

    def test_yields_timecourse_iterator_items(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A"]))
        self.assertTrue(all(isinstance(item, TimecourseIteratorItem) for item in items))

    def test_model_name_extracted_from_filename(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A"]))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].model_name, "model_A")

    def test_yields_timecourse_instances(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A"]))
        self.assertIsInstance(items[0].timecourse, Timecourse)

    def test_skips_non_timecourse_pkl_files(self) -> None:
        if IGNORE_TESTS:
            return
        _makeZipWithTimecourses(self._zip_path, ["model_A"])
        with zipfile.ZipFile(self._zip_path, 'a') as zf:
            zf.writestr("README.txt", "skip me")
            zf.writestr("model_B_other.pkl", b"skip me too")
        items = list(TimecourseIterator(zip_path=self._zip_path))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].model_name, "model_A")

    def test_yields_in_sorted_order(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_C", "model_A", "model_B"]))
        names = [item.model_name for item in items]
        self.assertEqual(names, sorted(names))

    def test_multiple_models_all_yielded(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A", "model_B"]))
        self.assertEqual(len(items), 2)

    def test_empty_zip_yields_no_items(self) -> None:
        if IGNORE_TESTS:
            return
        with zipfile.ZipFile(self._zip_path, 'w') as _:
            pass
        items = list(TimecourseIterator(zip_path=self._zip_path))
        self.assertEqual(items, [])


class TestTimecourseIteratorModelCollection(unittest.TestCase):
    """Tests for the model_collection argument of TimecourseIterator."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._zip_path = os.path.join(self._tmpdir.name, "test.zip")
        _makeZipWithTimecourses(self._zip_path, ["model_A", "model_B", "model_C"])

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _makeIterator(self, model_collection: list) -> TimecourseIterator:
        return TimecourseIterator(zip_path=self._zip_path, model_collection=model_collection)

    def test_yields_only_models_in_collection(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A", "model_C"]))
        names = [item.model_name for item in items]
        self.assertEqual(names, ["model_A", "model_C"])

    def test_yields_all_models_when_collection_is_none(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(TimecourseIterator(zip_path=self._zip_path))
        self.assertEqual(len(items), 3)

    def test_yields_no_items_for_empty_collection(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator([]))
        self.assertEqual(items, [])

    def test_unknown_names_in_collection_are_ignored(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A", "does_not_exist"]))
        names = [item.model_name for item in items]
        self.assertEqual(names, ["model_A"])

    def test_yield_order_follows_zip_not_collection(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_C", "model_A"]))
        names = [item.model_name for item in items]
        self.assertEqual(names, ["model_A", "model_C"])

    def test_yielded_timecourse_matches_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_B"]))
        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0].timecourse, Timecourse)
        self.assertEqual(items[0].timecourse.model.model_name, "model_B")

    def test_duplicate_names_in_collection_yield_each_model_once(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A", "model_B", "model_A"]))
        names = [item.model_name for item in items]
        self.assertEqual(names, ["model_A", "model_B"])

    def test_collection_containing_all_models_yields_same_as_none(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A", "model_B", "model_C"]))
        names = [item.model_name for item in items]
        self.assertEqual(names, ["model_A", "model_B", "model_C"])
        none_items = list(TimecourseIterator(zip_path=self._zip_path))
        self.assertEqual(names, [item.model_name for item in none_items])

    def test_each_yielded_timecourse_matches_its_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_A", "model_C"]))
        self.assertEqual(len(items), 2)
        for item in items:
            self.assertIsInstance(item.timecourse, Timecourse)
            self.assertEqual(item.timecourse.model.model_name, item.model_name)

    def test_yielded_timecourse_data_matches_zip_contents(self) -> None:
        if IGNORE_TESTS:
            return
        items = list(self._makeIterator(["model_B"]))
        self.assertEqual(len(items), 1)
        expected_df = _makeTimecourse(model_name="model_B")._timecourse_df
        pd.testing.assert_frame_equal(items[0].timecourse._timecourse_df, expected_df)

    def test_can_advance_with_next(self) -> None:
        if IGNORE_TESTS:
            return
        iterator = iter(self._makeIterator(["model_A", "model_C"]))
        first = next(iterator)
        second = next(iterator)
        self.assertEqual(first.model_name, "model_A")
        self.assertEqual(second.model_name, "model_C")
        with self.assertRaises(StopIteration):
            next(iterator)


class TestTimecourseIteratorGetTimecourse(unittest.TestCase):
    """Tests for TimecourseIterator.getTimecourse using a synthetic zip."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._zip_path = os.path.join(self._tmpdir.name, "test.zip")
        _makeZipWithTimecourses(self._zip_path, ["model_A", "model_B"])

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _iterator(self) -> TimecourseIterator:
        return TimecourseIterator(zip_path=self._zip_path)

    def test_returns_timecourse_for_valid_model(self) -> None:
        if IGNORE_TESTS:
            return
        tc = self._iterator().getTimecourse("model_A", zip_path=self._zip_path)
        self.assertIsInstance(tc, Timecourse)

    def test_raises_for_unknown_model(self) -> None:
        if IGNORE_TESTS:
            return
        with self.assertRaises(KeyError):
            self._iterator().getTimecourse("does_not_exist", zip_path=self._zip_path)

    def test_timecourse_df_is_nonempty(self) -> None:
        if IGNORE_TESTS:
            return
        tc = self._iterator().getTimecourse("model_A", zip_path=self._zip_path)
        self.assertFalse(tc._timecourse_df.empty)

    def test_can_get_each_model_independently(self) -> None:
        if IGNORE_TESTS:
            return
        it = self._iterator()
        tc_a = it.getTimecourse("model_A", zip_path=self._zip_path)
        tc_b = it.getTimecourse("model_B", zip_path=self._zip_path)
        self.assertEqual(tc_a.model.model_name, "model_A")
        self.assertEqual(tc_b.model.model_name, "model_B")


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestTimecourseIteratorRealZip(unittest.TestCase):
    """Integration tests using the real timecourse zip at cn.TIMECOURSE_ZIP_PATH."""

    def _first_item(self) -> TimecourseIteratorItem:
        return next(iter(TimecourseIterator()))

    def test_yields_timecourse_iterator_item(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self._first_item(), TimecourseIteratorItem)

    def test_model_name_is_nonempty_string(self) -> None:
        if IGNORE_TESTS:
            return
        item = self._first_item()
        self.assertIsInstance(item.model_name, str)
        self.assertGreater(len(item.model_name), 0)

    def test_first_model_is_sorted_first(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(self._first_item().model_name, FIRST_REAL_MODEL)

    def test_timecourse_is_correct_type(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self._first_item().timecourse, Timecourse)

    def test_zip_contains_many_timecourses(self) -> None:
        if IGNORE_TESTS:
            return
        with zipfile.ZipFile(TimecourseIterator.getZipPath(num_point=cn.NUM_POINT)) as zf:
            count = sum(1 for n in zf.namelist() if n.endswith('_timecourse.pkl'))
        self.assertGreater(count, 1)


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestTimecourseIteratorGetTimecourseRealZip(unittest.TestCase):
    """Integration tests for getTimecourse using the real zip."""

    def test_returns_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        tc = TimecourseIterator().getTimecourse(FIRST_REAL_MODEL)
        self.assertIsInstance(tc, Timecourse)

    def test_timecourse_df_is_nonempty(self) -> None:
        if IGNORE_TESTS:
            return
        tc = TimecourseIterator().getTimecourse(FIRST_REAL_MODEL)
        self.assertFalse(tc._timecourse_df.empty)


if __name__ == "__main__":
    unittest.main()
