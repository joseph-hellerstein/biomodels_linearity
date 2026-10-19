"""Tests for scripts/curate_models.py."""
import os, sys, tempfile, unittest
import pandas as pd  # type: ignore
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import src.constants as cn  # type: ignore
from curate_models import main, EXCLUDED_MODELS


def _make_mock_item(model_name="BIOMD0000000001", has_sbml=True, endtime_source="sedml",
                    num_events=0, num_conserved=0, simulate_df_min=1.0, simulate_df_max=1e6):
    item = MagicMock()
    item.model_name = model_name
    item.sbml_paths = ["mock.xml"] if has_sbml else []
    item.endtime_source = endtime_source
    item.end_time = 10.0
    rr_mock = MagicMock()
    rr_mock.getNumEvents.return_value = num_events
    rr_mock.getNumConservedMoieties.return_value = num_conserved
    rr_mock.getNumFloatingSpecies.return_value = 1 if simulate_df_min <= 0 else 1

    class _SimulatorMock(MagicMock):
        def makeRoadRunner(self): return rr_mock, None
        def simulate(self):
            df = pd.DataFrame([[simulate_df_min, simulate_df_max]], columns=["S1", "S2"])
            tc = MagicMock(); tc.timecourse_df = df; return tc

    item._simulator = _SimulatorMock()
    return item


class TestCurateModels(unittest.TestCase):
    def test_excluded_models_is_list(self) -> None:
        self.assertIsInstance(EXCLUDED_MODELS, list)
        for entry in EXCLUDED_MODELS:
            self.assertIsInstance(entry, str)

    @patch("curate_models.BiomodelsIterator")
    def test_creates_output_csv(self, mock_iterator_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "curation.csv")
            item = _make_mock_item(model_name="BIOMD0000000100", num_events=1)
            mock_iterator_cls.return_value.__iter__ = MagicMock(return_value=iter([item]))
            with patch("curate_models.Model.makeBiomodel"), patch("curate_models.Simulator") as ms:
                sim, rr = MagicMock(), MagicMock(); rr.getNumEvents.return_value = 1
                sim.makeRoadRunner.return_value = (rr, None); ms.return_value = sim
                main(output_path=out)
            self.assertTrue(os.path.isfile(out))

    @patch("curate_models.BiomodelsIterator")
    def test_skips_no_sbml(self, mock_iterator_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "curation.csv")
            item_bad = _make_mock_item(has_sbml=False)
            mock_iterator_cls.return_value.__iter__ = MagicMock(return_value=iter([item_bad]))
            with patch("curate_models.Model.makeBiomodel"), patch("curate_models.Simulator"):
                main(output_path=out)
            self.assertFalse(os.path.isfile(out))

    @patch("curate_models.BiomodelsIterator")
    def test_skips_non_sedml_endtime(self, mock_iterator_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "curation.csv")
            item_bad = _make_mock_item(endtime_source="default")
            mock_iterator_cls.return_value.__iter__ = MagicMock(return_value=iter([item_bad]))
            with patch("curate_models.Model.makeBiomodel"), patch("curate_models.Simulator"):
                main(output_path=out)
            self.assertFalse(os.path.isfile(out))


    @patch("curate_models.BiomodelsIterator")
    def test_detects_events_exclusion(self, mock_iterator_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "curation.csv")
            item = _make_mock_item(model_name="BIOMD0000009999", num_events=3)
            mock_iterator_cls.return_value.__iter__ = MagicMock(return_value=iter([item]))
            with patch("curate_models.Model.makeBiomodel"), patch("curate_models.Simulator") as ms:
                sim, rr = MagicMock(), MagicMock()
                rr.getNumEvents.return_value = 3; rr.getNumConservedMoieties.return_value = 0
                rr.getNumFloatingSpecies.return_value = 1
                sim.makeRoadRunner.return_value = (rr, None); ms.return_value = sim
                main(output_path=out)
            import pandas as pd; df = pd.read_csv(out, dtype={cn.COL_SYSTEM_ID: str})
            self.assertEqual(len(df), 1); self.assertEqual(
                df[cn.COL_REASON].iloc[0], "Events")

    @patch("curate_models.BiomodelsIterator")
    def test_detects_conserved_moiety_exclusion(self, mock_iterator_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "curation.csv")
            item = _make_mock_item(model_name="BIOMD0000009998", num_conserved=2)
            mock_iterator_cls.return_value.__iter__ = MagicMock(return_value=iter([item]))
            with patch("curate_models.Model.makeBiomodel"), patch("curate_models.Simulator") as ms:
                sim, rr = MagicMock(), MagicMock()
                rr.getNumEvents.return_value = 0; rr.getNumConservedMoieties.return_value = 2
                rr.getNumFloatingSpecies.return_value = 1
                sim.makeRoadRunner.return_value = (rr, None); ms.return_value = sim
                main(output_path=out)
            import pandas as pd; df = pd.read_csv(out, dtype={cn.COL_SYSTEM_ID: str})
            self.assertEqual(len(df), 1); self.assertEqual(
                df[cn.COL_REASON].iloc[0], "Conserved moieties")

    @patch("curate_models.BiomodelsIterator")
    def test_detects_negative_species_values(self, mock_iterator_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "curation.csv")
            item = _make_mock_item(model_name="BIOMD0000009997", simulate_df_min=-1.0)
            mock_iterator_cls.return_value.__iter__ = MagicMock(return_value=iter([item]))
            with patch("curate_models.Model.makeBiomodel"), patch("curate_models.Simulator") as ms:
                sim, rr = MagicMock(), MagicMock()
                rr.getNumEvents.return_value = 0; rr.getNumConservedMoieties.return_value = 0
                rr.getNumFloatingSpecies.return_value = 1
                import pandas as pd
                df_neg = pd.DataFrame([[-5.0, 2.0]], columns=["S1", "S2"])
                tc = MagicMock(); tc.timecourse_df = df_neg; sim.simulate.return_value = tc
                sim.makeRoadRunner.return_value = (rr, None); ms.return_value = sim
                main(output_path=out)
            import pandas as pd; df_res = pd.read_csv(out, dtype={cn.COL_SYSTEM_ID: str})
            self.assertEqual(len(df_res), 1)
            self.assertEqual(df_res[cn.COL_REASON].iloc[0], "Negative species values")

    @patch("curate_models.BiomodelsIterator")
    def test_detects_extreme_species_differences(self, mock_iterator_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "curation.csv")
            item = _make_mock_item(model_name="BIOMD0000009996", simulate_df_min=1.0, simulate_df_max=2e6)
            mock_iterator_cls.return_value.__iter__ = MagicMock(return_value=iter([item]))
            with patch("curate_models.Model.makeBiomodel"), patch("curate_models.Simulator") as ms:
                sim, rr = MagicMock(), MagicMock()
                rr.getNumEvents.return_value = 0; rr.getNumConservedMoieties.return_value = 0
                rr.getNumFloatingSpecies.return_value = 1
                import pandas as pd
                df_extreme = pd.DataFrame([[1.0, 2e6]], columns=["S1", "S2"])
                tc = MagicMock(); tc.timecourse_df = df_extreme; sim.simulate.return_value = tc
                sim.makeRoadRunner.return_value = (rr, None); ms.return_value = sim
                main(output_path=out)
            import pandas as pd; df_res = pd.read_csv(out, dtype={cn.COL_SYSTEM_ID: str})
            self.assertEqual(len(df_res), 1)
            self.assertEqual(
                df_res[cn.COL_REASON].iloc[0], "Extreme differences in species values")

    @patch("curate_models.BiomodelsIterator")
    def test_no_exclusions_when_all_ok(self, mock_iterator_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "curation.csv")
            item = _make_mock_item(model_name="BIOMD0000000001", num_events=0, num_conserved=0,
                                   simulate_df_min=1.0, simulate_df_max=1e5)  # ratio < 1e6 (ok)
            mock_iterator_cls.return_value.__iter__ = MagicMock(return_value=iter([item]))
            with patch("curate_models.Model.makeBiomodel"), patch("curate_models.Simulator") as ms:
                sim, rr = MagicMock(), MagicMock()
                rr.getNumEvents.return_value = 0; rr.getNumConservedMoieties.return_value = 0
                rr.getNumFloatingSpecies.return_value = 1
                import pandas as pd
                df_ok = pd.DataFrame([[5.0, 5e4]], columns=["S1", "S2"])
                tc = MagicMock(); tc.timecourse_df = df_ok; sim.simulate.return_value = tc
                sim.makeRoadRunner.return_value = (rr, None); ms.return_value = sim
                main(output_path=out)
            import pandas as pd; df_res = pd.read_csv(out, dtype={cn.COL_SYSTEM_ID: str})
            self.assertEqual(len(df_res), 0)


if __name__ == "__main__":
    unittest.main()

