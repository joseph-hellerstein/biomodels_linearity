"""Piecewise system discovery based on detecting change points in 1-step prediction of derivatives."""

"""
The approach is:
1. Find change points in the timecourse based on changes in the accuracy of one-step prediction of derivatives.
2. Fit a SystemDiscovery model to each segment of the timecourse between change points.
3. Return a PiecewiseSystemDiscovery object that contains the fitted SystemDiscovery models for each segment.
"""

import src.constants as cn
from src.model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from src.system_discovery import SystemDiscovery, NULL_DF  # type: ignore
import src.util as util     # type: ignore
from src.nl_curve import NLCurve  # type: ignore
from src.score import Score  # type: ignore

import ast
import collections
import os
import re
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Any, List, Optional, Tuple, cast, Union


PlotBiomodelsSignalResult = collections.namedtuple('PlotBiomodelsSignalResult',
        ['plot_options', 'piecewise_system_discovery', 'change_point_times'])
ChangepointLookupResult = collections.namedtuple("ChangepointLookupResult",
        ["changepoints", "accuracy", "csv_path"])


class PiecewiseSystemDiscovery(object):
    """Piecewise-linear ODE discovery across detected change-points."""

    def __init__(
        self,
        training_df:  pd.DataFrame,
        max_changepoint: int = 2,
        max_fractional_reduction: float = 0.01,  
        model_id: Optional[Union[str, int]] =None, 
        changepoints: Optional[List[int]] = None,
        is_changepoint_removal: bool = True,
        is_changepoints_from_file: bool = False,
        accuracy_col: str = cn.DEF_ACCURACY_COL,
        accuracy_stat: str = cn.DEF_ACCURACY_STAT,
        **sd_kwargs: Any,
    ) -> None:
        """Construct a piecewise-linear ODE discovery pipeline.

        Args:
            training_df (pd.DataFrame): Time-series data with one column per species.
            max_changepoint (int, optional): Maximum number of change points to detect. Defaults to 2.
                Can be adjusted so that there is enough data per segment.
            max_fractional_reduction (float, optional): Maximum fractional reduction in the sum of squared errors required to accept a new change point. Defaults to 0.01.
            model_id (str, int optional): Optional ID tag used in plots and error messages. Defaults to "".
            changepoints (List[int], optional): List of pre-determined change points. Defaults to None.
            is_changepoint_removal (bool, optional): Whether to allow removal of detected change points. Defaults to True.
            is_changepoints_from_file (bool, optional): Whether to load change points from a CSV file. Defaults to False.
            accuracy_col (str, optional): Column name for the accuracy metric. Defaults to cn.DEF_ACCURACY_COL.
            accuracy_stat (str, optional): Statistic for aggregating species accuracies into a model accuracy. Defaults to cn.DEF_ACCURACY_STAT.
            **sd_kwargs: Arguments forwarded to each per-segment ``SystemDiscovery`` constructor.
        """
        self.training_df = training_df
        self.species_names = list(training_df.columns)
        self.num_species = len(self.species_names)
        self.num_point = training_df.shape[0]
        self.accuracy_col = accuracy_col
        self.accuracy_stat = accuracy_stat
        if model_id is None:
            self.model_name = ""
            self.model_num = -1
        else:
            self.model_num, self.model_name = Model.getModelNumberAndName(model_id)
        self.max_changepoint = max_changepoint
        self.max_fractional_reduction = max_fractional_reduction
        sd_kwargs["poly_degree"] = sd_kwargs.get("poly_degree", 1)
        self._sd_kwargs = sd_kwargs
        self.changepoints : Optional[List[int]] = changepoints  # if None, will be determined during fit()a
        self._is_changepoint_removal = is_changepoint_removal
        self._is_changepoints_from_file = is_changepoints_from_file

        self._subsequence_models: List[SystemDiscovery] = []
        self._subsequence_boundaries: List[Tuple[float, float]] = []
        self._subsequence_lengths: List[int] = []
        self._is_fitted: bool = False
        # Baseline (whole-timecourse) SystemDiscovery model, fit lazily so construction stays cheap
        # when ``fit()`` is never invoked.  Accessed via :meth:`_getBaselineSystemDiscovery`.
        self._sys_disc: Optional[SystemDiscovery] = None
        # Values computed by fit
        self.baseline_pred_df: Optional[pd.DataFrame] = None  # Predictions for the whole-timecourse baseline model, constructed during ``fit()`` and used in plotting and scoring.
        self.score_df: Optional[pd.DataFrame] = None  # Score of baseline_pred_df vs. training_df, constructed during ``fit()`` and used by ``score()``.
        # Per-segment prediction cache used by _scoreFromSegmentList (always predicts
        # against self.training_df, so caching by (model identity, boundary) is safe).
        # Reset at the start of each eliminateChangepoints() call.
        self._predict_cache: dict = {}

    @property
    def num_changepoint(self) -> int:
        """Return the number of detected change points."""
        return len(self._subsequence_models) - 1 if self._is_fitted else 0

    @property
    def nl_curve(self) -> NLCurve:
        """Return the nonlinear curve object for the whole timecourse."""
        if not self._is_fitted:
            raise RuntimeError("PiecewiseSystemDiscovery must be fit() before accessing nl_curve.")
        result = NLCurve(self.boundaries, name=self.model_name)
        return result

    @property
    def boundaries(self) -> List[int]:
        """Return the start/end row indices for each segment, including 0 and n_rows."""
        if not self._is_fitted:
            raise RuntimeError("PiecewiseSystemDiscovery must be fit() before accessing boundaries.")
        return [0] + cast(List[int], self.changepoints) + [self.num_point - 1]

    def _getBaselineSystemDiscovery(self) -> SystemDiscovery:
        """Lazily build and cache the whole-timecourse baseline ``SystemDiscovery`` model."""
        if self._sys_disc is None:
            self._sys_disc = SystemDiscovery(
                self.training_df, is_normalize=True, **self._sd_kwargs).fit()
        return cast(SystemDiscovery, self._sys_disc)

    def _requireFitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                    "PiecewiseSystemDiscovery must be fit() before this operation.")

    @classmethod
    def parseChangepointsCell(cls, raw: object, model_name: str="") -> Optional[List[int]]:
        """Parse a single ``changepoints`` cell value into a list of ints.

        Accepts plain Python lists or their string forms, including cells
        written by :func:`str` on a list of numpy scalars (e.g.
        ``"[np.int64(20), np.int64(40)]"``); the numpy scalar wrappers are
        stripped before parsing. Raises ``ValueError`` when *raw* is not a
        string or list, when the cell is empty, or when it does not parse to
        a list of ints.
        """
        if not isinstance(raw, (str, list)):
            raise ValueError(f"Unexpected changepoints type in CSV for {model_name}: {type(raw).__name__}")
        text = str(raw).strip()
        if not text:
            raise ValueError("Changepoints cell is empty; cannot parse.")
        # Strip numpy scalar wrappers (e.g. "np.int64(20)") left behind when a
        # list of numpy scalars was stringified into the CSV (numpy >= 1.24 repr).
        text = re.sub(r"np\.\w+\(([^)]*)\)", r"\1", text)
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid changepoints format in CSV for {model_name}: {text}")

        if not isinstance(parsed, list):
            raise ValueError(
                f"Unexpected changepoints type in CSV for {model_name}: "
                f"{type(parsed).__name__}"
            )
        return [int(cp) for cp in parsed]

    def _getChangepointsFromFile(
            self, csv_path: str = cn.PIECEWISE_PREDICTIONS_MODEL_PATH,
    ) -> List[ChangepointLookupResult]:
        """Look up pre-computed changepoints in ``data/piecewise_predictions_model.csv``.

        Matches every CSV row whose ``(system_id, max_changepoint,
        max_fractional_reduction)`` equals ``(self.model_name,
        self.max_changepoint, self.max_fractional_reduction)`` and returns
        each matching row's ``changepoints`` column parsed back into a list
        of ints. When multiple rows match, one inner list is returned per row,
        in the order they appear in the CSV file.

        Arguments
        ---------
        csv_path : str
            Path to the CSV file containing pre-computed changepoints.
        accuracy_col : str
            Column name for the accuracy metric.

        Returns
        -------
        list[ChangepointLookupResult]
            A list of changepoint lists -- one per matching row -- when at
            least one valid match is found.  An empty cell on a matching row
            contributes an inner ``[]``.  Rows whose cell cannot be parsed via
            :func:`ast.literal_eval` are silently skipped.  Returns []
            only when (a) the CSV is missing, (b) no row matches the
            three-key lookup, or (c) every matching row has an unparseable
            ``changepoints`` cell.

        Notes
        -----
        The CSV has no ``model_name`` column; we match against ``system_id``,
        which holds values like ``"BIOMD0000000005"`` and is the only model
        identifier in that file.
        """
        if os.path.exists(csv_path) is False:
            return []

        df = pd.read_csv(csv_path)
        mask = (df[cn.COL_MAX_CHANGEPOINT] == self.max_changepoint) & \
                (df[cn.COL_MAX_FRACTIONAL_REDUCTION] == self.max_fractional_reduction) & \
                (df[cn.COL_SYSTEM_ID] == self.model_name)

        if not mask.any():
            return []

        results: List[ChangepointLookupResult] = []
        masked_df = df.loc[mask]
        cp_series = masked_df[cn.COL_CHANGEPOINTS]
        acc_series = masked_df[self.accuracy_col]

        for i in range(len(masked_df)):
            raw_cp = cp_series.iloc[i]
            parsed = self.parseChangepointsCell(raw_cp, model_name=self.model_name)
            accuracy = float(acc_series.iloc[i])
            if np.isnan(accuracy):
                raise ValueError(f"NaN accuracy in CSV for {self.model_name} at row {i}")
            results.append(ChangepointLookupResult(parsed, accuracy, csv_path=csv_path))

        # Sort by descending accuracy so callers see the best configuration first.
        results.sort(key=lambda r: r.accuracy, reverse=True)

        return results

    def _fitSegments(self, changepoints: List[int]) -> Tuple[List[SystemDiscovery],
            List[Tuple[float, float]], List[int]]:
        """Build (models, boundaries, lengths) from a given set of indices."""
        time_arr = self.training_df.index.to_numpy(dtype=float)
        boundary_index_arr = [0] + changepoints + [self.num_point]
        models: List[SystemDiscovery] = []
        boundaries: List[Tuple[float, float]] = []
        lengths: List[int] = []
        for lo, hi in zip(boundary_index_arr[:-1], boundary_index_arr[1:]):
            subsequence_df = self.training_df.iloc[lo:hi]
            end_time = time_arr[hi] if hi < self.num_point else time_arr[-1]
            boundaries.append((float(time_arr[lo]), float(end_time)))
            lengths.append(hi - lo)
            try:
                sys_disc = SystemDiscovery(subsequence_df, **self._sd_kwargs)
            except Exception as e:
                raise RuntimeError(f"Error fitting SystemDiscovery for segment {lo}:{hi}: {e}")
            try:
                sys_disc = sys_disc.fit()  # type: ignore
                models.append(sys_disc)
            except Exception as e:
                raise RuntimeError(f"Error re-fitting SystemDiscovery for segment {lo}:{hi}: {e}")
        return models, boundaries, lengths

    def _makeChangepointsWithoutElimination(self) -> List[int]:
        """Generate an initial set of evenly spaced changepoints and then repeatedly 
        eliminates changepoints that do not degrade accuracy by less than
        ``max_fractional_reduction``.

        Returns
        -------
        list[int]
            Sorted list of surviving changepoint indices into the training data.
        """
        num_point = self.num_point
        max_changepoint = self._adjustMaxChangepoints()

        if max_changepoint <= 0:
            return []
        if max_changepoint >= num_point:
            raise ValueError(
                f"max_changepoint {max_changepoint} exceeds number of points {num_point}.")

        # (a) Evenly spaced initial changepoints -- identical to the original.
        step = num_point / (max_changepoint + 1)
        candidate_indices = [int(round((i + 1) * step)) for i in range(max_changepoint)]
        changepoints: List[int] = []
        for cp in sorted(candidate_indices):
            if cp < 1 or cp >= num_point:
                continue
            changepoints.append(cp)

        return changepoints[:max_changepoint]

    def _makeChangepointsWithElimination(self) -> List[int]:
        """Generate an initial set of evenly spaced changepoints and then repeatedly 
        eliminates changepoints as long as the reduction in accuracy is less than ``
        ``max_fractional_reduction`` relative to the baseline (whole-timecourse) model.

        Returns
        -------
        list[int]
            Sorted list of surviving changepoint indices into the training data.
        """
        changepoints = self._makeChangepointsWithoutElimination()
        if not changepoints:
            return []
        return self.eliminateChangepoints(changepoints)

    def eliminateChangepoints(self, changepoints: List[int], num_estimate: int = 3) -> List[int]:
        """Eliminates changepoints as long as the estimated reduction in accuracy is less than
        ``max_fractional_reduction`` relative to the baseline (full-changepoint) piecewise fit.

        Parameters
        ----------
        changepoints : list[int]
            Initial set of changepoint indices to consider for elimination.
        num_estimate: int
            Number of estimates used for calculating accuracy rate

        Returns
        -------
        list[int]
            Sorted list of surviving changepoint indices into the training data.
        """
        self.changepoints = changepoints
        self.fit()
        # Fresh per elimination round: model objects from a *previous* fit() may have
        # been dropped (self._subsequence_models reassigned), so their ids could be
        # reused -- only safe to reuse cache entries within one round.
        self._predict_cache = {}

        # Calculate the accuracy rate
        accuracy_rate_estimators = [self._estimateAccuracyRate(changepoints)
                for _ in range(num_estimate)]
        sorted_estimator_result = sorted(accuracy_rate_estimators, key = lambda x: x.accuracy_rate)
        idx = (len(accuracy_rate_estimators) - 1)//2
        accuracy_rate_estimate = sorted_estimator_result[idx]

        # Find the surviving changepoints
        surviving_changepoint_arr = np.array(changepoints)
        if accuracy_rate_estimate.accuracy_rate <= 0:
            total_frob_dist = np.inf
        else:
            # Convert the maxiumum fraction reduction in accuracy to a Frobenius distance 
            total_frob_dist = self.max_fractional_reduction / accuracy_rate_estimate.accuracy_rate
        # Eliminate changepoints with the smallest Forbenius distances
        # that sum to the total Frobenius distance 
        frob_dist_idx = np.argsort(accuracy_rate_estimate.frob_dist_arr)
        culmulative_frobenius_dist = np.cumsum(accuracy_rate_estimate.frob_dist_arr[frob_dist_idx])
        sel_arr = culmulative_frobenius_dist > total_frob_dist
        surviving_changepoints = list(surviving_changepoint_arr[frob_dist_idx[sel_arr]])
        #
        return list(np.sort(surviving_changepoints))
    
    def _makeFrobeniusDistances(self) -> List[float]:
        """Compute Frobenius distances between consecutive segment Jacobian matrices.

        For a piecewise system with k changepoints, there are k+1 segments and k pairwise
        differences (between segments 0-1, 1-2, ..., k-1-k). Each distance is the Frobenius
        norm of the element-wise difference between adjacent segment Jacobians. These
        distances provide a measure of how much the system dynamics change across detected changepoints.

        Returns
        -------
        list[float]
            List of Frobenius distances between consecutive segments' Jacobian matrices.
            Length equals ``len(self.changepoints)`` (i.e., the number of detected change points).

        Raises
        ------
        RuntimeError
            If the instance has not been fit or any segment's models are empty.
        """
        self._requireFitted()
        if not hasattr(self, '_subsequence_models') or not self._subsequence_models:
            raise RuntimeError("No segment models available; call fit() before using this method.")

        # Extract each segment's Jacobian from its SINDy model. For a linear (poly_degree=1)
        # system the coefficient matrix is already the Jacobian A, but we slice off the bias
        # column via coef_arr[:, 1:] to match the convention used in ``_simulateSimple`` for
        # consistency across polynomial degrees.
        jacobians = []
        for sys_disc in self._subsequence_models:
            if not hasattr(sys_disc, 'model'):
                raise RuntimeError(
                    f"Segment SystemDiscovery model has no .model attribute: {type(sys_disc).__name__}"
                )
            coef_arr = np.asarray(sys_disc.model.coefficients())  # (n_species, n_features)
            jacobian = coef_arr[:, 1:]                             # drop bias column -> square A
            jacobians.append(jacobian)

        # Pairwise Frobenius differences between consecutive segments. Length == k changepoints.
        diffs = [
            float(np.linalg.norm(jacobians[i] - jacobians[i + 1], ord='fro'))
            for i in range(len(jacobians) - 1)
        ]
        return diffs

    def _buildTrialSegments(self, remove_positions):
        """Build trial segment (models, boundaries) via incremental merging.

        Only re-fits models for merged segments (those spanning multiple original 
        intervals); unchanged segments reuse their existing fitted models from ``self``.
        This avoids the O(k) segment re-fits that would occur if a full trial 
        PiecewiseSystemDiscovery were fit, reducing cost to O(c) where c is the number 
        of changepoints removed.

        Parameters
        ----------
        remove_positions : array-like of int
            Sorted indices (into ``changepoint_arr``) of changeppoints to be removed.

        Returns
        -------
        tuple[list[SystemDiscovery], list[tuple[float, float]]]
            ``(models, boundaries)`` for the trial configuration with merged segments 
            re-fit and unchanged segments reused from the original fit.
        """
        if self.changepoints is None or not self._subsequence_models:
            raise RuntimeError("PiecewiseSystemDiscovery must be fit() before building trial segments.")
        #
        num_changeppoints = len(self._subsequence_models) - 1
        orig_boundaries = list(self._subsequence_boundaries)
        orig_seg_lengths = list(self._subsequence_lengths)
        time_arr = self.training_df.index.to_numpy(dtype=float)
        n_rows = len(time_arr)

        remove_set = set(int(x) for x in remove_positions)

        # Surviving changeppoint row indices (sorted ascending), plus boundaries 0 and n_rows.
        surviving_cps_rows = [int(self.changepoints[i])
                for i in range(num_changeppoints) if i not in remove_set]
        new_seg_end_rows = [0] + surviving_cps_rows + [n_rows]

        # Precompute original segment start row indices (cumulative).
        orig_cum_starts = np.cumsum([0] + list(orig_seg_lengths))  # length k+2

        def _find_orig_seg_idx(row):
            """Find which original segment contains the given row index.

            Binary search (orig_cum_starts is sorted ascending) instead of a linear
            scan -- this is called twice per final segment, so a linear scan here
            made the whole function O(n^2) in the number of original segments.
            """
            idx = int(np.searchsorted(orig_cum_starts, max(row, 0), side="right")) - 1
            return max(idx, 0)

        new_models = []
        new_boundaries = []

        for seg_i in range(len(new_seg_end_rows) - 1):
            s_row = new_seg_end_rows[seg_i]
            e_row = new_seg_end_rows[seg_i + 1]

            first_idx = _find_orig_seg_idx(s_row)
            last_idx = (_find_orig_seg_idx(e_row - 1) if e_row > 0 else first_idx)

            if first_idx == last_idx:
                # Spans exactly one original segment -- reuse model (no re-fit needed).
                new_models.append(self._subsequence_models[first_idx])
                new_boundaries.append(orig_boundaries[first_idx])
            else:
                # Multiple segments merged -- need to re-fit on combined data.
                start_time = (float(time_arr[s_row]) 
                        if s_row < n_rows else float(time_arr[-1]))
                end_time = (float(time_arr[e_row - 1]) 
                            if e_row > 0 and e_row <= n_rows else float(time_arr[-1]))

                merged_data = self.training_df.iloc[s_row:e_row]
                new_model = SystemDiscovery(merged_data, **self._sd_kwargs).fit()

                new_models.append(new_model)
                new_boundaries.append((start_time, end_time))

        return new_models, new_boundaries

    def _computeScoreDf(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Score a predicted timecourse directly against ``self.training_df``.

        Parameters
        ----------
        pred_df : pd.DataFrame
            Predicted timecourse, same shape/index as (a subset of) ``self.training_df``.

        Returns
        -------
        pd.DataFrame
            ``Score.score_df``: one row per species plus one ``"model"`` aggregation row
            (see ``cn.COL_AGGREGATION_TYPE``), with columns for every statistic in
            ``cn.COLUMN_STATISTICS`` (mean, min, max, p05...p99, etc.).
        """
        score = Score(accuracy_col=self.accuracy_col)
        score.add(self.training_df, pred_df, system_id=self.model_name)
        return score.score_df

    def _readScore(self, score_df: pd.DataFrame) -> float:
        """Read ``self.accuracy_stat`` off the ``"model"`` aggregation row of *score_df*."""
        if self.accuracy_stat not in score_df.columns:
            raise ValueError(f"Unknown statistic: {self.accuracy_stat}")
        mask = score_df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL
        return float(score_df.loc[mask, self.accuracy_stat].iloc[0])

    def _scoreFromSegmentList(self, models, boundaries) -> float:
        """Score an arbitrary list of (model, boundary) pairs using the same aggregation
        as :meth:`PiecewiseSystemDiscovery.score`: stitch each segment's prediction into a
        single whole-timecourse prediction and compare it directly against
        ``self.training_df``.

        Parameters
        ----------
        models : list[SystemDiscovery]
            Fitted SystemDiscovery models for each segment.
        boundaries : list[tuple[float, float]]
            Start/end times for each segment.

        Returns
        -------
        float
            Aggregated score across all segments and species.
        """
        pred_df = self._stitchPredictionsCached(models, boundaries)
        score_df = self._computeScoreDf(pred_df)
        return self._readScore(score_df)

    def _stitchPredictionsCached(self,
            models: List[SystemDiscovery],
            boundaries: List[Tuple[float, float]]) -> pd.DataFrame:
        """Like :meth:`_stitchPredictions`, but caches each segment's prediction
        keyed by ``(id(model), t_start, t_end)``.

        Only safe because this is always called against ``self.training_df`` (never
        an arbitrary ``test_df``, unlike :meth:`predict`) -- the same (model, boundary)
        pair always yields the same prediction. ``_buildTrialSegments`` reuses model
        objects from ``self._subsequence_models`` across repeated trial-removal calls
        within one ``eliminateChangepoints()`` round, so most segments hit this cache
        instead of re-integrating the ODE.
        """
        pred_frames: List[pd.DataFrame] = []
        for seg_model, (t_start, t_end) in zip(models, boundaries):
            key = (id(seg_model), t_start, t_end)
            cached = self._predict_cache.get(key)
            if cached is None:
                test_seg_df = self.training_df[
                    (self.training_df.index >= t_start) & (self.training_df.index <= t_end)]
                cached = seg_model.predict(test_df=test_seg_df)
                self._predict_cache[key] = cached
            pred_df = cached
            if t_start > 0:
                pred_df = pred_df[pred_df.index > t_start]
            pred_frames.append(pred_df)
        full_pred_df = pd.concat(pred_frames)
        if full_pred_df.index.duplicated().any():
            raise RuntimeError(
                "Duplicate rows in prediction result.  This should not happen.")
        return full_pred_df

    EstimatorResult = collections.namedtuple("EstimatorResult",
            ["accuracy_rate", "total_frob_dist", "delta_accuracy", "num_candidate",
            "remove_idx_arr", "frob_dist_arr"]) 
    # accuracy_rate: rate at which accuracy decreases with Frobenius distance
    # total_frob_dist: total Frobenius distance eliminated
    # delta_accuracy: change in accuracy for the trial PiecewiseSystemDiscovery
    # num_candidate: number of candidate changepoints for removal
    # remove_idx_arr: indices of the changepoints that are candidates for removal
    # frob_dist_arr: Frobenius differences for the removed changepoints. Ordered by time sequence
    def _estimateAccuracyRate(
            self,
            changepoints: List[int],
            num_estimation_changepoint: int =10,
            max_frac_frob_dist: float = 0.1) -> EstimatorResult:
        """Estimate the accuracy_rate, the reduction in accuracy per Frobenius Jacobian difference.

        The estimate is constructed as follows:
            1. Compute Frobenius distances D between consecutive segment Jacobians. 
            2. Create a trial piecewise system discovery by randomly removing num_estimation_changepoint
                that have a small difference in the fractional Frobenius distance.
            3. Calculate the total_frob_dist for the selected changepoints.
            4. Calculate baseline_score and trail_score from the 
                baseline and trial PiecewiseSystemDiscoverys
            5. accuracy_rate = (score_baseline - score_trial)/total_frobenius_distance

        Parameters
        ----------
        changepoints: List[int]
            Changepoints to consider
        num_estimation_changepoint: int
            Number of changepoints used to estimate accuracy rate
        max_frac_frob_dist: float
            Maximum fractional Frobenius distance between the Jacobians of
            adjust segments
            for the associated changepoint to be included in the estimate of
            accuracy_rate

        Returns
        -------
        EstimatorResult
            accuracy_rate: rate at which accuracy decreases with Frobenius distance
            total_frob_dist: total frobenius distance eliminated
            delta_accuracy: change in accuracy for the trail PiecewiseSystemDiscovery
            num_candidate: number of candidate changepoints for removal
            remove_idx_arr: indices of the changepoints that are candidates for removal
            frob_dist_arr: Frobenius differences for the removed changepoints

        Raises
        ------
        RuntimeError
            If the instance has not been fit or there are no changepoints / segments to merge.
        """
        if len(changepoints) == 0:
            raise ValueError(
                "No changepoints provided; cannot estimate accuracy reduction.")
        self._requireFitted()
        changepoint_arr = np.array(changepoints)
        # Sort so positional indices into changepoint_arr align with the order used
        # by _makeFrobeniusDifferences() (which corresponds to self.changepoints, stored sorted).
        changepoint_arr = np.sort(changepoint_arr)
        # 1. Get Frobenius distances between consecutive segment Jacobians.
        frob_dist_arr = np.array(self._makeFrobeniusDistances())
        total_frob_dist = np.sum(frob_dist_arr)
        if np.isclose(total_frob_dist, 0.0):
            return self.EstimatorResult(
                accuracy_rate=0.0,
                total_frob_dist=np.nan,
                delta_accuracy=np.nan,
                num_candidate=-1,
                remove_idx_arr=[],
                frob_dist_arr=frob_dist_arr)
        else:
            frob_dist_arr = frob_dist_arr/total_frob_dist

        # Compute baseline (piecewise) score across all segments.
        base_score = self.score()

        # Randomly select which changepoints to use in estimation (positional indices).
        # The removed changepoints are the complement: those present in self but absent in trial_psd.
        all_idx_arr = np.random.permutation(len(changepoint_arr))
        sorted_frob_dist_arr = frob_dist_arr[all_idx_arr]
        # Only choose changepoints whose fractional distance is below max_frac_frob_dist
        all_remove_idx_arr = np.array([i for i in all_idx_arr
                if  sorted_frob_dist_arr[i] < max_frac_frob_dist])
        if len(all_remove_idx_arr) == 0:
            return self.EstimatorResult(
                accuracy_rate=0.0,
                total_frob_dist=np.nan,
                delta_accuracy=np.nan,
                num_candidate=-1,
                remove_idx_arr=[],
                frob_dist_arr=frob_dist_arr)
        num_candidate = min(len(all_remove_idx_arr), num_estimation_changepoint)
        remove_idx_arr = all_remove_idx_arr[:num_candidate]
        # Build trial segment list via incremental merge of adjacent segments 
        # around removed changeppoints. Only re-fits merged segments; unchanged 
        # segments reuse existing fitted models from self to avoid O(k) re-fits.
        try:
            trial_models, trial_boundaries = self._buildTrialSegments(remove_idx_arr)
            adjusted_score = self._scoreFromSegmentList(trial_models, trial_boundaries)
        except Exception as e:
            raise ValueError(
                f"Cannot estimate reduction in accuracy for {self.model_name}: {e}") from e
        # Estimate the accuracy rate without the removed changepoints, which is what was "eliminated".
        accuracy_diff = max(0, base_score - adjusted_score)
        total_frob_dist = float(np.sum(frob_dist_arr[remove_idx_arr]))
        if total_frob_dist == 0.0:
            raise ValueError(
                f"Total Frobenius distance for removed changepoints is zero; "
                f"cannot estimate accuracy rate for {self.model_name}")
        accuracy_rate = accuracy_diff / total_frob_dist
        return self.EstimatorResult(
                accuracy_rate=accuracy_rate,
                total_frob_dist=total_frob_dist,
                delta_accuracy=accuracy_diff,
                remove_idx_arr=remove_idx_arr,
                num_candidate=num_candidate,
                frob_dist_arr=frob_dist_arr)

    def _adjustMaxChangepoints(self) -> int:
        """Adjust the maximum number of changepoints based on data points and species."""
        if (self.max_changepoint > 0) and (self.num_species > self.num_point / self.max_changepoint):
            # Few data points per species relative to changepoints -- cap so each
            # segment retains enough rows for a reliable PySINDy estimate.
            return self.num_point // self.num_species - 1
        else:
            return self.max_changepoint

    def fit(self,  min_accuracy: float = 0.0) -> 'PiecewiseSystemDiscovery':
        """Detect change points and fit a ``SystemDiscovery`` model to each segment.

        After this call, :attr:`_subsequence_models`, :attr:`_subsequence_boundaries`,
        and :attr:`_subsequence_lengths` are populated; :meth:`predict` is available.
        The baseline whole-timecourse model is built lazily on first access.

        Parameters
        ----------
        min_accuracy : float, optional
            Minimum accuracy required for a changepoint to be considered valid. Defaults to 0.80.
        """
        if (self.changepoints is None) and self._is_changepoints_from_file:
            df = util.getPSDPredictionDF(max_fractional_reduction=self.max_fractional_reduction)
            mask = df[cn.COL_SYSTEM_ID] == self.model_name
            mask &= df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL
            mask &= df[self.accuracy_col] >= min_accuracy
            if mask.any():
                dff = df[mask]
                # Try to use auc column if it exists, otherwise fall back to num_changepoint
                if cn.COL_AUC in dff.columns:
                    auc_ser = dff[cn.COL_AUC]
                    best_idx = auc_ser.idxmin()
                else:
                    # Fall back to minimizing num_changepoint if auc not available
                    best_idx = dff[cn.COL_NUM_CHANGEPOINT].idxmin()
                changepoints = dff.loc[[best_idx]][cn.COL_CHANGEPOINTS].values[0]
                if isinstance(changepoints, str):
                    changepoints = eval(changepoints)  # type: ignore
                self.changepoints = cast(List[int], changepoints)
            else:
                self._is_fitted = False
                return self
        if self.changepoints is None:
            if self._is_changepoint_removal:
                self.changepoints = self._makeChangepointsWithElimination()
            else:
                self.changepoints = self._makeChangepointsWithoutElimination()
        (self._subsequence_models, self._subsequence_boundaries,
        self._subsequence_lengths) = self._fitSegments(cast(List[int], self.changepoints))
        self._is_fitted = True
        # Cache the whole-timecourse piecewise prediction for reuse in scoring/plotting.
        self.baseline_pred_df = self.predict(test_df=self.training_df)
        self.score_df = self._computeScoreDf(self.baseline_pred_df)
        return self

    def predict(self, test_df: Optional[pd.DataFrame] = NULL_DF) -> pd.DataFrame:
        """Predict concentrations by using SystemDiscovery models for
            each segment of the timecourse based on the initial condition in
            test_df. Only timepoint 0 of test_df is used for initial conditions;
            the rest of the rows are ignored.
            Verifies that the time grid of test_df matches the training data's time grid.

        Parameters
        ----------
        test_df : pd.DataFrame, optional
            If provided, provides initial conditions (first row) and time grid
            (index).  When omitted, the training data's first row and index are used.

        Returns
        -------
        pd.DataFrame
            Predicted concentrations with one column per species; time as the index.
        """
        self._requireFitted()

        # Build a full time grid for this prediction run.
        if test_df is None:
            test_df = NULL_DF
        if test_df.empty:
            test_df = self.training_df.copy()
        else:
            self._checkColumns(list(test_df.columns))
            self._checkTimegGrid(test_df.index.to_numpy(dtype=float))
        test_df = cast(pd.DataFrame, test_df)
        return self._stitchPredictions(self._subsequence_models, self._subsequence_boundaries, test_df)

    def _stitchPredictions(self,
            models: List[SystemDiscovery],
            boundaries: List[Tuple[float, float]],
            test_df: pd.DataFrame) -> pd.DataFrame:
        """Integrate each segment with its own fitted ``SystemDiscovery`` model and concatenate.

        Parameters
        ----------
        models : list[SystemDiscovery]
            Fitted SystemDiscovery models for each segment.
        boundaries : list[tuple[float, float]]
            Start/end times for each segment.
        test_df : pd.DataFrame
            Full timecourse (initial conditions and time grid) to predict over.

        Returns
        -------
        pd.DataFrame
            Predicted concentrations with one column per species; time as the index.
        """
        pred_frames: List[pd.DataFrame] = []
        for seg_model, (t_start, t_end) in zip(models, boundaries):
            test_seg_df = test_df[(test_df.index >= t_start) & (test_df.index <= t_end)]
            pred_df = seg_model.predict(test_df=test_seg_df)
            if t_start > 0:
                # Do not include the first row of each segment except for the first segment,
                # since it is already included in the previous segment's prediction.
                pred_df = pred_df[pred_df.index > t_start]
            pred_frames.append(pred_df)
        full_pred_df = pd.concat(pred_frames)
        # Check for duplicate rows
        if full_pred_df.index.duplicated().any():
            raise RuntimeError(
                "Duplicate rows in prediction result.  This should not happen.")
        return full_pred_df

    def _checkColumns(self, col_list: List[str]) -> None:
        """Validate that *col_list* matches the species names of the first segment model."""
        if sorted(col_list) != sorted(self.species_names):
            raise ValueError(
                f"Column mismatch: test_df columns {sorted(col_list)} do not match "
                f"expected species names {self.species_names}."
            )

    def _checkTimegGrid(self, time_arr: np.ndarray) -> None:
        """Validate that *time_arr* matches the time grid of the first segment model."""
        if not np.allclose(time_arr, self.training_df.index.to_numpy(dtype=float)):
            raise ValueError(
                f"Time grid mismatch: test_df index {time_arr} does not match "
                f"expected time grid {self.training_df.index.to_numpy(dtype=float)}."
            )
        if not time_arr[0] == 0:
            raise ValueError(
                f"Time grid mismatch: test_df index {time_arr} does not start at 0."
            )

    def getScoreDetails(self, test_df: Optional[pd.DataFrame] = None,
                score_type="timecourse") -> pd.DataFrame:
        """Return a DataFrame of per-subsequence ScoreInfo."""
        self._requireFitted()
        if test_df is None:
            test_df = NULL_DF
        score_dfs = []
        for sys_disc, (start, end) in zip(self._subsequence_models, self._subsequence_boundaries):
            test_seg_df = test_df.iloc[(test_df.index >= start) & (test_df.index <= end)]
            score_info = sys_disc.getScoreDetails(test_df=test_seg_df, score_type=score_type)
            score_info[cn.COL_START_TIME] = start
            score_info[cn.COL_ENDTIME] = end
            score_dfs.append(score_info)
        result_df = pd.concat(score_dfs, ignore_index=True)
        return result_df

    def __str__(self) -> str:
        block_list: List[str] = []
        for idx, (model, (start, end)) in enumerate(
                zip(self._subsequence_models, self._subsequence_boundaries), start=1):
            header = f"[subsequence {idx}: t in [{start:.1f}, {end:.1f})]"
            equation_line_list = [f"  {line}" for line in str(model).strip().split("\n")]
            block_list.append("\n".join([header] + equation_line_list))
        return "\n\n".join(block_list)

    def plotPiecewise(self, num_true_point: int = -1, 
                suptitle="Actual vs. Predicted",
                species_names: Optional[List[str]] = None,
                is_nochangepoint_plot: bool = True,
                is_changepoint_plot: bool = True,
                title_additional: str = "",
                **plt_kwargs: Any) -> PlotOptions:
        """Two-panel comparison: 0 change points (top) vs max_changepoint (bottom).

        Both panels show actual (scatter) vs predicted (line) species concentrations.
        The bottom panel marks each detected change point with a vertical dashed line.

        Parameters
        ----------
        num_true_point : int
            Number of actual-data scatter points to show per panel.
            -1 means show all points.  If the training_df has more than this many points,
        species_names : Optional[List[str]]
            List of species names to plot. If None, all species are plotted.
        suptitle : str
            Title for the entire figure.  Defaults to "Actual vs. Predicted".
        is_nochangepoint_plot: bool
            Plot with no changepoints
        is_changepoint_plot: bool
            Plot with changepoints

        **plt_kwargs
            Forwarded to PlotOptions. Supported keys: fig, ax, title, xlabel,
            ylabel, legend, xlim, ylim, model_name.  ``figsize`` is also
            accepted and consumed here (not passed to PlotOptions).

        Returns
        -------
        PlotOptions
            Wraps the figure and the bottom axes.  Call ``plt.show()`` or
            ``po.fig.savefig(...)`` on the returned object as needed.
        """
        if species_names is None:
            species_names = self.species_names
        self._requireFitted()
        figsize: tuple[float, float] = plt_kwargs.pop("figsize", (10, 8))

        time_arr = self.training_df.index.to_numpy(dtype=float)
        actual_arr = self.training_df.to_numpy(dtype=float)
        if num_true_point < 0 or num_true_point >= len(time_arr):
            num_true_point = len(time_arr)
        num_skip = max(1, len(time_arr) // num_true_point)

        # Get baseline vs. piecewise scores and predictions; lazily build the whole-timecourse model.
        sys_disc = self._getBaselineSystemDiscovery()
        sys_disc_score = sys_disc.score()
        sys_disc_pred_df = sys_disc.predict()
        psd_score = self.score()
        # Construct the plot
        if is_nochangepoint_plot and is_changepoint_plot:
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            plot_options = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)
        elif is_nochangepoint_plot:
            fig, ax_top = plt.subplots(1, 1, figsize=figsize, sharex=True)
            plot_options = PlotOptions(fig=fig, ax=ax_top, **plt_kwargs)
        elif is_changepoint_plot:
            fig, ax_bot = plt.subplots(1, 1, figsize=figsize, sharex=True)
            plot_options = PlotOptions(fig=fig, ax=ax_bot, **plt_kwargs)
        else:
            raise ValueError("At least one of is_nochangepoint_plot or is_changepoint_plot must be True.")
        change_point_times = [start for start, _ in self._subsequence_boundaries[1:]]
        ##
        def _draw(pred_df: pd.DataFrame, score: float,
                auc : Optional[float] = None,
                vlines: Optional[List[float]] = None, **plt_options) -> None:
            po = PlotOptions(**plt_options)
            ax = po.ax
            ymax = actual_arr.max().max()
            legends:List[str] = []
            for idx, name in enumerate(species_names):
                color = f"C{idx}"
                ax.scatter(  # type: ignore
                    time_arr[::num_skip], actual_arr[::num_skip, idx],
                    marker="o", s=30, linestyle="-", color=color, label=f"{name} actual", zorder=3,
                )
                legends.append(name)
            for idx, name in enumerate(species_names):
                color = f"C{idx}"
                if pred_df is not None and name in pred_df.columns:
                    ax.plot(  # type: ignore
                        pred_df.index, pred_df[name],
                        "--", lw=1.5, color=color, label=f"{name} predicted",
                    )
            if vlines:
                for t in vlines:
                    ax.axvline(t, color="black", linestyle=":", lw=2.5, alpha=0.6)  # type: ignore
            ax.grid(True, alpha=0.3)  # type: ignore
            if self.model_name.startswith("BIOMD"):
                model_num_str = str(int(self.model_name[6:]))
            else:
                model_num_str = self.model_name
            if auc is not None:
                auc_title = f", AUC={auc:.3f}"
            else:
                auc_title = ""
            po.title = model_num_str + ": " + plt_options.get("title", "") + f" ({self.accuracy_stat} species p10 accuracy={score:.3f}{auc_title}"
            if len(title_additional) > 0:
                po.title += f", {title_additional})" # type: ignore
            else:
                po.title += ")" # type: ignore
            if ymax > 0.0:
                po.ylim = (0.0, ymax)
            po.apply()
            ax.legend(legends)  # type: ignore
        ##
        if is_nochangepoint_plot:
            _draw(fig=fig, ax=ax_top, pred_df=sys_disc_pred_df, score=sys_disc_score,  # type: ignore
                    title="0 change points", **plt_kwargs)
        if is_changepoint_plot:
            _draw(fig=fig, ax=ax_bot, pred_df=self.baseline_pred_df, score=psd_score,  # type: ignore
                    title=f"{self.num_changepoint} change points",
                    vlines=change_point_times,
                    auc=self.nl_curve.auc, **plt_kwargs)
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return plot_options

    def printEquations(self) -> None:
        """Pretty-print the discovered ODE for each subsequence."""
        self._requireFitted()
        print(str(self))

    def score(self) -> float:
        """Return the aggregated score for the fitted piecewise model.

        Reads ``self.accuracy_stat`` off ``self.score_df`` (computed in ``fit()`` by
        scoring ``self.baseline_pred_df`` directly against ``self.training_df``). Uses
        the same aggregation as :meth:`_scoreFromSegmentList` so that scores computed
        for the fitted piecewise model and for trial (candidate) segmentations are
        directly comparable.
        """
        self._requireFitted()
        return self._readScore(cast(pd.DataFrame, self.score_df))