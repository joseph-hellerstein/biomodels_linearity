'''Represents a time course and related properties.'''


import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore
from src.simulator import Simulator, SimulationResult  # type: ignore
from src.biomodels_iterator import getBiomodelsEndtimes  # type: ignore
from src.plot_options import PlotOptions  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pickle
import os
import pandas as pd  # type: ignore
from typing import List, Optional, cast


class Timecourse(object):

    def __init__(self, model: Model,
        start_time: float = cn.START_TIME,
        end_time: Optional[float] = None,
        num_point: int = cn.NUM_POINT,
        timecourse_df: pd.DataFrame = pd.DataFrame(),
        perturbation_value_fraction: float = cn.PERTURBATION_VALUE_FRACTION,
        perturbation_species_fraction: float = cn.PERTURBATION_SPECIES_FRACTION,
        modifable_species_names: Optional[List[str]] = None,
        ) -> None:
        """ 
        Parameters
        ----------
        model : Model
            The model to simulate.
        start_time : float
            Time to start the simulation.
        end_time : float
            Time to end the simulation.
        num_points : int
            Number of time points to simulate.
        timecourse_df : pd.DataFrame
            Optional pre-computed timecourse DataFrame (index: time, columns: species).
        perturbation_value_fraction : float
            Amount of perturbation of initial values as a fraction of the original value.
            May be positive or negative.
        perturbation_species_fraction : float
            Fraction of non-zero initial values that are perturbed
        modifable_species_names : Optional[List[str]]
            Optional list of species names that are modifiable. If not provided, it will be determined
        """
        self.model = model
        self.start_time = start_time
        self.end_time = self._updateEndtime(end_time)
        self.num_point = num_point
        self.perturbation_value_fraction = perturbation_value_fraction
        self.perturbation_species_fraction = perturbation_species_fraction
        self.modifable_species_names = modifable_species_names
        #
        self._timecourse_df = timecourse_df

    def makePerturbationTimecourse(self,
        perturbation_value_fraction: float,
        perturbation_species_fraction: float) -> "Timecourse":
        """Create a new Timecourse with perturbed initial values.
        """
        return type(self)(
            model=self.model,
            start_time=self.start_time,
            end_time=self.end_time,
            num_point=self.num_point,
            perturbation_value_fraction=perturbation_value_fraction,
            perturbation_species_fraction=perturbation_species_fraction
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timecourse):
            return NotImplemented
        return (self.model == other.model and
                bool(np.isclose(self.start_time, other.start_time)) and
                (self.end_time == other.end_time
                        if self.end_time is None or other.end_time is None
                        else bool(np.isclose(self.end_time, other.end_time))) and
                self.num_point == other.num_point and
                bool(np.allclose(self.timecourse_df.values,
                        other.timecourse_df.values, equal_nan=True))
        )

    def _updateEndtime(self, end_time: Optional[float]=None):
        """Determine the end time and its source."""
        if end_time is not None:
            return end_time
        # FIXME: Need a test for that covers this branch
        if self.model.model_name.startswith("BIOMD"):
            endtime_dct = getBiomodelsEndtimes()
            csv_end_time = endtime_dct.get(self.model.model_name, None)
            if csv_end_time is not None:
                return csv_end_time
        return end_time

    @property
    def timecourse_df(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        if self._timecourse_df.empty:
            simulation_result = self._simulate()
            self._timecourse_df = simulation_result.timecourse_df
        return self._timecourse_df
    
    @property
    def num_timepoint(self) -> int:
        """Number of time points in the timecourse."""
        return self.timecourse_df.shape[0]
    
    def _simulate(self) -> SimulationResult:
        """Delegate simulation to a Simulator instance.

        end_time resolution order:
            1. Caller-supplied value (source: user_specified).
            2. BioModels CSV lookup (source: sedml).
            3. Auto-detection via _updateEndtime (source: set by that method).

        Returns
        -------
        SimulationResult
        """
        simulator = Simulator(
            model=self.model,
            start_time=self.start_time,
            end_time=cast(float, self.end_time),
            num_point=self.num_point,
            perturbation_value_fraction=self.perturbation_value_fraction,
            perturbation_species_fraction=self.perturbation_species_fraction,
        )
        return simulator.simulate()
    
    def serialize(self) -> str:
        """
        Serialize the Timecourse to a file

        Returns:
            str: The path to the serialized file. 
        """
        if not self.model.model_name:
            raise ValueError("Model must have a name to serialize Timecourse.")
        path = self.makeBiomodelSerializePath(self.model.model_name)
        dct = {
            "model": self.model,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_point": self.num_point,
            "timecourse_df": self.timecourse_df,
            }
        with open(path, 'wb') as f:
            pickle.dump(dct, f)
        return path
    
    @staticmethod
    def makeBiomodelSerializePath(model_name: str) -> str:
        """
        Get the expected path for a serialized Timecourse of a BioModel.

        Parameters:
            model_name (str): The name of the BioModel.
        """
        return os.path.join(cn.TIMECOURSE_SERIALIZATION_DIR, f"{model_name}_timecourse.pkl")

    @classmethod
    def deserialize(cls, path: str = "", model_name: str = "") -> 'Timecourse':
        """
        Deserialize a Timecourse from a file
        At least one of `path` or `model_name` must be provided.
        If both are provided, `path` takes precedence.

        Parameters:
            path (str): The path to the serialized file.
            model_name (str): The name of the BioModel (used if path is not specified).

        Returns:
            Timecourse: The deserialized Timecourse object.
        """
        if not path and not model_name:
            raise ValueError("At least one of `path` or `model_name` must be provided.")
        if not path:
            path = cls.makeBiomodelSerializePath(model_name)
        # Check if the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No serialized Timecourse found at {path}")    
        # Deserialize
        with open(path, 'rb') as f:
            dct = pickle.load(f)
        # Make sure that columns don't have square brackets (e.g., "[species]") which can happen due to RoadRunner's output formatting
        df = dct['timecourse_df']
        new_column_names = [c[1:-1] if c[0] == "[" else c for c in df.columns]
        df.columns = new_column_names
        return cls(
            model=dct['model'],
            start_time=dct['start_time'],
            end_time=dct['end_time'],
            num_point=dct['num_point'],
            timecourse_df=df,
        )
    
    def plot(self, species_names: Optional[List[str]] = None, is_scatter: bool = False,
             **kwargs) -> PlotOptions:
        """Plot the simulated timecourse for species.

        Parameters
        ----------
        species_names : Optional[List[str]]
            List of species names to plot. If None, all species are plotted.
        is_scatter : bool
            If True, plot as scatter plot. If False, plot as line plot.
        **kwargs
            Passed to PlotOptions. Supported keys: ax, fig, title, xlabel,
            ylabel, legend, xlim, ylim, model_name.

        Returns
        -------
        PlotOptions
        """
        plot_options = PlotOptions(**kwargs)
        ax = plot_options.ax
        if species_names is None:
            species_names = list(self.timecourse_df.columns)
        for i, name in enumerate(species_names):
            if is_scatter:
                ax.scatter(  # type: ignore
                    self.timecourse_df.index,
                    self.timecourse_df[name],
                    color=f"C{i}",
                    label=name,
                )
            else:
                ax.plot(  # type: ignore
                    self.timecourse_df.index,
                    self.timecourse_df[name],
                    color=f"C{i}",
                    label=name,
            )
        plot_options.apply()
        return plot_options

    @classmethod
    def makeBiomodelDF(cls, model_name: str, num_point: int = cn.NUM_POINT,
            end_time: Optional[float] = None) -> "Timecourse":
        """Create a dataframe for a BioModel.

        Parameters
        ----------
        model_name : str
            BioModel identifier (e.g. 'BIOMD0000000001'). Must start with 'BIOMD'.
        num_point : int
            Number of points in the timecourse.

        Returns
        -------
        Timecourse
        """
        model = Model.makeBiomodel(model_name)
        timecourse = cls(model=model, num_point=num_point, end_time=end_time)
        return timecourse
    
    @classmethod
    def makeTimecourses(cls, model: Model,
        start_time: float = cn.START_TIME,
        end_time: Optional[float] = None,
        num_point: int = cn.NUM_POINT,
        perturbation_value_fraction: List[float] = [0.0],
        perturbation_species_fraction: List[float] = [1.0],
        is_plot: bool = True,
        ) -> List["Timecourse"]:
        """Create one Timecourse for every combination of perturbation parameters.
        Constructs a subplot that contains each species. Values are plotted as a scatter plot.


        Parameters
        ----------
        model : Model
        start_time : float
        end_time : Optional[float]
            None uses BioModels CSV lookup or leaves end_time unset.
        num_point : int
        perturbation_value_fraction : List[float]
            Each value is the fractional shift applied to perturbed initial values.
        perturbation_species_fraction : List[float]
            Each value is the fraction of species whose initial values are perturbed.

        Returns
        -------
        List[Timecourse]
            One Timecourse per combination of perturbation parameters.
        """
        timecourses: List[Timecourse] = []
        perturbation_names: List[str] = []
        for value_frac in perturbation_value_fraction:
            for species_frac in perturbation_species_fraction:
                timecourse = cls(
                    model=model,
                    start_time=start_time,
                    end_time=end_time,
                    num_point=num_point,
                    perturbation_value_fraction=value_frac,
                    perturbation_species_fraction=species_frac,
                )
                timecourses.append(timecourse)
                perturbation_names.append(f"vfrc:{value_frac}__sfrc:{species_frac}")
        if is_plot:
            num_species = model.num_species
            num_col = 4
            num_row = (num_species + num_col - 1) // num_col
            fig, axes = plt.subplots(num_row, num_col, figsize=(4 * num_col, 4 * num_row),
                squeeze=False)
            for i, name in enumerate(model.species_names):
                irow = i // num_col
                icol = i % num_col
                ax = axes[irow, icol]  # type: ignore
                for tc in timecourses:
                    ax.plot(tc.timecourse_df.index, tc.timecourse_df[name])
                ax.legend(perturbation_names)
                ax.set_title(name)
                ax.set_xlabel("time")
                ax.set_ylabel("concentration")
            for i in range(num_row):
                for j in range(num_col):
                    if i * num_col + j >= num_species:
                        fig.delaxes(axes[i, j])  # type: ignore
            fig.tight_layout()
            plt.show()
        return timecourses