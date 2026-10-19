'''Sets up plot options'''

import matplotlib.pyplot as plt # type: ignore
from typing import Optional, Tuple, Union
import src.constants as cn
import numpy as np # type: ignore

class PlotOptions(object):

    def __init__(self,
            ax=None,
            fig=None,
            title: Optional[str] = None,
            xlabel: str = "time",
            ylabel: str = "concentration",
            legend: Union[bool, list[str]] = True,
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            model_name: str = "",
            figsize: Tuple[float, float] = (8, 6),
            suptitle: Optional[str] = None,
            ):
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        self.ax = ax
        self.fig = fig
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.model_name = model_name
        self.figsize = figsize
        self.suptitle = suptitle

    def to_dict(self):
        return self.__dict__
    
    def apply(self):
        if self.title is not None:
            if len(self.model_name) > 0:
                title = f"{self.model_name}: {self.title}"
            else: 
                title = self.title
            self.ax.set_title (title) # type: ignore
        if self.xlabel is not None:
            self.ax.set_xlabel(self.xlabel)  # type: ignore
        if self.ylabel is not None:
            self.ax.set_ylabel(self.ylabel) # type: ignore
        if isinstance(self.legend, bool) and self.legend and self.ax.get_legend_handles_labels()[0]:  # type: ignore
            self.ax.legend() # type: ignore
        if isinstance(self.legend, list):
            self.ax.legend(self.legend) # type: ignore
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim) # type: ignore
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)  # type: ignore
        if self.suptitle is not None:
            self.fig.suptitle(self.suptitle) # type: ignore