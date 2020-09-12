import string
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler
from six.moves import zip


def set_mpl_defaults(matplotlib):
    """This function updates the matplotlib library to adjust 
    some default plot parameters
    
    Args:
        matplotlib : matplotlib instance
    
    Returns:
        matplotlib instance
    """
    params = {
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.titlesize": 6,
        "legend.fancybox": True,
        "legend.fontsize": 6,
        "legend.handletextpad": 0.25,
        "legend.handlelength": 1,
        "legend.labelspacing": 0.7,
        "legend.columnspacing": 1.5,
        "legend.edgecolor": (0, 0, 0, 1),  # solid black
        "patch.linewidth": 0.75,
        "figure.dpi": 300,
        "figure.figsize": (2, 2),
        "lines.linewidth": 0.75,
        "lines.markeredgewidth": 1,
        "lines.markeredgecolor": "black",
        "lines.markersize": 2,
        "axes.linewidth": 0.75,
        # "axes.spines.right": False,
        # "axes.spines.top": False,
        "axes.prop_cycle": cycler(
            "color",
            [
                "slategray",
                "darksalmon",
                "mediumaquamarine",
                "indianred",
                "orchid",
                "paleturquoise",
                "tan",
                "lightpink",
            ],
        ),
    }

    # Update parameters
    matplotlib.rcParams.update(params)

    return matplotlib


def cm2inch(*tupl):
    """This function converts cm to inches
    Obtained from: https://stackoverflow.com/questions/14708695/
    specify-figure-size-in-centimeter-in-matplotlib/22787457
    
    Args:
        tupl (tuple) : Size of plot in cm
    
    Returns:
        tuple : Converted image size in inches
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)
