from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..decorators import *
from ....data_objs.analysis import ParamapAnalysisBase

@dependencies('compute_power_spectra')
def plot_ps_window_data(analysis_obj: ParamapAnalysisBase, dest_folder: str, **kwargs) -> None:
    """Plots the power spectrum data for each window in the ROI.

    The power spectrum data is plotted along with the average power spectrum and a line of best fit
    used for the midband fit, spectral slope, and spectral intercept calculations. Also plots the 
    frequency band used for analysis.
    """
    
    assert Path(dest_folder).is_dir(), "plot_ps_data visualization: Power spectrum plot output folder doesn't exist"
    assert hasattr(analysis_obj.windows[0].results, 'ss'), "Spectral slope not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'si'), "Spectral intercept not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'nps'), "Normalized power spectrum not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'f'), "Frequency not found in results"
    
    ss_arr = [window.results.ss for window in analysis_obj.windows]
    si_arr = [window.results.si for window in analysis_obj.windows]
    nps_arr = [window.results.nps for window in analysis_obj.windows]

    fig, ax = plt.subplots()

    ss_mean = np.mean(np.array(ss_arr)/1e6)
    si_mean = np.mean(si_arr)
    nps_arr = [window.results.nps for window in analysis_obj.windows]
    av_nps = np.mean(nps_arr, axis=0)
    f = analysis_obj.windows[0].results.f
    x = np.linspace(min(f), max(f), 100)
    y = ss_mean*x + si_mean

    for nps in nps_arr[:-1]:
        ax.plot(f/1e6, nps, c="b", alpha=0.2)
    ax.plot(f/1e6, nps_arr[-1], c="b", alpha=0.2, label="Window NPS")
    ax.plot(f/1e6, av_nps, color="r", label="Av NPS")
    ax.plot(x/1e6, y, c="orange", label="Av LOBF")
    ax.plot(2*[analysis_obj.config.analysis_freq_band[0]/1e6], [np.amin(nps_arr), np.amax(nps_arr)], c="purple")
    ax.plot(2*[analysis_obj.config.analysis_freq_band[1]/1e6], [np.amin(nps_arr), np.amax(nps_arr)], c="purple", label="Analysis Band")
    ax.set_title("Normalized Power Spectra")
    ax.legend()
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_ylim([np.amin(nps_arr), np.amax(nps_arr)])
    ax.set_xlim([min(f)/1e6, max(f)/1e6])
    
    fig.savefig(Path(dest_folder) / 'nps_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
