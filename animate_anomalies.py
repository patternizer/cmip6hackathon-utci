#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: animate_anomalies.py
#------------------------------------------------------------------------------
# Version 0.1
# 4 June, 2021
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------

from itertools import chain
from glob import glob
import numpy as np
import xarray as xr
import os, sys
from pathlib import Path
import imageio


# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib import colors as mplc
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.ticker as mticker

import cmocean

# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

print("numpy   : ", np.__version__) # numpy   :  1.19.2
print("xarray   : ", xr.__version__) # xarray   :  0.17.0
print("matplotlib   : ", matplotlib.__version__) # matplotlib   :  3.3.4
print("cartopy   : ", cartopy.__version__) # cartopy   :  0.18.0

#------------------------------------------------------------------------------
# SETTINGS:
#------------------------------------------------------------------------------

fontsize = 16
cbstr = r'UTCI [$^{\circ}$C] anomaly (from 1985-2015)'
threshold = 32
flag_land = False # False --> land+sea
flag_threshold = False # False --> full range of UTCI
projection = 'equalearth'
if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0)
if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0)
if projection == 'robinson': p = ccrs.Robinson(central_longitude=0)
if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0)
if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0)
if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0)
if projection == 'europp': p = ccrs.EuroPP()
if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo()
if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo()
if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0)

scenariostr = dict({'ssp126':'SSP1 2.6','ssp245':'SSP2 4.5','ssp585':'SSP5 8.5'})

# COLORMAP:

#'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cmo.algae', 'cmo.algae_r', 'cmo.amp', 'cmo.amp_r', 'cmo.balance', 'cmo.balance_r', 'cmo.curl', 'cmo.curl_r', 'cmo.deep', 'cmo.deep_r', 'cmo.delta', 'cmo.delta_r', 'cmo.dense', 'cmo.dense_r', 'cmo.diff', 'cmo.diff_r', 'cmo.gray', 'cmo.gray_r', 'cmo.haline', 'cmo.haline_r', 'cmo.ice', 'cmo.ice_r', 'cmo.matter', 'cmo.matter_r', 'cmo.oxy', 'cmo.oxy_r', 'cmo.phase', 'cmo.phase_r', 'cmo.rain', 'cmo.rain_r', 'cmo.solar', 'cmo.solar_r', 'cmo.speed', 'cmo.speed_r', 'cmo.tarn', 'cmo.tarn_r', 'cmo.tempo', 'cmo.tempo_r', 'cmo.thermal', 'cmo.thermal_r', 'cmo.topo', 'cmo.topo_r', 'cmo.turbid', 'cmo.turbid_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

cmap = cmocean.cm.balance
#cmap = 'twilight_shifted'
#cmap = 'afmhot'
#cmap = cmocean.cm.oxy
#cmap = 'gist_heat'

# for other geo apps:

#cmap = cmocean.cm.curl
#cmap = cmocean.cm.topo
#cmap = cmocean.cm.delta

flag_animate = True

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------
    
def make_plot(axi,v,vmin,vmax,cbstr,titlestr,cmap,fontsize):

    # Xarray plotting function with 5x5 grid overlay
    
    g = v.plot(ax=axi, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'orientation':'vertical','extend':'both','shrink':0.7, 'pad':0.1})         
    cb = g.colorbar; cb.ax.tick_params(labelsize=fontsize); cb.set_label(label=cbstr, size=fontsize); cb.remove()
    axi.set_global()        
    axi.coastlines(color='grey')
    axi.set_title(titlestr, fontsize=fontsize)    
    gl = axi.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=0.2, linestyle='-')
    gl.top_labels = False; gl.bottom_labels = False; gl.left_ylabels = False; gl.right_ylabels = False
    gl.xlines = True; gl.ylines = True
    gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,73)) # every 5 degrees
    gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,37))   # every 5 degrees
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER

    return g

def plot_anomaly(model, scenario, threshold, variable, plotfile, titlestr):
  
    fig, axs = plt.subplots(1,1, figsize=(15,10), subplot_kw=dict(projection=p))
    #vmin = np.nanmin(ssp126_land_anomaly_mean); vmax = np.nanmax(ssp126_land_anomaly_mean)
    vmin=-4; vmax=12
    g = make_plot(axs, variable, vmin, vmax, cbstr, titlestr, cmap, fontsize)
    axs.add_feature(cartopy.feature.OCEAN, zorder=100, alpha=0.2, edgecolor='k')
    cb = fig.colorbar(g, ax=axs, shrink=0.6, extend='both')
    cb.set_label(cbstr, rotation=90, labelpad=20, fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize)
    plt.savefig(plotfile, dpi=300)
    plt.close('all')

#------------------------------------------------------------------------------
# SET: paths
#------------------------------------------------------------------------------

#project_directory = '/gws/pw/j05/cop26_hackathons/bristol/project10/'
project_directory = os.curdir + '/' + 'DATA' + '/'
png_directory = os.curdir + '/' + 'PNG' + '/'

#data_directory = project_directory + 'utci_projections_1deg/BCC-CSM2-MR/historical/r1i1p1f1/'
data_directory = project_directory + 'utci_projections_1deg_monthly/'
utci_path = Path(data_directory)

#------------------------------------------------------------------------------
# LOAD: UTCI baseline + projections into a dataframe
#------------------------------------------------------------------------------

# Populate the lists from the directory structure (many thanks to Charles Simpson for this segment)

model_list = []
scenario_list = []

for model_path in utci_path.glob("*"):
    model = str(model_path).split("/")[-1]
    model_list.append(model)
    for scenario_path in model_path.glob("*"):
        scenario = str(scenario_path).split("/")[-1]
        scenario_list.append(scenario)
        print(model, scenario)
                     
#------------------------------------------------------------------------------
# SAVE: UTCI anomaly 1x1 mean
#------------------------------------------------------------------------------

# model_list:    ['CMCC-ESM2','BCC-CSM2-MR','HadGEM3-GC31-LL']
# scenarios: ['ssp126','ssp245','ssp585']    
# runid_list:    ['r1i1p1f1','r1i1p1f1','r1i1p1f3']

scenarios = np.sort(scenario_list[0:3])

for i in range(len(model_list)):

    for j in range(len(scenarios)):        
                 
        # LOAD: UTCI timeseries netCDF

        if flag_land == True:
            filename_timeseries = 'utci_anomaly_land_timeseries' + '_' + model_list[i] + '_' + scenarios[j] + '.nc'
        else:    
            filename_timeseries = 'utci_anomaly_timeseries' + '_' + model_list[i] + '_' + scenarios[j] + '.nc'
        utci_anomaly_land_timeseries = xr.open_dataset(filename_timeseries)

        if flag_animate == True:

            years = [ utci_anomaly_land_timeseries.time.dt.year[k].values for k in range(len(utci_anomaly_land_timeseries.time)) ]
            yearlist = np.unique(years)
            year_pointer = yearlist[0]
            for k in range(len(utci_anomaly_land_timeseries.time)):

                year = str(utci_anomaly_land_timeseries.time.dt.year[k].values)                
                if year == str(year_pointer):
                
                
                    year_pointer += 1
                    variable = utci_anomaly_land_timeseries.utci[k,:,:]
                    plotfile = png_directory + year + '_' + model_list[i] + '_' + scenarios[j] + '_' + 'UTCI' + '_' + 'anomaly' + '_'+ 'land' + '_' + projection + '.png'
                    titlestr = model_list[i] + ':' + ' ' + scenariostr[scenarios[j]] + ' ' + 'UTCI anomaly:' + ' ' + year
                    plot_anomaly(model, scenario, threshold, variable, plotfile, titlestr)
            
            #------------------------------------------------------------------------------
            # GENERATE: Animated GIF of UTCI anomaly per model per scenario
            #------------------------------------------------------------------------------
            
            #use_reverse_order = False
            #png_list = png_directory + '*.png'
            #gif_str = 'model_scenario_utci_timeseries.gif'
            #if use_reverse_order == True:
            #    a = glob.glob(png_list)
            #    images = sorted(a, reverse=True)
            #else:
            #    images = sorted(glob.glob(png_list))
            #var = [imageio.imread(file) for file in images]
            #imageio.mimsave(gif_str, var, fps = 2)
            
            #----------------------------------------------------------------------------
            # CLI --> MAKE GIF & MP4
            #----------------------------------------------------------------------------
            
            # PNG --> GIF:
            # convert -delay 10 -loop 0 png_dir gif_str
            # GIF --> MP4
            # ffmpeg -i gif_str -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" mp4_str
            
        else:
            continue

#-----------------------------------------------------------------------------
print('*** END')


