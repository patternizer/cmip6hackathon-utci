#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: utci_model_projection_quickplot.py
#------------------------------------------------------------------------------
# Version 0.1
# 6 June, 2021
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------

import numpy as np
import xarray as xr
import os, sys

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

# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import cmocean

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

# %matplotlib inline # for Jupyter Notebooks

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 16
cbstr = r'UTCI [$^{\circ}$C] anomaly (from 1985-2015)'
model = 'hadgem3' # hadgem3, bcc or cmcc
scenario = 'ssp585' # ssp126, ssp245 or ssp585
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

modelstr = dict({'hadgem3':'HadGEM3-GC31-LL','bcc':'BCC-CSM2-MR','cmcc':'CMCC-ESM2'})
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

def plot_anomaly(model, scenario, threshold, variable):

    # SET: filename and title
    
    if flag_land == False:
        landstr = 'landsea'
    else:        
        landstr = 'land'
    if flag_threshold == False:
        thresholdstr = 'full'
        utcistr = 'UTCI'
    else:
        thresholdstr = 'utci_over' + '_' + str(threshold)
        utcistr = 'UTCI>' + str(threshold) + r'$^{\circ}$C'

    plotfile = model + '_' + scenario + '_' + thresholdstr + '_' + 'anomaly' + '_'+ landstr + '_' + projection + '.png'
    titlestr = modelstr[model] + ': ' + scenariostr[scenario] + ' (2015-2100)' + ' ' + utcistr + ' ' + 'anomaly'
  
    fig, axs = plt.subplots(1,1, figsize=(15,10), subplot_kw=dict(projection=p))
    #vmin = np.nanmin(ssp126_land_anomaly_mean); vmax = np.nanmax(ssp126_land_anomaly_mean)
    vmin=0; vmax=6
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

#data_directory = project_directory + 'utci_projections_1deg/BCC-CSM2-MR/historical/r1i1p1f1/'
data_directory = project_directory + 'utci_projections_1deg_monthly/'

#------------------------------------------------------------------------------
# LOAD: landseamask & latitudinal weights
#------------------------------------------------------------------------------

landseamask = xr.load_dataset(project_directory + 'landseamask.nc')
weights = np.cos(np.deg2rad(landseamask.lat))

#------------------------------------------------------------------------------
# LOAD: baseline and projection
#------------------------------------------------------------------------------

if model == 'hadgem3': 
    runcode = 'r1i1p1f3'
else: 
    runcode = 'r1i1p1f1'    
baseline = xr.open_dataset(data_directory + '/' + modelstr[model] + '/' + 'historical' + '/' + runcode + '/' + 'monthly_avg.nc')
ssp = xr.open_dataset(data_directory + '/' + modelstr[model] + '/' + scenario + '/' + runcode + '/' + 'monthly_avg.nc')

#------------------------------------------------------------------------------
# CALCULATIONS
#------------------------------------------------------------------------------

# Convert UTCI to degrees Centigrade (set the 32 degree threshold and apply land mask if selected) and average over time dimension for a quick inspection map

if flag_land == False:
    baseline_landsea = baseline.utci-273.15
    ssp_landsea = ssp.utci-273.15    
    if flag_threshold == False:           
        baseline_landsea_normals = baseline_landsea.sel(time=slice("1986", "2016")).groupby('time.month').mean(['time'])
        ssp_landsea_anomaly = ssp_landsea.groupby('time.month') - baseline_landsea_normals
        variable = ssp_landsea_anomaly.mean('time')
    else:      
        baseline_landsea_over_threshold = xr.where(baseline_landsea>threshold, baseline_landsea, np.nan)
        baseline_landsea_over_threshold_normals = baseline_landsea_over_threshold.sel(time=slice("1986", "2016")).groupby('time.month').mean(['time'])                                          
        ssp_landsea_over_threshold = xr.where(ssp_landsea>threshold, ssp_landsea, np.nan)
        ssp_landsea_over_threshold_anomaly = ssp_landsea_over_threshold.groupby('time.month') - baseline_landsea_over_threshold_normals
        variable = ssp_landsea_over_threshold_anomaly.mean('time')
else:
    baseline_land = baseline.utci.where(landseamask.LSMASK)-273.15
    ssp_land = ssp.utci.where(landseamask.LSMASK)-273.15    
    if flag_threshold == False:                
        baseline_land_normals = baseline_land.sel(time=slice("1986", "2016")).groupby('time.month').mean(['time'])
        ssp_land_anomaly = ssp_land.groupby('time.month') - baseline_land_normals
        variable = ssp_land_anomaly.mean('time')
    else:
        baseline_land_over_threshold = xr.where(baseline_land>threshold, baseline_land, np.nan)
        baseline_land_over_threshold_normals = baseline_land_over_threshold.sel(time=slice("1986", "2016")).groupby('time.month').mean(['time'])                                          
        ssp_land_over_threshold = xr.where(ssp_land>threshold, ssp_land, np.nan)
        ssp_land_over_threshold_anomaly = ssp_land_over_threshold.groupby('time.month') - baseline_land_over_threshold_normals
        variable = ssp_land_over_threshold_anomaly.mean('time')

#------------------------------------------------------------------------------
# PLOT: case
#------------------------------------------------------------------------------

plot_anomaly(model, scenario, threshold, variable)

#-----------------------------------------------------------------------------
print('*** END')


