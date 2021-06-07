#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: utci_exceedence_anomaly.py
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

print("cartopy   : ", cartopy.__version__) # cartopy   :  0.18.0
print("matplotlib   : ", matplotlib.__version__) # matplotlib   :  3.3.4
print("numpy   : ", np.__version__) # numpy   :  1.19.2
print("xarray   : ", xr.__version__) # xarray   :  0.17.0

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
projection = 'geostationary'
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

#cmap = 'twilight_r'
#cmap = cmocean.cm.curl
cmap = cmocean.cm.oxy
#cmap = cmocean.cm.tarn

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

# Convert UTCI to degrees Centigrade (set the 32 degree threshold and/or apply land mask) and average over time dimension

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
# PLOTS
#------------------------------------------------------------------------------

plot_anomaly(model, scenario, threshold, variable)

#------------------------------------------------------------------------------
# WORK IN PROGRESS
#------------------------------------------------------------------------------

# Extract fraction of time UTCI is above threshold, weight and slice by latitude

#ssp126_utci_over_threshold_frac = (np.isfinite(ssp126_utci_over_threshold_mean)/len(ssp126_utci.time))*100.
#utci_over_threshold_frac_mean = utci_over_threshold_frac.mean('time') # time-averaged map
#utci_over_threshold_frac_weighted = utci_over_threshold_frac.weighted(weights)
#utci_over_threshold_frac_weighted_lat = utci_over_threshold_frac_weighted.mean('lon')
#utci_over_threshold_frac_weighted_lat_mean = utci_over_threshold_frac_weighted.mean('lon').mean('time')
#utci_over_threshold_frac_weighted_mean = utci_over_threshold_frac_weighted.mean(("lon", "lat"))

# SAVE: extracts to netCDF

#utci_over_threshold_mean.to_netcdf('global_over_32_mean.nc')
#utci_over_threshold_frac.to_netcdf('global_over_32_frac.nc')
#utci_over_threshold_frac_mean.to_netcdf('global_over_32_frac.nc')
#utci_over_threshold_frac_weighted_lat.to_netcdf('global_over_32_frac.nc')
#utci_over_threshold_frac_weighted_mean.to_netcdf('global_over_32_frac.nc')

#-----------------------------------------------------------------------------
print('*** END')


