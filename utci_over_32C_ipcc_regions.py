#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: utci_over_32_ipcc_regions.py
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
import pandas as pd
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

# Geo libraries:

import geopandas as gp
#import pooch
#import pyproj
#from shapely.geometry import Polygon
#conda install -c conda-forge regionmask
#pip install git+https://github.com/mathause/regionmask
import regionmask

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

cmap = 'gist_heat_r'
fontsize = 14
cbstr = r'UTCI [$^{\circ}$C]'
threshold = 32
scenario = 126 # 126, 245 or 585
region = 'Amazon Basin'
flag_ar6 = False # False -->  use AR5
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
        
#------------------------------------------------------------------------------
# SET: paths
#------------------------------------------------------------------------------

#project_directory = '/gws/pw/j05/cop26_hackathons/bristol/project10/'
project_directory = os.curdir + '/' + 'DATA' + '/' 
data_directory = project_directory + 'utci_projections_1deg_monthly/'

#------------------------------------------------------------------------------
# LOAD: landseamask & latitudinal weights
#------------------------------------------------------------------------------

# I regridded a land-sea mask I have for 20CRv3 to 1x1 and then reset the longitude dimension to match the CMIP6 UTCI dataset format. 
# I use CDO to reset the longitude with: cdo sellonlatbox,-180,180,-90,90 landseamask_1x1.nc landseamask.nc

landseamask = xr.load_dataset(project_directory + 'landseamask.nc')
weights = np.cos(np.deg2rad(landseamask.lat)) # latitudinal weights
          
#------------------------------------------------------------------------------
# LOAD: UTCI (model) (scenario) (monthly) dataset
#------------------------------------------------------------------------------

# Lazy load data

#data_directory = project_directory + 'utci_projections_1deg/BCC-CSM2-MR/historical/r1i1p1f1/'
#filelist = data_directory + 'utci_3hr*.nc'
#paths_to_load = [ glob(f'utci_3hr*.nc') for variable in ['utci'] ]
#paths_to_load = [ glob(filelist) for variable in ['utci'] ]
#dataset = xr.open_mfdataset(paths=chain(*paths_to_load))

if scenario == 126:
	dataset = xr.open_dataset(data_directory + '/HadGEM3-GC31-LL/ssp126/r1i1p1f3/monthly_avg.nc')
elif scenario == 245:
	dataset = xr.open_dataset(data_directory + '/HadGEM3-GC31-LL/ssp245/r1i1p1f3/monthly_avg.nc')
else:
	dataset = xr.open_dataset(data_directory + '/HadGEM3-GC31-LL/ssp585/r1i1p1f3/monthly_avg.nc')

# Convert UTCI to degrees Centigrade, set the 32 degree threshold, apply land mask and average over time dimension
	
utci = dataset.utci[:,:,:]-273.15
utci_over_threshold = xr.where( utci > threshold, utci, np.nan)
utci_over_threshold_mean = utci_over_threshold.mean('time').where( landseamask.LSMASK )
          
#-----------------------------------------------------------------------------
# DEFINE: regions
#-----------------------------------------------------------------------------

#file = pooch.retrieve("https://pubs.usgs.gov/of/2006/1187/basemaps/continents/continents.zip", None)
#continents = gp.read_file("zip://" + file)
#display(continents)

#regions_ar6 = regionmask.defined_regions.ar6.all
regions_giorgi = regionmask.defined_regions.giorgi
regions_srex = regionmask.defined_regions.srex
ar5regions = regions_giorgi

mask = ar5regions.mask_3D(dataset.lon, dataset.lat)

# 1) by the index of the region:
# 2) with the abbreviation
# 3) with the long name

#array(['Alaska/N.W. Canada', 'Canada/Greenl./Icel.', 'W. North America',
#       'C. North America', 'E. North America', 'Central America/Mexico',
#       'Amazon', 'N.E. Brazil', 'Coast South America',
#       'S.E. South America', 'N. Europe', 'C. Europe',
#       'S. Europe/Mediterranean', 'Sahara', 'W. Africa', 'E. Africa',
#       'S. Africa', 'N. Asia', 'W. Asia', 'C. Asia', 'Tibetan Plateau',
#       'E. Asia', 'S. Asia', 'S.E. Asia', 'N. Australia',
#       'S. Australia/New Zealand'], dtype='<U24')

#['Australia',
# 'Amazon Basin',
# 'Southern South America',
# 'Central America',
# 'Western North America',
# 'Central North America',
# 'Eastern North America',
# 'Alaska',
# 'Greenland',
# 'Mediterranean Basin',
# 'Northern Europe',
# 'Western Africa',
# 'Eastern Africa',
# 'Southern Africa',
# 'Sahara',
# 'Southeast Asia',
# 'East Asia',
# 'South Asia',
# 'Central Asia',
# 'Tibet',
# 'North Asia']

#r = mask.sel(region=3)
#r = mask.isel(region=(mask.abbrevs == "WNA"))
r = mask.isel(region=(mask.names == region))

#-----------------------------------------------------------------------------
# PLOT: IPCC AR5 Giorgi regions
#-----------------------------------------------------------------------------

# PLOT: AR5 Giorgi regions

text_kws = dict(color="#67000d", fontsize=fontsize, bbox=dict(pad=0.2, color="w"))

plotfile = 'ipcc-ar5-regions-giorgi.png'
titlestr = 'IPCC AR5 Regions: Giorgi'

fig, axs = plt.subplots(1,1, figsize=(15,10), subplot_kw=dict(projection=p))
regions_giorgi.plot(ax=axs, label='name', text_kws=text_kws)
axs.set_title(titlestr, fontsize=fontsize)    
plt.savefig(plotfile, dpi=300)
plt.close('all')

# PLOT: AR5 Giorgi regions (masked)

plotfile = 'ipcc-ar5-regions-giorgi-masked.png'
titlestr = 'IPCC AR5 Regions: Giorgi (masked)'

fig, axs = plt.subplots(4,3, figsize=(15,10), subplot_kw=dict(projection=p))
fg = mask.isel(region=slice(12)).plot(
    subplot_kws=dict(projection=ccrs.PlateCarree()),
    col="region",
    col_wrap=2,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    aspect=1.5,
    cmap='coolwarm')
for ax in fg.axes.flatten(): ax.coastlines()
fg.fig.subplots_adjust(hspace=0, wspace=0.1)
plt.savefig(plotfile, dpi=300)
plt.close('all')

#-----------------------------------------------------------------------------
# PLOT: IPCC AR5 SREX regions
#-----------------------------------------------------------------------------

text_kws = dict(color="#67000d", fontsize=fontsize, bbox=dict(pad=0.2, color="w"))

plotfile = 'ipcc-ar5-regions-srex.png'
titlestr = 'IPCC AR5 Regions: SREX'

fig, axs = plt.subplots(1,1, figsize=(15,10), subplot_kw=dict(projection=p))
regions_srex.plot(ax=axs, label='name', text_kws=text_kws)
axs.set_title(titlestr, fontsize=fontsize)    
plt.savefig(plotfile, dpi=300)
plt.close('all')

#-----------------------------------------------------------------------------
# PLOT: UTCI (global): HadGEM3 projection illustration
#-----------------------------------------------------------------------------

if scenario == 126:
	plotfile = 'hadgem3' + '_' + 'ssp126' + '_' + 'utci_over_' + str(threshold) + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP1 2.6 (2015-2100) mean UTCI > ' + str(threshold) + r'$^{\circ}$C'    	
elif scenario == 245:
	plotfile = 'hadgem3' + '_' + 'ssp245' + '_' + 'utci_over_' + str(threshold) + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP2 4.5 (2015-2100) mean UTCI > ' + str(threshold) + r'$^{\circ}$C'    	
else:
	plotfile = 'hadgem3' + '_' + 'ssp585' + '_' + 'utci_over_' + str(threshold) + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP5 8.5 (2015-2100) mean UTCI > ' + str(threshold) + r'$^{\circ}$C'    
	
fig, axs = plt.subplots(1,1, figsize=(15,10), subplot_kw=dict(projection=p))
vmin = 32; vmax = 42
g = make_plot(axs, utci_over_threshold_mean, vmin, vmax, cbstr, titlestr, cmap, fontsize)
axs.add_feature(cartopy.feature.OCEAN, zorder=100, alpha=0.2, edgecolor='k')
cb = fig.colorbar(g, ax=axs, shrink=0.6, extend='both')
cb.set_label(cbstr, rotation=90, labelpad=20, fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)
plt.savefig(plotfile, dpi=300)
plt.close('all')

#-----------------------------------------------------------------------------
# PLOT: UTCI HadGEM3 projection illustration (masked region = Amazon Basin)
#-----------------------------------------------------------------------------

utci_region = utci_over_threshold_mean.where(r)

if scenario == 126:
	plotfile = 'hadgem3' + '_' + 'ssp126' + '_' + 'utci_over_' + str(threshold) + '_' + region.replace(' ','_').lower() + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP1 2.6 (2015-2100) mean UTCI > ' + str(threshold) + r'$^{\circ}$C'    	
elif scenario == 245:
	plotfile = 'hadgem3' + '_' + 'ssp245' + '_' + 'utci_over_' + str(threshold) + '_' + region.replace(' ','_').lower() + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP2 4.5 (2015-2100) mean UTCI > ' + str(threshold) + r'$^{\circ}$C'    	
else:
	plotfile = 'hadgem3' + '_' + 'ssp585' + '_' + 'utci_over_' + str(threshold) + '_' + region.replace(' ','_').lower() + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP5 8.5 (2015-2100) mean UTCI > ' + str(threshold) + r'$^{\circ}$C'    

fig, axs = plt.subplots(1,1, figsize=(15,10), subplot_kw=dict(projection=p))
vmin = 32; vmax = 42
g = make_plot(axs, utci_region, vmin, vmax, cbstr, titlestr, cmap, fontsize)
axs.add_feature(cartopy.feature.OCEAN, zorder=100, alpha=0.2, edgecolor='k')
cb = fig.colorbar(g, ax=axs, shrink=0.6, extend='both')
cb.set_label(cbstr, rotation=90, labelpad=20, fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)
plt.savefig(plotfile, dpi=300)
plt.close('all')

#-----------------------------------------------------------------------------
print('*** END')


