#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: utci_over_32_time_fraction.py
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
fontsize = 16
flag_percentage = False
if flag_percentage == True:
    cbstr = r'% of record'
else:
    cbstr = r'Exceedence fraction'
threshold = 32
scenario = 126 # 126, 245 or 585
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
if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0

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
#data_directory = project_directory + ''
#filelist = data_directory + 'utci_3hr*.nc'
#paths_to_load = [ glob(f'utci_3hr*.nc') for variable in ['utci'] ]
#paths_to_load = [ glob(filelist) for variable in ['utci'] ]
#paths_to_load = glob(filelist)   
#dataset = xr.open_mfdataset(paths=chain(*paths_to_load))    
#dataset = xr.open_mfdataset(paths_to_load, concat_dim="time", combine="nested", data_vars='minimal', coords='minimal', compat='override', parallel=True)
#dataset = xr.open_mfdataset(paths_to_load[0])

if scenario == 126:
	dataset = xr.open_dataset(data_directory + '/HadGEM3-GC31-LL/ssp126/r1i1p1f3/monthly_avg.nc')
elif scenario == 245:
	dataset = xr.open_dataset(data_directory + '/HadGEM3-GC31-LL/ssp245/r1i1p1f3/monthly_avg.nc')
else:
	dataset = xr.open_dataset(data_directory + '/HadGEM3-GC31-LL/ssp585/r1i1p1f3/monthly_avg.nc')

#------------------------------------------------------------------------------
# CALCULATIONS
#------------------------------------------------------------------------------

# CONVERT: UTCI to degrees Centigrade, set the 32 degree threshold, apply land mask and average over time dimension
	
utci = dataset.utci[:,:,:]-273.15
utci_mean = utci.mean('time') # UTCI 1x1 (2015-2100 mean)
utci_land = utci.where(landseamask.LSMASK) # UTCI(t) 1x1 2015-2100 (land masked)
utci_land_mean = utci_land.mean('time') # UTCI 1x1 (2015-2100 mean) (land masked)
utci_over_threshold = xr.where(utci>threshold, utci, np.nan) # UTCI>32(t) 1x1 2015-2100
utci_over_threshold_land = utci_over_threshold.where(landseamask.LSMASK) # UTCI>32(t) 1x1 2015-2100 (land masked)
utci_over_threshold_land_mean = utci_over_threshold_land.mean('time') # UTCI>32 1x1 (2015-2100 mean) (land masked)

# SLICE: 

#utci_over_threshold_land_weighted = utci_over_threshold_land.weighted(weights) # [DataArrayWeighted]
#utci_over_threshold_land_weighted_mean = utci_over_threshold_land_weighted.mean('time') # (weighted) UTCI>32 1x1 (2015-2100 mean) (land masked)
#utci_over_threshold_land_weighted_lat = utci_over_threshold_land_weighted.mean('lon') # (weighted) UTCI>32(t,lat) 2015-2100 (land masked)
#utci_over_threshold_land_weighted_lat_mean = utci_over_threshold_land_weighted.mean('lon').mean('time') # (weighted) UTCI>32(lat) (2015-2100 mean) (land masked)
#utci_over_threshold_land_weighted_mean_time = utci_over_threshold_land_weighted.mean(("lon", "lat")) # (weighted) UTCI>32(t) (2015-2100 mean) (land masked)

# COUNT:

utci_over_threshold_boolean = (np.isfinite(utci_over_threshold_land)==True) # boolean UTCI>32(t) 1x1 2015-2100 (land masked)
utci_over_threshold_frac_mean = utci_over_threshold_boolean.sum('time')/len(utci.time) # time frac UTCI>32 1x1 (2015-2100 mean) (land masked)
utci_over_threshold_frac_weighted = utci_over_threshold_boolean.weighted(weights) # [DataArrayWeighted]
utci_over_threshold_frac_weighted_mean = utci_over_threshold_frac_weighted.mean('time') # (weighted) time frac UTCI>32 1x1 (2015-2100 mean) (land masked)
utci_over_threshold_frac_weighted_lat = utci_over_threshold_frac_weighted.mean('lon') # (weighted) time frac UTCI>32(t,lat) 2015-2100 (land masked)
utci_over_threshold_frac_weighted_lat_mean = utci_over_threshold_frac_weighted.mean('lon').mean('time') # (weighted) time frac UTCI>32(lat) (2015-2100 mean) (land masked)
utci_over_threshold_frac_weighted_mean_time = utci_over_threshold_frac_weighted.mean(("lon", "lat")) # (weighted) time frac UTCI>32(t) (2015-2100 mean) (land masked)

# CALCULATE: exceedence statistics

N = ((2100-2015+1)*12+(1))*180*360 # = 66938400 (NB: datetime range: 2015-01-30 - 2101-01-30 from inputs
N_vector = [ (np.isfinite(utci[i,:,:])==True).sum() for i in range(len(utci.time)) ]
Nexceedences_vector = [ (np.isfinite(utci_over_threshold[i,:,:])==True).sum() for i in range(len(utci.time)) ]
Nexceedences = np.sum(np.array(Nexceedences_vector)) # = 6140750 ~ 9.17% of all UTCI
Nexceedences_land_vector = [ (np.isfinite(utci_over_threshold_land[i,:,:])==True).sum() for i in range(len(utci.time)) ]
Nexceedences_land = np.sum(np.array(Nexceedences_land_vector)) # = 2558147
Nexceedences_sea = Nexceedences - Nexceedences_land # = 3582603
Nexceedences_mean = (np.isfinite(utci_mean[:,:])==True).sum() # = 9919 --> * (2100-2015+1) * 3 ~ Nexceedences_land

# SAVE: to netCDF

#utci_over_threshold_mean.to_netcdf('global_over_32_mean.nc')
#utci_over_threshold_frac.to_netcdf('global_over_32_frac.nc')
#utci_over_threshold_frac_mean.to_netcdf('global_over_32_frac.nc')
#utci_over_threshold_frac_weighted_lat.to_netcdf('global_over_32_frac.nc')
#utci_over_threshold_frac_weighted_mean.to_netcdf('global_over_32_frac.nc')

#-----------------------------------------------------------------------------
# PLOT: UTCI (model) (scenario) (global) time fraction >32C area-weighted latitudinal monthly timeseries
#-----------------------------------------------------------------------------

# NH and SH boundaries

x = utci.time.values
y = utci.lat.values
Y,X = np.meshgrid(y,x)
Z = utci_over_threshold_frac_weighted_lat.values
Z_nonzero = Z
Z_nonzero[Z_nonzero == 0.] = np.nan
Z_nh = Z_nonzero[:,y>0]
Z_sh = Z_nonzero[:,y<0]
boundary_nh = []
boundary_sh = []
y_nh = y[y>0]
y_sh = y[y<0]
for i in range(len(Z)):
    mask_nh = Z_nh[i,:] == np.nanmin(Z_nh[i,:])
    mask_sh = Z_sh[i,:] == np.nanmin(Z_sh[i,:])
    if mask_nh.sum() == 0: 
        min_nh = np.nan
    else:
        min_nh = y_nh[mask_nh][0]
    if mask_sh.sum() == 0: 
        min_sh = np.nan
    else:
        min_sh = y_sh[mask_sh][0]
    boundary_nh.append(min_nh)
    boundary_sh.append(min_sh)
nh_boundary = np.array(boundary_nh).ravel()    
sh_boundary = np.array(boundary_sh).ravel()    
nh_mask = np.isfinite(nh_boundary)
sh_mask = np.isfinite(sh_boundary)

if scenario == 126:
	plotfile = 'hadgem3' + '_' + 'ssp126' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_lat' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP1 2.6 (2015-2100) zonally-averaged UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    	
elif scenario == 245:
	plotfile = 'hadgem3' + '_' + 'ssp245' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_lat' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP2 4.5 (2015-2100) zonally-averaged  UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    	
else:
	plotfile = 'hadgem3' + '_' + 'ssp585' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_lat' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP5 8.5 (2015-2100) zonally-averaged  UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    

fig,ax = plt.subplots(figsize=(15,10))
#vmin = np.nanmin(Z); vmax = np.nanmax(Z)
vmin = 0; vmax = 1
levels = np.linspace(0,1,21)
g = plt.contourf(X,Y,Z, vmin=vmin, vmax=vmax, levels=levels, cmap=cmap, extend='both')
cb = fig.colorbar(g, ax=ax, shrink=0.6)
cb.set_label(cbstr, rotation=90, labelpad=20, fontsize=fontsize)
cb.set_ticks(np.linspace(0,1,11))        
cb.ax.tick_params(labelsize=fontsize)
#label=r'UTCI > 32$^{\circ}$C boundary: 12m MA'
plt.plot(X[nh_mask],pd.Series(nh_boundary[nh_mask]).rolling(12,center=True).mean(), lw=3, color='red')
plt.plot(X[sh_mask],pd.Series(sh_boundary[sh_mask]).rolling(12,center=True).mean(), lw=3, color='red')
plt.tick_params(labelsize=fontsize)
plt.xlabel('Time', fontsize=fontsize)
plt.ylabel('Latitude, $^{\circ}$N', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
#plt.legend(loc='lower left', bbox_to_anchor=(0, -0.8), ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
#fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)             
plt.savefig(plotfile)
plt.close('all')

#-----------------------------------------------------------------------------
# PLOT: UTCI (model) (scenario) (global) >32C time fraction area-weighted mean latitudinal exceedence
#-----------------------------------------------------------------------------

if scenario == 126:
	plotfile = 'hadgem3' + '_' + 'ssp126' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_lat_mean' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP1 2.6 (2015-2100) mean zonally-averaged UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    	
elif scenario == 245:
	plotfile = 'hadgem3' + '_' + 'ssp245' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_lat_mean' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP2 4.5 (2015-2100) mean zonally-averaged  UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    	
else:
	plotfile = 'hadgem3' + '_' + 'ssp585' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_lat_mean' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP5 8.5 (2015-2100) mean zonally-averaged UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    

fig,ax = plt.subplots(figsize=(15,10))
plt.plot(utci_over_threshold_frac_weighted_lat_mean, utci_over_threshold_frac_weighted_lat_mean.lat, lw=3, color='red')
plt.axhline(y=0, lw=1, ls='dashed', color='black')
plt.xlim(0,0.2)
plt.tick_params(labelsize=fontsize)
plt.xlabel(cbstr, fontsize=fontsize)
plt.ylabel('Latitude, $^{\circ}$N', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(plotfile)
plt.close('all')

#-----------------------------------------------------------------------------
# PLOT: UTCI (model) (scenario) (global) >32C time fraction area-weighted mean timeseries
#-----------------------------------------------------------------------------

if scenario == 126:
	plotfile = 'hadgem3' + '_' + 'ssp126' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_mean' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP1 2.6 (2015-2100) global mean UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    	
elif scenario == 245:
	plotfile = 'hadgem3' + '_' + 'ssp245' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_mean' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP2 4.5 (2015-2100) global mean UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    	
else:
	plotfile = 'hadgem3' + '_' + 'ssp585' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction_area_weighted_mean' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP5 8.5 (2015-2100) global mean UTCI > ' + str(threshold) + r'$^{\circ}$C exceedence fraction'    

fig,ax = plt.subplots(figsize=(15,10))
plt.plot(utci.time,utci_over_threshold_frac_weighted_mean_time, alpha=0.2)
plt.plot(utci.time,pd.Series(utci_over_threshold_frac_weighted_mean_time).rolling(12*10,center=True).mean(), lw=3, color='red')
plt.ylim(0,0.2)
plt.xlabel('Time', fontsize=fontsize)
plt.ylabel(cbstr, fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)
#plt.legend(loc='lower left', bbox_to_anchor=(0, -0.8), ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
#fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)             
plt.savefig(plotfile)
plt.close('all')

#-----------------------------------------------------------------------------
# PLOT: UTCI (model) (scenario) (global) >32C time fraction
#-----------------------------------------------------------------------------

if scenario == 126:
	plotfile = 'hadgem3' + '_' + 'ssp126' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP1 2.6 (2015-2100) UTCI > ' + str(threshold) + r'$^{\circ}$C'    	
elif scenario == 245:
	plotfile = 'hadgem3' + '_' + 'ssp245' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP2 4.5 (2015-2100) UTCI > ' + str(threshold) + r'$^{\circ}$C'    	
else:
	plotfile = 'hadgem3' + '_' + 'ssp585' + '_' + 'utci_over' + '_' + str(threshold) + '_' + 'time_fraction' + '.png'
	titlestr = 'HadGEM3-GC31-LL: SSP5 8.5 (2015-2100) UTCI > ' + str(threshold) + r'$^{\circ}$C'    
	                              
fig, axs = plt.subplots(1,1, figsize=(15,10), subplot_kw=dict(projection=p))
#vmin = utci_over_threshold_frac_mean.min(); vmax = utci_over_threshold_frac_mean.max()
vmin = 0; vmax = 1
g = make_plot(axs, utci_over_threshold_frac_mean, vmin, vmax, cbstr, titlestr, cmap, fontsize)
h = make_plot(axs, utci_over_threshold_frac_weighted_mean, vmin, vmax, cbstr, titlestr, cmap, fontsize)
axs.add_feature(cartopy.feature.OCEAN, zorder=100, alpha=0.2, edgecolor='k')
#cb = fig.colorbar(g, ax=axs.ravel().tolist(), shrink=0.6, extend='both')
cb = fig.colorbar(g, ax=axs, shrink=0.6, extend='both')
cb.set_label(cbstr, rotation=90, labelpad=20, fontsize=fontsize)
cb.set_ticks(np.linspace(0,1,11))  
cb.ax.tick_params(labelsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(plotfile, dpi=300)
plt.close('all')

#-----------------------------------------------------------------------------
print('*** END')




