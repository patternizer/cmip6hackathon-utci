#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: load_baselines_and_projections.py
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

#------------------------------------------------------------------------------
# SET: paths
#------------------------------------------------------------------------------

#project_directory = '/gws/pw/j05/cop26_hackathons/bristol/project10/'
project_directory = os.curdir + '/' + 'DATA' + '/'
png_directory = project_directory + 'PNG' + '/'

#data_directory = project_directory + 'utci_projections_1deg/BCC-CSM2-MR/historical/r1i1p1f1/'
data_directory = project_directory + 'utci_projections_1deg_monthly/'
utci_path = Path(data_directory)

#------------------------------------------------------------------------------
# LOAD: landseamask & latitudinal weights
#------------------------------------------------------------------------------

landseamask = xr.load_dataset(project_directory + 'landseamask.nc')
weights = np.cos(np.deg2rad(landseamask.lat))

#------------------------------------------------------------------------------
# LOAD: UTCI baseline + projections into a dataframe
#------------------------------------------------------------------------------

# Populate the lists from the directory structure (many thanks to Charles Simpson for this segment)

file_nested_list = []
model_list = []
scenario_list = []
runid_list = []

for model_path in utci_path.glob("*"):
    model = str(model_path).split("/")[-1]
    model_list.append(model)
    file_nested_list.append([])
    for scenario_path in model_path.glob("*"):
        scenario = str(scenario_path).split("/")[-1]
        scenario_list.append(scenario)
        file_nested_list[-1].append([])
        for runid_path in scenario_path.glob("*"):
            runid = str(runid_path).split("/")[-1]
            runid_list.append(runid)
            print(model, scenario, runid)
            file_nested_list[-1][-1].append(list(runid_path.glob("*")))
            
ds_scenario_results = []
for model in model_list:

    print(model)
    # Get the historical baseline
    scenario = "historical"
    monthly_path = list((utci_path / model / "historical").glob("*/monthly_avg.nc"))[0]
    runid = str(monthly_path).split("/")[-2]
    ds_monthly = xr.open_dataset(monthly_path)

    ds_baseline = (
        ds_monthly
        .sel(time=slice("1986", "2016"))
        .mean(["time"])
    )
    print(ds_baseline)
    
    # Get the land fraction
#    ds_sftof = xr.open_dataset(project_directory+'sftof.nc')
#    ds_sftof = ds_sftof.interp(lat=ds_baseline.lat, lon=ds_baseline.lon)

    ds_scenario_results.append([])
    
    # Get the scenario results
    for scenario in np.unique(scenario_list):
    
        print(scenario)
        if scenario == "historical":
            continue
        monthly_path = list((utci_path / model / scenario).glob("*/monthly_avg.nc"))[0]
        ds_monthly = xr.open_dataset(monthly_path)

        utci = (
            (ds_monthly - ds_baseline)
#            .where(ds_sftof.sftof==0)
            .assign_coords(scenario=[scenario], model=[model])
        )
        ds_scenario_results[-1].append(utci)
    ds_scenario_results[-1] = xr.combine_by_coords(ds_scenario_results[-1])
          
#------------------------------------------------------------------------------
# SAVE: UTCI anomaly 1x1 mean
#------------------------------------------------------------------------------

# model_list:    ['CMCC-ESM2','BCC-CSM2-MR','HadGEM3-GC31-LL']
# scenarios: ['ssp126','ssp245','ssp585']    
# runid_list:    ['r1i1p1f1','r1i1p1f1','r1i1p1f3']

scenarios = np.sort(scenario_list[0:3])

for i in range(len(model_list)):

    for j in range(len(scenarios)):        
        
        # APPLY: land mask, extract timeseries and timeseries mean averaged over time dimension
 
        utci_anomaly = ds_scenario_results[i].utci[j,:,:,:]
        utci_anomaly_mean = utci_anomaly.mean('time')
        utci_anomaly_land_timeseries = utci_anomaly.where(landseamask.LSMASK)
        utci_anomaly_land_mean = utci_anomaly.mean('time').where(landseamask.LSMASK)
        
        filename_timeseries = 'utci_anomaly_timeseries' + '_' + model_list[i] + '_' + scenarios[j] + '.nc'
        filename_mean = 'utci_anomaly_mean' + '_' + model_list[i] + '_' + scenarios[j] + '.nc'
        filename_land_timeseries = 'utci_anomaly_land_timeseries' + '_' + model_list[i] + '_' + scenarios[j] + '.nc'
        filename_land_mean = 'utci_anomaly_land_mean' + '_' + model_list[i] + '_' + scenarios[j] + '.nc'

        # SAVE: UTCI mean and timeseries to netCDF
        utci_anomaly.to_netcdf(filename_timeseries)
        utci_anomaly_mean.to_netcdf(filename_mean)
        utci_anomaly_land_timeseries.to_netcdf(filename_land_timeseries)
        utci_anomaly_land_mean.to_netcdf(filename_land_mean)

#-----------------------------------------------------------------------------
print('*** END')


