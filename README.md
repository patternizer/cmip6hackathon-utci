![image](https://github.com/patternizer/cmip6hackathon-utci/blob/master/hadgem3_ssp585_utci_over_32.png)
![image](https://github.com/patternizer/cmip6hackathon-utci/blob/master/hadgem3_ssp585_utci_over_32_area_weighted_lat.png)


# cmip6hackathon-utci

CMIP6 Hackathon UTCI codebase developing with colleagues from the Project 10 team 

## Contents

* `utci_model_projection_quickplot.py` - single (model)(projection) run version with flags for land masking and thresholding
* `utci_over_32C_ipcc_regions.py` - python script to read in SSP projections plot the mean UTCI>32C masked by IPCC AR5 (or AR6) regions. Illustrative example with the Amazon Basin masked.
* `utci_over_32C.py` - python script to read in SSP projections and extract the UTCI>32C exceedences. Calculates the latitudinally weighted zonally-averaged mean UTCI>32C edge for the NH and SH. Calculates the mean weighted zonally-average. Calculates the global area-averaged weighted mean timeseires.
* `utci_over_32C_time_fraction.py` - python script to read in SSP projections and create a boolean array of UTCI>32C exceedences for calculation of time fraction statistics. Calculates the latitudinally weighted zonally-averaged mean UTCI>32C edge for the NH and SH. Calculates the mean weighted zonally-average. Calculates the global area-averaged weighted mean timeseires.
* `load_baselines_and_projections.py` - python script to lazy load all model baselines and projections into a dataframe and write out netcdfs containing the (model)(projection) area-averaged mean and gridded timeseries with the option of land masking as input to `animate_anomalies.py`.
* `animate_anomalies.py` - python script to load the (model)(projection) netCDF area-averaged means and monthly gridded timeseries data and plot monthly gridded anomalies for production of  animated GIFs.

## Instructions for use

The first step is to clone the latest cmip6hackathon-utci code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/cmip6hackathon-utci.git
    $ cd cmip6hackathon-utci

Then create a DATA/ directory and copy to it the required datasets listed in the code.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64 (see requirements.txt)

    $ python utci_model_projection_quickplot.py
    $ python utci_over_32C_ipcc_regions.py
    $ python utci_over_32C.py
    $ python utci_over_32C_time_fraction.py
    $ python load_baselines_and_projections.py (prerequisite for `animate_anomalies.py`
    $ python animate_anomalies.py
    
## License

To be confirmed (probably CC-BY 4.0) but for now [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

