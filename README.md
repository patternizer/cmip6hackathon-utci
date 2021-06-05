![image](https://github.com/patternizer/cmip6hackathon-utci/blob/master/hadgem3_ssp585_utci_over_32.png)

# cmpi6hackathon-utci

CMIP6 Hackathon UTCI codebase developing with colleagues from the Project 10 team 

## Contents

* `utci_over_32C_ipcc_regions.py` - python script to read in SSP projections plot the mean UTCI>32C masked by IPCC AR5 (or AR6) regions

## Instructions for use

The first step is to clone the latest cmpi6hackathon-utci code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/cmpi6hackathon-utci.git
    $ cd cmpi6hackathon-utci

Then create a DATA/ directory and copy to it the required datasets listed in the code.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64 (see requirements.txt)

    $ python utci_over_32C_ipcc_regions.py
    
## License

To be confirmed but for now [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

