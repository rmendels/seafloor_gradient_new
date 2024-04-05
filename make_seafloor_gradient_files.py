"""Generates seafloor gradient NetCDF files from ETOPO15 data using Canny edge detection.

This script processes ETOPO_22 bathymetric data to create seafloor gradient information by employing
Canny edge detection techniques. The primary objective is to analyze and visualize seafloor topography
through gradient and edge detection analysis, highlighting significant features such as underwater
mountains, trenches, and ridges.

The script performs the following operations:
1. Extracts depth, longitude, and latitude data from the 'etopo_22.nc' NetCDF file using a custom 
   library function `extract_etopo15`.
2. Utilizes the extracted data to calculate depth gradients, employing another custom function 
   `create_depth_gradient_nc` for initial processing.
3. Applies Canny edge detection on the depth data to calculate x and y gradients and the overall 
   magnitude of the seafloor's gradient, using a custom function `myCanny` from the `Canny2` module.
4. Writes the results of the Canny edge detection (x gradient, y gradient, and magnitude) to a new 
   NetCDF file using the `write_depth_gradient_nc` function.

Dependencies:
    - canny_lib: Custom library assumed to contain the `extract_etopo15` and `create_depth_gradient_nc` functions.
    - Canny2: Custom module assumed to contain the `myCanny` function for Canny edge detection.
    - netCDF4: For reading NetCDF files.
    - numpy: For numerical operations, especially array manipulations.
    - numpy.ma: For handling masked arrays, particularly useful in masking operations on depth data.
    - xarray: Although not explicitly used in the provided snippet, it is listed as an import, indicating its potential use in extended functionalities or data handling.

Example:
    To use this script, ensure that the 'etopo_22.nc' NetCDF file is located in the specified directory and 
    that all custom libraries (`canny_lib` and `Canny2`) are accessible in the Python environment. The script
    can be executed directly to generate output NetCDF files containing seafloor gradient data in the 
    designated output directory.

Attributes:
    f_name (str): The base filename ('etopo_22') for both input and output NetCDF files, used to identify the dataset.
    base_dir (str): The directory path where input data is located and output NetCDF files will be saved.
    depth (MaskedArray): A masked array of depth values extracted from the ETOPO15 dataset.
    lon_etopo (ndarray): An array of longitude values associated with the ETOPO15 data.
    lat_etopo (ndarray): An array of latitude values associated with the ETOPO15 data.
    x_gradient, y_gradient, magnitude (ndarray): Arrays representing the x and y gradients and the overall gradient magnitude of the seafloor, as calculated by Canny edge detection.

Note:
    The script assumes the existence and correct functioning of the custom libraries (`canny_lib` and `Canny2`). It is also tailored for a specific dataset and output requirements. For operational use, especially in different or more dynamic environments, further modifications and error handling mechanisms may be necessary.
"""
from canny_lib import *
from Canny2 import *
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import xarray as xr

f_name = 'etopo_22'
base_dir = '/Users/rmendels/WorkFiles/seafloor_gradient_new/'
depth, lon_etopo, lat_etopo = extract_etopo15('etopo_22.nc')
create_depth_gradient_nc(f_name, base_dir, depth, lat_etopo, lon_etopo)
x_gradient, y_gradient, magnitude = myCanny(depth, ~depth.mask, sigma = 12.5)
write_depth_gradient_nc(f_name, base_dir, x_gradient, y_gradient, magnitude)

