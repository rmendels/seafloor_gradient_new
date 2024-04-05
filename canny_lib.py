from Canny2 import *
from cartopy import crs
import cmocean
# import cv2
import geoviews as gv
import geoviews.feature as gf
import holoviews as hv
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import xarray as xr

def isleap(year):
    from datetime import date, datetime, timedelta
    try:
        date(year,2,29)
        return True
    except ValueError: return False

def extract_etopo15(file_name, file_base = '/Users/rmendels/WorkFiles/', lat_min = 20.0020833, lat_max = 50.0020833, lon_min = -130.00208333,  lon_max = -105.0020833):
    """Extracts subset of depth, latitude, and longitude data from an ETOPO_22 NetCDF file.

    Args:
        file_name (str): The name of the ETOPO_22 file to extract data from.
        file_base (str): The base directory where the ETOPO_22 file is located. Defaults to '/Users/rmendels/WorkFiles/'.
        lat_min (float): The minimum latitude for the data extraction. Defaults to 20.0020833.
        lat_max (float): The maximum latitude for the data extraction. Defaults to 50.0020833.
        lon_min (float): The minimum longitude for the data extraction. Defaults to -130.00208333.
        lon_max (float): The maximum longitude for the data extraction. Defaults to -105.0020833.

    Returns:
        tuple: A tuple containing the extracted depth array, longitude array, and latitude array.

    Example:
        >>> depth, lon_etopo, lat_etopo = extract_etopo15('etopo_22.nc')
    """
    import numpy as np
    import numpy.ma as ma
    from netCDF4 import Dataset
    nc_file = file_base + file_name
    root = Dataset(nc_file)
    lat = root.variables['lat'][:]
    lon = root.variables['lon'][:]
    lat_min_index = np.argwhere(lat >= lat_min)
    lat_min_index = lat_min_index[0, 0]
    lat_max_index = np.argwhere(lat >= lat_max)
    lat_max_index = lat_max_index[0, 0]
    lon_min_index = np.argwhere(lon <= lon_min)
    lon_min_index = lon_min_index[-1, 0]
    lon_max_index = np.argwhere(lon <= lon_max)
    lon_max_index = lon_max_index[-1, 0]
    lon_etopo = lon[lon_min_index:lon_max_index + 1]
    lat_etopo = lat[lat_min_index:lat_max_index + 1]
    depth = root.variables['z'][lat_min_index:(lat_max_index + 1), lon_min_index:(lon_max_index + 1) ]
    depth = ma.array(depth, mask = (depth >= 0))
    root.close()
    return depth, lon_etopo, lat_etopo




def myCanny(myData, myMask, sigma = 12.5, lower = .8, upper = .9, use_quantiles = True):
    """Applies the Canny edge detection algorithm to input data.

    Uses the Canny edge detection algorithm on the provided data array, utilizing the dataset mask. 
    The function configures the algorithm's sensitivity through the sigma, lower, and upper threshold parameters.


    Args:
        myData (ndarray): Input data for edge detection.
        myMask (ndarray): Boolean mask for the data, where True indicates a valid data point.
        sigma (float): Standard deviation of the Gaussian filter. Defaults to 12.5.
        lower (float): Lower bound for hysteresis thresholding. Defaults to 0.8.
        upper (float): Upper bound for hysteresis thresholding. Defaults to 0.9.
        use_quantiles (bool): Whether to use quantiles for thresholding. Defaults to True.

    Returns:
        x_gradient (MaskedArray): The gradient of the data in the x-direction, masked similarly to input data.
        y_gradient (MaskedArray): The gradient of the data in the y-direction, masked similarly to input data.
        magnitude (MaskedArray): The magnitude of the gradient, masked similarly to input data.

    Example:
        >>> x_grad, y_grad, magnitude = myCanny(data, ~data.mask)
    """
    # because of the way masks operate,  if you read in sst using netcdf4,  then the mask to use is ~sst.mask
    y_gradient, x_gradient, magnitude  = canny2(myData, sigma = sigma, mask = myMask, low_threshold = lower, high_threshold = upper,
                              use_quantiles = use_quantiles)
    x_gradient = ma.array(x_gradient, mask = myData.mask)
    y_gradient = ma.array(y_gradient, mask = myData.mask)
    magnitude = ma.array(magnitude, mask = myData.mask)
    return x_gradient, y_gradient, magnitude

def my_contours(edges):
    """Finds contours in an edge-detected image.

    Uses OpenCV's findContours function to detect contours in a binary edge-detected image. 

    Args:
        edges (ndarray): Binary edge-detected image where edges are marked as True or 1.

    Returns:
        contours (list): A list of contours found in the image, where each contour is represented as an array of points.

    Note:
        Requires OpenCV (cv2) for contour detection. Ensure cv2 is installed and imported as needed.

    Example:
        >>> contours = my_contours(edge_detected_image)
    """
    edge_image = edges.astype(np.uint8)
    contours, hierarchy = cv2.findContours(edge_image ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return(contours)

def contours_to_edges(contours, edge_shape, min_len = 10):
    """Converts contour points into a binary edge image.

    This function iterates through a list of contours and marks corresponding points on a binary
    edge image. Only contours longer than a specified minimum length are processed to filter out
    smaller, potentially less significant features.

    Args:
        contours (list): A list of contour arrays, where each contour is represented by its points.
        edge_shape (tuple): The shape of the output edge image (height, width).
        min_len (int): Minimum length of a contour to be included in the edge image. Defaults to 10.

    Returns:
        contour_edges (numpy.ndarray): A binary edge image with marked contours.
        contour_lens (list): Lengths of all contours found, for further analysis.

    Example:
        >>> edges, lengths = contours_to_edges(contours, image.shape)
    """
    num_contours  = len(contours)
    contour_lens = []
    contour_edges = np.zeros(edge_shape)
    for i in list(range(0, num_contours)):
        contour = contours[i]
        contour_len = contour.shape[0]
        contour_lens.append(contour_len)
        if (contour_len > min_len):
            for ilen in list(range(0, contour_len)):
                xloc = contour[ilen, 0, 1]
                yloc = contour[ilen, 0, 0]
                contour_edges[xloc, yloc] = 1
    return contour_edges, contour_lens

def plot_canny_edges(myData, edges, latitudes, longitudes, title = ' ', fig_size = ([8, 6]) ):
    """Plots Canny edge detection results overlaid on the original data.

    Uses xarray and matplotlib to display the original data and its Canny edges. The edges are plotted
    in grayscale on top of the original data colored by a thermal colormap.

    Args:
        myData (numpy.ndarray): Original data array.
        edges (numpy.ndarray): Binary edge data from Canny edge detection.
        latitudes (numpy.ndarray): Latitude coordinates for the data.
        longitudes (numpy.ndarray): Longitude coordinates for the data.
        title (str): Title for the plot. Defaults to a blank space.
        fig_size (list): Figure size. Defaults to [8, 6].

    Example:
        >>> plot_canny_edges(data, edges, lats, lons, 'Canny Edge Detection', [10, 8])
    """
    myData_xr = xr.DataArray(myData, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'sst')
    myData_xr.values[myData_xr.values < 5.] = 5.
    edges1 = edges.astype(int)
    edges1 = ma.array(edges1, mask = (edges1 == 0))
    edges1_xr = xr.DataArray(edges1, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'edge')
    im1 = myData_xr.plot(cmap = cmocean.cm.thermal)
    im2 = edges1_xr.plot(cmap = plt.cm.gray)
    plt.title(title)
    plt.rcParams["figure.figsize"] = fig_size
    plt.tight_layout()


def plot_bathy(depth, latitudes, longitudes, title = ' ', fig_size = ([10, 8])):
    """Plots bathymetric (seafloor depth) data.

    This function uses xarray and matplotlib to visualize bathymetric data. The depth values are
    displayed using a colormap designed for deep water.

    Args:
        depth (numpy.ndarray): Array of depth values.
        latitudes (numpy.ndarray): Latitude coordinates for the depth data.
        longitudes (numpy.ndarray): Longitude coordinates for the depth data.
        title (str): Title of the plot. Defaults to a blank space.
        fig_size (list): Dimensions of the plot. Defaults to [10, 8].

    Example:
        >>> plot_bathy(depth_data, lat_array, lon_array, 'Bathymetric Data Visualization')
    """
    myData_xr = xr.DataArray(depth, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'depth')
    myData_xr.plot(cmap = cmocean.cm.deep.reversed())



def plot_canny_gradient(my_grad, latitudes, longitudes, title = ' ', fig_size = ([10, 8]) ):
    """Plots the gradient magnitude from Canny edge detection alongside its histogram.

    Visualizes the gradient magnitude as an image and its distribution as a histogram in a side-by-side view.
    Uses xarray for plotting the gradient and matplotlib for histograms.

    Args:
        my_grad (numpy.ndarray): Gradient magnitude array.
        latitudes (numpy.ndarray): Latitude coordinates for the gradient data.
        longitudes (numpy.ndarray): Longitude coordinates for the gradient data.
        title (str): Title for the subplot. Defaults to a blank space.
        fig_size (list): Figure dimensions. Defaults to [10, 8].

    Example:
        >>> plot_canny_gradient(gradient_magnitude, latitudes, longitudes, 'Gradient and Histogram')
    """
    fig, axes = plt.subplots(ncols=2)
    myData_xr = xr.DataArray(my_grad, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'gradient')
    if(my_grad.min() < 0.):
        myData_xr.plot(cmap = cmocean.cm.balance, ax=axes[0])
    else:
        myData_xr.plot(cmap = cmocean.cm.amp, ax=axes[0])
    myData_xr = xr.DataArray(np.abs(my_grad), coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'gradient')
    myData_xr.plot.hist(bins = 100, histtype='step', density = True, stacked = True, cumulative=True, ax=axes[1])
    plt.title('')
    plt.rcParams["figure.figsize"] = fig_size
    plt.tight_layout()
    fig.suptitle(title, y =  1.0)


def plot_canny_contours(myData, edges, contour_lens, latitudes, longitudes, title = ' ', fig_size = ([8, 6]) ):
    """Plots Canny edges and the original data with a histogram of contour lengths.

    Displays the original data with overlaid Canny edges in one subplot and a histogram of contour lengths
    in another subplot, providing insight into the distribution of contour sizes.

    Args:
        myData (numpy.ndarray): Original dataset array.
        edges (numpy.ndarray): Binary edge data from Canny edge detection.
        contour_lens (list): Lengths of detected contours.
        latitudes (numpy.ndarray): Latitude coordinates for the data.
        longitudes (numpy.ndarray): Longitude coordinates for the data.
        title (str): Title for the overall figure. Defaults to a blank space.
        fig_size (list): Dimensions of the figure. Defaults to [8, 6].

    Example:
        >>> plot_canny_contours(data, edges, lengths, lat_array, lon_array,
    """
    fig, axes = plt.subplots(ncols=2)
    myData_xr = xr.DataArray(myData, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'sst')
    myData_xr.values[myData_xr.values < 5.] = 5.
    edges1 = edges.astype(int)
    edges1 = ma.array(edges1, mask = (edges1 == 0))
    edges1_xr = xr.DataArray(edges1, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'edge')
    im1 = myData_xr.plot(cmap = cmocean.cm.thermal, ax=axes[0])
    im2 = edges1_xr.plot(cmap = plt.cm.gray, ax=axes[0])
    plt.hist(contour_lens, bins = [1, 5 , 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150], histtype='bar', density = False)
    #plt.title('')
    plt.rcParams["figure.figsize"] = fig_size
    plt.tight_layout()
    fig.suptitle(title, y =  1.0)


def write_depth_gradient_nc(f_name, base_dir, x_gradient, y_gradient, magnitude):
    """Writes x and y gradients, and magnitude of gradients to an existing NetCDF file.

    This function appends gradient data to an existing NetCDF file for seafloor gradients. It updates
    the file with x and y gradients, along with the magnitude of these gradients.

    Args:
        f_name (str): Filename (without extension) to identify the dataset.
        base_dir (str): Directory path where the NetCDF file is stored.
        x_gradient (numpy.ndarray): Array of x-direction gradients.
        y_gradient (numpy.ndarray): Array of y-direction gradients.
        magnitude (numpy.ndarray): Array representing the magnitude of gradients.

    Example:
        >>> write_depth_gradient_nc('seafloor_dataset', '/path/to/data/', x_grad, y_grad, mag)
    """
    from netCDF4 import Dataset, num2date, date2num
    import numpy as np
    import numpy.ma as ma
    file_name = base_dir + f_name +  '_seafloor_gradient.nc'
    ncfile  = Dataset(file_name, 'a')
    x_grad = ncfile.variables['x_gradient']
    y_grad = ncfile.variables['y_gradient']
    mag_grad = ncfile.variables['magnitude_gradient']
    x_grad[ :, :] = x_gradient[:, :]
    y_grad[ :, :] = y_gradient[:, :]
    mag_grad[ :, :] = magnitude[:, :]
    ncfile.close()


def create_depth_gradient_nc(f_name, base_dir, depths, lats, lons):
    """Creates a new NetCDF file for storing depth gradients along with geographical information.

    Initializes a NetCDF file with dimensions and variables for storing seafloor depth data,
    gradients of depth in the x and y directions, and the magnitude of these gradients.
    Also includes metadata about projection and geographical bounds.

    Args:
        f_name (str): Base filename for the new NetCDF file, indicating the dataset's name.
        base_dir (str): Directory path where the new NetCDF file will be created.
        depths (numpy.ndarray): 2D array of seafloor depths.
        lats (numpy.ndarray): 1D array of latitude values.
        lons (numpy.ndarray): 1D array of longitude values.

    Example:
        >>> create_depth_gradient_nc('new_seafloor_dataset', '/path/to/output/', depth_data, lat_values, lon_values)
    """
    from netCDF4 import Dataset, num2date, date2num
    import numpy as np
    import numpy.ma as ma
    file_name = base_dir + f_name +  '_seafloor_gradient.nc'
    ncfile  = Dataset(file_name, 'w', format = 'NETCDF4')
    latsdim = lats.size
    lonsdim = lons.size
    #Create Dimensions
    latdim = ncfile.createDimension('latitude', latsdim)
    londim = ncfile.createDimension('longitude', lonsdim)
    #Create Variables
    LatLon_Projection = ncfile.createVariable('LatLon_Projection', 'i4')
    latFile = ncfile.createVariable('latitude', 'f4', ('latitude'), zlib = True, complevel = 2)
    lonFile = ncfile.createVariable('longitude', 'f4', ('longitude'), zlib = True, complevel = 2)
    # edges = ncfile.createVariable('edges', 'f4', ('time', 'altitude', 'lat', 'lon'), fill_value = -9999.0, zlib = True, complevel = 2)
    sea_floor_depth = ncfile.createVariable('sea_floor_depth', 'f4', ( 'latitude', 'longitude'), fill_value = -99999.0, zlib = True, complevel = 2)
    x_gradient = ncfile.createVariable('x_gradient', 'f4', ( 'latitude', 'longitude'), fill_value = -9999.0, zlib = True, complevel = 2)
    y_gradient = ncfile.createVariable('y_gradient', 'f4', ('latitude', 'longitude'), fill_value = -9999.0, zlib = True, complevel = 2)
    magnitude_gradient = ncfile.createVariable('magnitude_gradient', 'f4', ('latitude', 'longitude'), fill_value = -9999.0, zlib = True, complevel = 2)
    # int LatLon_Projection ;
    LatLon_Projection.grid_mapping_name = "latitude_longitude"
    LatLon_Projection.earth_radius = 6367470.
    #float lat(lat) ;
    latFile._CoordinateAxisType = "Lat"
    junk = (lats.min(), lats.max())
    latFile.actual_range = junk
    latFile.axis = "Y"
    latFile.grid_mapping = "Equidistant Cylindrical"
    latFile.ioos_category = "Location"
    latFile.long_name = "Latitude"
    latFile.reference_datum = "geographical coordinates, WGS84 projection"
    latFile.standard_name = "latitude"
    latFile.units = "degrees_north"
    latFile.valid_max = lats.max()
    latFile.valid_min = lats.min()
    #float lon(lon) ;
    lonFile._CoordinateAxisType = "Lon"
    junk = (lons.min(), lons.max())
    lonFile.actual_range = junk
    lonFile.axis = "X"
    lonFile.grid_mapping = "Equidistant Cylindrical"
    lonFile.ioos_category = "Location"
    lonFile.long_name = "Longitude"
    lonFile.reference_datum = "geographical coordinates, WGS84 projection"
    lonFile.standard_name = "longitude"
    lonFile.units = "degrees_east"
    lonFile.valid_max = lons.max()
    lonFile.valid_min = lons.min()
    #float edges(lat, lon) ;
    #edges.long_name = "Frontal Edge"
    #edges.missing_value = -9999.
    #edges.grid_mapping = "LatLon_Projection"
    #edges.coordinates = "time altitude lat lon "
    #float sea_floor_depth(lat, lon) ;
    sea_floor_depth.long_name = "Sea Floor Depth"
    sea_floor_depth.standard_name = "sea_floor_depth"
    sea_floor_depth.missing_value = -9999.
    sea_floor_depth.grid_mapping = "LatLon_Projection"
    sea_floor_depth.coordinates = "lat lon "
    sea_floor_depth.units = 'm'
    #float x_gradient(lat, lon) ;
    x_gradient.long_name = "East-West Gradient of sea_floor depth"
    x_gradient.missing_value = -99999.
    x_gradient.grid_mapping = "LatLon_Projection"
    x_gradient.coordinates = "lat lon "
    x_gradient.units = 'm'
    # float y_gradient(lat, lon) ;
    y_gradient.long_name = "North-South Gradient of sea_floor depth"
    y_gradient.missing_value = -9999.
    y_gradient.grid_mapping = "LatLon_Projection"
    y_gradient.coordinates = "lat lon "
    y_gradient.units = 'm'
    # float magnitude( lat, lon) ;
    magnitude_gradient.long_name = "Magnitude of sea_floor depth Gradient"
    magnitude_gradient.missing_value = -9999.
    magnitude_gradient.grid_mapping = "LatLon_Projection"
    magnitude_gradient.coordinates = "lat lon "
    magnitude_gradient.units = 'm'
    ## global
    ncfile.title = "Estimated" + f_name + "sea_floor depth x_gradient, y_gradient and gradient magnitude"
    ncfile.cdm_data_type = "Grid"
    ncfile.Conventions = "COARDS, CF-1.6, ACDD-1.3"
    ncfile.standard_name_vocabulary = "CF Standard Name Table v55"
    ncfile.creator_email = "erd.data@noaa.gov"
    ncfile.creator_name =  "NOAA NMFS SWFSC ERD"
    ncfile.creator_type =  "institution"
    ncfile.creator_url  = "https://www.pfeg.noaa.gov"
    temp = lons.max()
    ncfile.Easternmost_Easting = temp
    temp = lats.max()
    ncfile.Northernmost_Northing = temp
    temp = lons.min()
    ncfile.Westernmost_Easting = temp
    temp = lats.max()
    ncfile.Southernmost_Northing =  temp
    ncfile.geospatial_lat_max = temp
    temp = lats.min()
    ncfile.geospatial_lat_min =  temp
    ncfile.geospatial_lat_resolution = 0.01
    ncfile.geospatial_lat_units = "degrees_north"
    ncfile.geospatial_lon_max = lons.max()
    ncfile.geospatial_lon_min = lons.min()
    ncfile.geospatial_lon_resolution = 0.01
    ncfile.geospatial_lon_units = "degrees_east"
    ncfile.infoUrl = ""
    ncfile.institution = "NOAA ERD"
    ncfile.keywords = ""
    ncfile.keywords_vocabulary = "GCMD Science Keywords"
    ncfile.summary = '''Estimated sea_floor depth  x-gradient, y-gradient and gradient magnitude
    using the Python scikit-image canny algorithm  with sigma = 12.5,x-gradient, y-gradient and gradient magnitude.
    '''
    ncfile.license = '''The data may be used and redistributed for free but is not intended
    for legal use, since it may contain inaccuracies. Neither the data
    Contributor, ERD, NOAA, nor the United States Government, nor any
    of their employees or contractors, makes any warranty, express or
    implied, including warranties of merchantability and fitness for a
    particular purpose, or assumes any legal liability for the accuracy,
    completeness, or usefulness, of this information.
    '''
    history = 'created from ' + f_name + 'using python scikit-image canny algorithm, sigma = 10'
    ncfile.history = history
    longitude[:] = lons[:]
    latitude[:] = lats[:]
    sea_floor_depth[:, :] = depths[:, :]
    ncfile.close()






