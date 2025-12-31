import numpy as np

er = 6378137.0  # average earth radius at the equator

def latlon_to_mercator(lat, lon, scale):
    """converts lat/lon coordinates to mercator coordinates using mercator scale"""
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
    return mx, my


def mercator_to_latlon(mx, my, scale):
    """converts mercator coordinates using mercator scale to lat/lon coordinates"""
    lon = mx * 180.0 / (scale * np.pi * er)
    lat = 360.0 / np.pi * np.arctan(np.exp(my / (scale * er))) - 90.0
    return lat, lon


def mercator_to_ego(lat, lon, scale):
    """
    Converts latitude and longitude to Mercator coordinates.
    
    Parameters:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        scale (float): Mercator scale factor.
        
    Returns:
        mx (float): Mercator x-coordinate.
        my (float): Mercator y-coordinate.
    """
    mx = (lon / 180.0) * (scale * np.pi * er)
    my = scale * er * np.log(np.tan(np.pi / 360.0 * (lat + 90.0)))
    return mx, my


def lat_to_scale(lat):
    """compute mercator scale from latitude"""
    scale = np.cos(lat * np.pi / 180.0)
    return scale


def post_process_points(points_in):
    R = np.array([[1,  0,  0, 0], 
                  [0, -1,  0, 0], 
                  [0,  0, -1, 0], 
                  [0,  0,  0, 1]]
                )

    poses = []
    for i in range(len(points_in)):
        # if there is no data => no points
        if not len(points_in[i]):
            poses.append([])
            continue
        P = points_in[i]
        poses.append(np.matmul(R, P.T).T)

    return poses