from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import errno


# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min_, max_, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min_) / float(max_ - min_)) * 255).astype(dtype)


# ==============================================================================
#                                                          BIRDS_EYE_POINT_CLOUD
# ==============================================================================
def birds_eye_point_cloud(points, idx,
                          side_range=(-10, 10),
                          fwd_range=(-10, 10),
                          res=0.1,
                          min_height=-2.73,
                          max_height=1.27,
                          saveto=None):
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices] / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (x_lidar[indices] / res).astype(np.int32)  # y axis is -x in LIDAR
    # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0] / res))
    y_img -= int(np.floor(fwd_range[0] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values, min_=min_height, max_=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values  # -y because images start from top left
    # return im
    # Convert from numpy array to a PIL image
    # im2=np.asarray(im)
    im = Image.fromarray(im)

    # SAVE THE IMAGE
    if saveto is not None:
        biv_path = os.path.join(saveto, 'bird_eye_view')
        dir_i = os.path.join(biv_path, str(idx)) + '.png'
        im.save(dir_i)
    else:
        im.show()
        plt.show()


def point_cloud_to_panorama(points,
                            v_res=0.42,
                            h_res=0.35,
                            v_fov=(-24.9, 2.0),
                            d_range=(0, 100),
                            y_fudge=3
                            ):

    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    r_points = points[:, 3]
    d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    # d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2) # abs distance

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_fov_total = -v_fov[0] + v_fov[1]

    # CONVERT TO RADIANS
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = -(np.arctan2(z_points, d_points) / v_res_rad)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total / v_res) / (v_fov_total * (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0] * (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below + h_above + y_fudge))

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # CLIP DISTANCES
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])

    # CONVERT TO IMAGE ARRAY
    img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
    img[y_img, x_img] = scale_to_255(d_points, min_=d_range[0], max_=d_range[1])

    return img


def lidar_to_2d_front_view(points, id, v_res, h_res, v_fov, y_fudge=0.0, saveto=None):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.
    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.
            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) == 2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3]  # Reflectance

    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar) / h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar) / v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min  # Shift
    x_max = 360.0 / h_res  # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res  # theoretical min y value based on sensor specs
    y_img -= y_min  # Shift
    y_max = v_fov_total / v_res  # Theoretical max x value after shifting

    y_max += y_fudge
    # Fudge factor if the calculations based on spec sheet do not match the range of angles collected by in the data.

    range = np.sqrt((np.power(x_lidar,2) + np.power(y_lidar,2)+np.power(z_lidar,2)))
    XYZRUV = {'xyz': points, 'range_':range, 'u': x_img, 'v': y_img}

    if saveto is not None:
        data_path = os.path.join(saveto, 'Processed_Data/')
        mkdir_p(data_path)
        name =  os.path.join(data_path, str(id)) + '.txt'
        file_txt = open(name, "wb")
        pickle.dump(XYZRUV, file_txt)
        file_txt.close()

    return XYZRUV


def Lidar_2d_vis_save(XYZRUV, id, v_res, h_res, v_fov, cmap="jet", val="depth", saveto=None, show=False):

    pc = XYZRUV['xyz']
    u = XYZRUV['u']
    v = XYZRUV['v']
    r = XYZRUV['range_']

    x_lidar = pc[:, 0]
    y_lidar = pc[:, 1]
    z_lidar = pc[:, 2]
    r_lidar = pc[:, 3]  # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pixel_values = r_lidar
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar

    v_fov_total = -v_fov[0] + v_fov[1]

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_max = 360.0 / h_res  # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res  # theoretical min y value based on sensor specs
    y_max = v_fov_total / v_res  # Theoretical max x value after shifting

    # ____________________________________________________________________________________________________PLOT THE IMAGE
    # cmap = "jet"  # Color map to use
    dpi = 100  # Image resolution
    fig, ax = plt.subplots(figsize=(x_max / dpi, y_max / dpi), dpi=dpi)
    ax.scatter(u, v, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_facecolor((0, 0, 0))  # Set regions with no points to black
    ax.axis('scaled')  # {equal, scaled}
    # ax.xaxis.set_visible(False)  # Do not draw axis tick marks
    # ax.yaxis.set_visible(False)  # Do not draw axis tick marks
    plt.xlim([0, x_max])  # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])  # prevent drawing empty space outside of vertical FOV

    if saveto is not None:
        project_path = os.path.join(saveto, 'Spherical_Image/')
        mkdir_p(project_path)
        _dir = os.path.join(project_path, str(id)) + '.png'
        fig.savefig(_dir, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    if show:
        fig.show()
        plt.show()

    plt.close('all')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise