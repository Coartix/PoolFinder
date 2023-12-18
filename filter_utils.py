import os
import glob
import numpy as np
import cv2
import folium
from pyproj import Proj, transform


def grab_files(file_path):
    files_grabbed = []
    files_info = []
    path_check_include = ''

    path_ign = file_path
    files_grabbed = glob.glob(os.path.join(path_ign, '*.jp2'))
    files_info = glob.glob(os.path.join(path_ign, '*.tab'))
    files_grabbed = [x for x in files_grabbed if path_check_include in x]
    files_info = [x for x in files_info if path_check_include in x]

    print("- nombre d'images:", len(files_grabbed), ', nb files info coord:', len(files_info))
    return files_grabbed, files_info

def read_one(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def split_image(img, nb_split):
    img_split = []
    h, w, c = img.shape
    for i in range(nb_split):
        for j in range(nb_split):
            img_split.append(img[i*h//nb_split:(i+1)*h//nb_split, j*w//nb_split:(j+1)*w//nb_split, :])
    return img_split

def detect_swimming_pools(image):
    # Convert to HSV color space for easier color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Orange color range
    lower_orange = np.array([10, 90, 90])
    upper_orange = np.array([20, 255, 255])

    # Red color range (might need two parts)
    # Lower red
    lower_red1 = np.array([0, 75, 75])
    upper_red1 = np.array([10, 255, 255])

    # Create masks for each color
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = cv2.bitwise_or(mask_yellow, mask_orange)
    mask = cv2.bitwise_or(mask, mask_red1)
    #mask = cv2.bitwise_or(mask, mask_red2)
    
    pool_centers = []
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter for rectangular or circular shapes
    for cnt in contours:

        # Erase too small contours
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 3.2 or radius > 13:
            continue

        # Check for rectangular or circular shapes and get center
        pool_centers.append((int(x), int(y)))

    return pool_centers

def process_and_save_coordinates(img, nb_split, original_width, original_height):
    img_split = split_image(img, nb_split)
    all_pool_centers = []

    for i in range(nb_split):
        for j in range(nb_split):
            split_img = img_split[i * nb_split + j]
            split_pool_centers = detect_swimming_pools(split_img)

            # Adjust coordinates based on split offset
            split_x_offset = j * original_width // nb_split
            split_y_offset = i * original_height // nb_split
            adjusted_centers = [(x + split_x_offset, y + split_y_offset) for x, y in split_pool_centers]

            all_pool_centers.extend(adjusted_centers)

    return all_pool_centers


def get_info_coords(fname_info):
    """Get the coordinates of the 4 corners of the image from the .tab file"""
    def_table = ""

    file1 = open(fname_info, 'r')
    lines = file1.readlines()
    file1.close()

    for line in lines:
        def_table += line + '\n'

    # Parse the definition table
    lines = def_table.split('\n')
    lines = [line for line in lines if line != '']
    lines = [line for line in lines if line[0] == '(']

    # Get the coordinates of the 4 corners of the image
    coords = []
    pixel_coords = []
    for line in lines:
        coords.append(line.split('(')[1].split(')')[0].split(','))
        pixel_coords.append(line.split('(')[2].split(')')[0].split(','))
    coords = np.array(coords).astype(np.float32)
    pixel_coords = np.array(pixel_coords).astype(np.float32)

    return coords

def pixel_to_local(x_pixel, y_pixel, img_width, img_height, coords):
    top_left = coords[0]  # pixel (0,0)
    bottom_right = coords[2]  # pixel (img_width, img_height)

    # Calculate scale factors
    x_scale = (bottom_right[0] - top_left[0]) / img_width
    y_scale = (bottom_right[1] - top_left[1]) / img_height

    # Convert pixel to local coordinates
    x_local = top_left[0] + x_pixel * x_scale
    y_local = top_left[1] + y_pixel * y_scale

    return x_local, y_local


# Function to convert Lambert-93 to WGS84
def lambert_to_wgs84(x_lambert, y_lambert, lambert_proj, wgs84_proj):
    lon, lat = transform(lambert_proj, wgs84_proj, x_lambert, y_lambert)
    return lon, lat

def get_wgs84_poolcenters(img_width, img_height, pool_centers, coords, lambert_proj, wgs84_proj):
    wgs84_pool_centers = []
    for x_pixel, y_pixel in pool_centers:
        x_local, y_local = pixel_to_local(x_pixel, y_pixel, img_width, img_height, coords)
        lon, lat = lambert_to_wgs84(x_local, y_local, lambert_proj, wgs84_proj)
        wgs84_pool_centers.append((lon, lat))

    return wgs84_pool_centers