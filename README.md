# Swimming Pool Detection from Satellite Images

## Overview
This project involves detecting swimming pools in satellite images by analyzing the images for certain color and shape characteristics typically associated with swimming pools. The methodology includes segmenting the images, detecting specific colors, identifying pool-like structures, and then converting these detections into geographical coordinates.

### Methodology

1. **Image Segmentation**: 
   - Satellite images often cover large areas and come in high resolutions. 
   - To efficiently process these images, we first split them into smaller segments.

2. **Color Filtering**:
   - Initially, the images are in Near-Infrared (NIR) format color, which requires specific processing techniques.
   - We apply color filtering to detect shades corresponding to swimming pools from HSV transformed images.
   - Specific tones of blue, yellow, orange, and red are targeted as these are common colors when transforming RGB from NIR to BGR.

3. **Shape Detection**:
   - We focus on finding circular or rectangular shapes, which are common shapes for swimming pools.
   - This involves contour detection in the segmented images.

4. **Coordinate Calculation**:
   - After detecting these pools in the segmented images, we calculate their pixel coordinates.
   - These pixel coordinates are then converted into geographical coordinates using the image's metadata and coordinate transformation techniques.

### Tools and Technologies

- **OpenCV**: Used for image processing tasks such as color conversion, masking, contour detection, and shape analysis.
- **Python Libraries**:
  - `glob` for file handling.
  - `numpy` for numerical operations.
  - `matplotlib` for image visualization.
  - `pyproj` for coordinate system transformations.
- **Coordinate Transformation**:
  - Conversion from the Lambert-93 projection system (commonly used in French GIS data) to the WGS84 system, enabling the integration of detected pool locations with global mapping tools.

### Workflow

1. **File Handling**:
   - The program reads `.jp2` satellite images and their corresponding `.tab` files containing metadata.

2. **Image Reading and Preprocessing**:
   - Images are read and converted to the RGB color space for further processing.

3. **Detection Algorithm**:
   - The main algorithm involves segmenting the full image, applying color filters, detecting shapes, and then calculating the pool centers in each segment.

4. **Conversion of Pool Centers to Geographical Coordinates**:
   - The pool center pixel coordinates from each segment are adjusted to their position in the full image.
   - Using the metadata from `.tab` files, these pixel coordinates are transformed into geographical coordinates (longitude and latitude).

### Usage

- The detection process can be initiated on a set of satellite images, where it will process each image, detect pools, and output their geographical locations.
- This system can be particularly useful in urban planning, recreational facility management, and geographical analysis.

### Future Enhancements

- Improving the accuracy of color and shape detection to reduce false positives.
- Optimizing the segmentation process to handle different image sizes and resolutions more efficiently.
- Integrating machine learning techniques for more robust pool detection.
