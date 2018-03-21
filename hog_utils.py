import cv2
from skimage.feature import hog


def compute_hog_features(image, orient, pixels_per_cell, cells_per_block):
    """Computes hog features from a single image.

    :param image: image in RGB format
    :param orient: number of HoG orientations
    :param pixels_per_cell: number of pixels per cell
    :param cells_per_block: number of HoG cells per block
    :returns: a list of feature vectors.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return hog(
        grayscale,
        orientations=orient,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        transform_sqrt=True,
        feature_vector=True
    )


def extract_features(images, orient, pixels_per_cell, cells_per_block):
    """Extracts hog features from the given list of images.

    :param images: a list of images in RGB format
    :param orient: number of HoG orientations
    :param pixels_per_cell: number of pixels per cell
    :param cells_per_block: number of HoG cells per block
    :returns: a list of feature vectors.
    """
    return [compute_hog_features(image, orient, pixels_per_cell, cells_per_block) for image in images]
