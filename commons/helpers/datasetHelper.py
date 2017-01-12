import numpy as np
from skimage.filters import threshold_otsu


class DatasetHelper:
    WIDTH_INDEX = 0
    HEIGHT_INDEX = 1

    @staticmethod
    def extract_features_from_image(image, threshold=1):
        image_width = image.shape[DatasetHelper.WIDTH_INDEX]
        image_height = image.shape[DatasetHelper.HEIGHT_INDEX]

        image = image >= threshold_otsu(image)

        return {"x1": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[:image_width / 3, :image_height / 3]).flat)),
                "x2": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[image_width / 3:(image_width / 3) * 2, :image_height / 3]).flat)),
                "x3": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[(image_width / 3) * 2:image_width, :image_height / 3]).flat)),
                "x4": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[:image_width / 3, image_height / 3:(image_height / 3) * 2]).flat)),
                "x5": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[image_width / 3:(image_width / 3) * 2,
                                          image_height / 3:(image_height / 3) * 2]).flat)),
                "x6": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[(image_width / 3) * 2:image_width,
                                          image_height / 3:(image_height / 3) * 2]).flat)),
                "x7": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[:image_width / 3, (image_height / 3) * 2:]).flat)),
                "x8": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[image_width / 3:(image_width / 3) * 2, (image_height / 3) * 2:]).flat)),
                "x9": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[(image_width / 3) * 2:image_width, (image_height / 3) * 2:]).flat))}
