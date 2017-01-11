import numpy as np


class DatasetHelper:
    WIDTH_INDEX = 0
    HEIGHT_INDEX = 1

    @staticmethod
    def extract_features_from_image(image, threshold=1):
        image_width = image.shape[DatasetHelper.WIDTH_INDEX]
        image_height = image.shape[DatasetHelper.HEIGHT_INDEX]

        return {"x1": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[:image_width / 2, :image_height / 2]).flat)),
                "x2": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[image_width / 2:, :image_height / 2]).flat)),
                "x3": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[:image_width / 2, image_height / 2:]).flat)),
                "x4": len(filter(lambda pixel: pixel >= threshold,
                                 np.array(image[image_width / 2:, image_height / 2:]).flat))}
