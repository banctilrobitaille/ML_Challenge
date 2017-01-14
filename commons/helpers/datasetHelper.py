import numpy as np
from skimage.filters import threshold_otsu


class DatasetHelper:
    @staticmethod
    def extract_features_from_image(image, number_of_features=196, threshold=1):
        features_value = []

        binary_image = image >= threshold_otsu(image)
        blocks = DatasetHelper.__split_image_into_blocks(binary_image, number_of_blocks=number_of_features)

        for block in blocks:
            features_value.append(len(filter(lambda pixel: pixel >= threshold, block.flat)))

        return dict(enumerate(features_value))

    @staticmethod
    def __split_image_into_blocks(image, number_of_blocks):
        blocks = []
        block_divided_image = map(lambda sub_array: np.array_split(sub_array, np.sqrt(number_of_blocks), axis=1),
                                  np.array_split(image, np.sqrt(number_of_blocks)))

        for row in block_divided_image:
            for block in row:
                blocks.append(np.array(block))

        return blocks
