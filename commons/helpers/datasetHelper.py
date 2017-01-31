import numpy as np
from skimage.filters import threshold_otsu


class DatasetHelper:
    @staticmethod
    def extract_features_from_image(image, with_data_normalization, with_threshold, number_of_features):
        features_value = []

        if with_threshold:
            binary_image = image >= threshold_otsu(image)
            blocks = DatasetHelper.__split_image_into_blocks(binary_image, number_of_blocks=number_of_features)
            for block in blocks:
                features_value.append(len(filter(lambda pixel: pixel >= 1, block.flat)))
        else:
            features_value = DatasetHelper.__split_image_into_blocks(image, number_of_blocks=number_of_features)

        if with_data_normalization:
            features_value = DatasetHelper.normalize(np.array(features_value))

        return features_value

    @staticmethod
    def __split_image_into_blocks(image, number_of_blocks):
        blocks = None
        if number_of_blocks != (image.shape[0] * image.shape[1]):
            blocks = []
            block_divided_image = map(lambda sub_array: np.array_split(sub_array, np.sqrt(number_of_blocks), axis=1),
                                      np.array_split(image, np.sqrt(number_of_blocks)))

            for row in block_divided_image:
                for block in row:
                    blocks.append(np.array(block))
        else:
            blocks = image.flatten()
        return blocks

    @staticmethod
    def normalize(input_vector):
        max_value = np.max(input_vector)
        return np.array(map(lambda feature: feature / float(max_value), input_vector))
