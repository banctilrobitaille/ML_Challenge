class DatasetHelper:
    WIDTH_INDEX = 0
    HEIGHT_INDEX = 1

    @staticmethod
    def extract_features_from_image(image, threshold=1):
        imageWidth = image.shape[DatasetHelper.WIDTH_INDEX]
        imageHeight = image.shape[DatasetHelper.HEIGHT_INDEX]

        return {"x1": filter(lambda pixel: pixel >= threshold, image[:imageWidth / 2, :imageHeight / 2]).count(),
                "x2": filter(lambda pixel: pixel >= threshold, image[imageWidth / 2:, :imageHeight / 2]).count(),
                "x3": filter(lambda pixel: pixel >= threshold, image[:imageWidth / 2, imageHeight / 2:]).count(),
                "x4": filter(lambda pixel: pixel >= threshold, image[imageWidth / 2:, imageHeight / 2:]).count()}
