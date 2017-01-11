from commons.models.datasetFactory import DatasetFactory

if __name__ == '__main__':
    dataset = DatasetFactory().createDatasetFromOnlineResource()
    print dataset.flattenDataArray

