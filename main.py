from commons.models.datasetFactory import DatasetFactory

if __name__ == '__main__':
    train, label = DatasetFactory().createDatasetFromFiles()
    print train[1]
