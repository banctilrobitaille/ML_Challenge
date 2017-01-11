from commons.models.datasetFactory import DatasetFactory

if __name__ == '__main__':
    dataset = DatasetFactory.create_dataset_from_files()
    print dataset.data_instances[0].features
