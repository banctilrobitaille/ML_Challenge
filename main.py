from commons.models.datasetFactory import DatasetFactory
from knn.model import Model

if __name__ == '__main__':
    training_dataset = DatasetFactory.create_dataset_from_files(data_type='training')
    knnModel = Model().train_with_dataset(training_dataset)
    number_of_correctly_estimated_label = 0
    number_of_classification_made = 0.0
    goodDict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
    }

    total = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
    }

    for data_instance in training_dataset.data_instances:
        total[data_instance.label] += 1
        number_of_classification_made += 1
        estimatedLabel = knnModel.classify(data_instance, 15)
        if estimatedLabel == data_instance.label:
            number_of_correctly_estimated_label += 1
            goodDict[data_instance.label] += 1

    print("Success rate: " +
          str((float(number_of_correctly_estimated_label) / float(number_of_classification_made) * 100)) + "%")
    print("Good predication: " + str(goodDict))
    print("Total predication: " + str(total))
