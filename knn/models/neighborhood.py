from collections import OrderedDict, Counter


class Neighborhood(object):
    __number_of_neighbors = 0
    __neighbors = OrderedDict()

    def __init__(self, number_of_neighbors=3):
        self.__number_of_neighbors = number_of_neighbors

    def accept(self, new_neighbor, new_neighbor_distance):
        if len(self.__neighbors) < self.__number_of_neighbors:
            self.__neighbors[new_neighbor] = new_neighbor_distance
        else:
            self.__neighbors = OrderedDict(sorted(self.__neighbors.items(), key=lambda t: t[1], reverse=True))
            for neighbor, distance in self.__neighbors.iteritems():
                if new_neighbor_distance < distance:
                    del self.__neighbors[neighbor]
                    self.__neighbors[new_neighbor] = new_neighbor_distance
                    return True

    def get_the_label_trend(self):
        labels = Counter(map(lambda neighbor: neighbor.label, self.__neighbors.keys()))
        return OrderedDict(sorted(labels.items(), key=lambda t: t[1], reverse=True)).keys()[0]
