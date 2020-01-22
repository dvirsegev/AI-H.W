def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences."""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length.")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def knn_algorithm(folds, k):
    """

    :param folds: list of sets that hold the tests and the trains data/
    :param k: the parameters of KNN
    :return: the  predict average of the algorithm
    """

    list_of_results = list()
    for train, test in folds:
        good_predict = 0
        # pick train data
        for node_test in test:
            list_of_distance = list()
            # pass all the train_data
            for sample in train:
                # ignore yes/no in the data
                str_sample = ' '.join([str(elem) for elem in sample[:-1]])
                str_test = ' '.join([str(elem) for elem in node_test[:-1]])
                dist = hamming_distance(str_sample, str_test)
                list_of_distance.append((sample, dist))
            # sort the list according to the distance
            list_of_distance.sort(key=lambda tup: tup[1])
            neighbors = list()
            for i in range(k):
                neighbors.append(list_of_distance[i][0])
            # pick only the last element( yes or no)
            neighbors = [row[-1] for row in neighbors]
            # pick the predict that shows most of the time.
            predict = max(set(neighbors), key=neighbors.count)
            # if we right on the predict, add the count by 1.
            if node_test[-1] == predict:
                good_predict += 1
        list_of_results.append(good_predict / len(test))
    # cal the avg predict of the algorithm.
    avg = sum(list_of_results) / len(list_of_results)
    avg = round(avg * 100, 2)
    return avg

