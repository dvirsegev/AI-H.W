import csv
from math import floor
import knn
import decision_tree
import naive_base


ATTRIBUTES = []

def readfiles(filename):
    """
    :return: the data examples.
    """
    global ATTRIBUTES
    f = open(filename)
    data = list(csv.reader(f, delimiter='\t'))
    ATTRIBUTES = data[0]
    data.pop(0)
    f.close()
    return list(data)


def k_fold_cross_validation(data, k):
    """
    :param data: our data examples.
    :param k: k is how many times we want to split our data into example and testing.
    :return: list that contains tuples of data and testing.
    """
    attr_row = data[0]
    length = len(data)
    folds = []  # train, test data
    for i in range(k):
        start_idx = floor(length * i / 5)
        end_idx = floor(length * (i + 1) / 5)
        test_k = [attr_row] + data[start_idx: end_idx]
        train_k = data[0:start_idx] + data[end_idx: length]
        folds.append((train_k, test_k))
    return folds


def writing_results(results,filename):
    """
    :param results: the result
    :param filename: the name of the file
    :return: write the result
    """
    f = open(filename, "w+")
    f.write("<DT_accuracy>" + str(results[0]) + "\t<KNN_accuracy>" + str(results[1]) +
            "\t<naiveBase_accuracy>" + str(results[2]))
    f.flush()
    f.close()



def start_testing():
    """
    run all the algorithms.
    :return: the results.
    """
    global ATTRIBUTES
    data = readfiles("dataset.txt")
    results = []
    folds = k_fold_cross_validation(data, k=5)
    results.append(decision_tree.start_algorithm(data,folds,ATTRIBUTES))
    results.append(knn.knn_algorithm(folds, k=5))
    results.append(naive_base.naive_base_result(folds))
    writing_results(results,"accuracy.txt")

def real_test():
    """
        run all the algorithms (real test).
        :return: the results.
        """
    global ATTRIBUTES
    data = readfiles("train.txt")
    test = readfiles("test.txt")
    results = []
    folds = (data,test)
    results.append(decision_tree.start_algorithm(data, folds, ATTRIBUTES))
    results.append(knn.knn_algorithm(folds ,k=5))
    results.append(naive_base.naive_base_result(folds))
    writing_results(results, "output.txt")

if __name__ == '__main__':
    #start_testing()
    real_test()
