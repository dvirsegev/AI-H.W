import csv
from math import floor


def separate_by_class(dataset):
    """
    :param dataset: train data.
    :return: dic that the key is the class(yes/no) and the value is the data example.
    """
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector[:-1])
    return separated


def calculate_max_predict(list_of_yes_probalities, list_of_no_probalities):
    """

    :param list_of_yes_probalities: how much yes there is the data.
    :param list_of_no_probalities:  how much no there is in the data.
    :return: our predict if it should be yes or no .
    """
    yes_result = 1
    no_result = 1
    for x in list_of_yes_probalities:
        if x != 0:
            yes_result = yes_result * x
    for x in list_of_no_probalities:
        if x != 0:
            no_result = no_result * x
    if (yes_result > no_result):
        return "yes"
    else:
        return "no"


def calculate_prob(train, x_test):
    """

    :param train: the train data
    :param x_test: the example we testing.
    :return: we return if the specific data is yes or no answer.
    """
    size_no = len(train["no"])
    size_yes = len(train["yes"])
    list_of_yes_probalities = list()
    list_of_no_probalities = list()
    list_of_no_probalities.append(size_no / (size_no + size_yes))
    list_of_yes_probalities.append(size_yes / (size_no + size_yes))
    # index is the place in the array, like 0 is the first attribute .
    for index in range(len(x_test)):
        # for loop all the dic of the train .
        for key, list_in_key in train.items():
            count = 0
            # for loop all the values in the same key
            for row in list_in_key:
                if row[index] == x_test[index]:
                    count += 1
            if key == "yes":
                list_of_yes_probalities.append(count / len(list_in_key))
            else:
                list_of_no_probalities.append(count / len(list_in_key))
    return calculate_max_predict(list_of_yes_probalities, list_of_no_probalities)


def naive_base_result(dataset):
    """

    :param dataset: dic with keys yes/no and the row that belong each key.
    :return:
    """

    list_of_results = list()
    for train_data, test_data in dataset:
        correct = 0
        train_data = separate_by_class(train_data)
        # pick train data
        for test_line in test_data:
            excpeded = test_line[-1]
            predict = calculate_prob(train_data, test_line[:-1])
            if predict == excpeded:
                correct += 1
        list_of_results.append(correct / len(test_data))
    avg = sum(list_of_results) / len(list_of_results)
    avg = round(avg * 100 , 2)
    return avg

