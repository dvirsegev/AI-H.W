from math import log
import copy

ATTRIBUTES = list()
yes_table_count = 0
no_table_count = 0
map_attributes = {}


class Node:
    def __init__(self):
        self.links = {}
        self.name_of_attr = ""
        self.gain = 0
        self.index = -1
        self.class_type = ""

    def add_subtree(self, attr_value, node):
        """
        Adds a branch with value attr_value to given node
        :param attr_value: value of the node
        :param node: our node child.
        """
        self.links[attr_value] = node

    def set_setting(self, attr):
        """

        :param attr: the attribute.
        :return: the Node.
        """
        self.name_of_attr = attr
        self.index = ATTRIBUTES.index(attr)
        return self

    def predict(self, tests, default):
        """
        :param tests: our tests data
        :param default: Node Default.
        :return:
        """
        count_correct = 0
        for test in tests:
            prediction = self.classify(test, default)
            if prediction == test[-1]:
                count_correct += 1
        return count_correct / len(tests)

    def classify(self, sample, default):
        """
        Traverse the tree and classify the sample
        :param default: Node Default.
        :param sample: sample from data
        :return: yes or no
        """
        node = self
        while node.class_type is "":
            current_sample_attr_val = sample[node.index]
            #  If attr_val not belongs to a branch in tree, return default choice based on the data the tree used
            if current_sample_attr_val not in node.links.keys():
                return default
            node = node.links[current_sample_attr_val]
        return node.class_type

    def write_to_file(self, depth):
        """
        Write tree content to file
        :param node: our tree.
        :param depth: recursion depth(will help with tab printing
        """
        with open("tree.txt", 'a+') as f:
            sorted_keys = sorted(self.links)
            for attr in sorted_keys:
                for d in range(depth):
                    f.write('\t')
                if depth > 0:
                    f.write('|')
                f.write(self.name_of_attr + '=' + attr)
                child_node = self.links[attr]
                #  Leaf:
                if child_node.class_type == "yes" or child_node.class_type == "no":
                    f.write(':' + child_node.class_type + '\n')
                    f.flush()
                else:
                    f.write('\n')
                    f.flush()
                    child_node.write_to_file(depth + 1)


def same_class_equal(train_data):
    """

    :param train_data: our train data
    :return: true if everyone is the same class, false otherwise.
    """
    first_time_yes = False
    first_time_no = False
    for data in train_data:
        if data[-1] == "yes":
            first_time_yes = True
        if data[-1] == "no":
            first_time_no = True
        if first_time_no and first_time_yes:
            return False
    return True


def calculate_entropy(yes_result, no_result):
    """
    :param yes_result: yes_result
    :param no_result:  no_result
    :return: calculate the entropy .
    """
    length = yes_result + no_result
    if (length == 0):
        return 0
    first_break = yes_result / length
    second_break = no_result / length
    entopy_start = 0
    entropy_end = 0
    if first_break != 0:
        entopy_start = -first_break * log(first_break, 2)
    if second_break != 0:
        entropy_end = second_break * log(second_break, 2)
    return entopy_start - entropy_end


def setMapAttributes(data):
    """
    :param data: our data examples.
    :return: None.
    """
    global map_attributes, ATTRIBUTES
    # dic that the key is the name of the attr and value is all the types.
    for attributes in ATTRIBUTES:
        map_attributes[attributes] = set()
        for example in data:
            # go all the examples and check about the spesific type.
            map_attributes[attributes].add(example[ATTRIBUTES.index(attributes)])


def gain(train_data, node):
    """
    :param train_data: our data
    :param node: our node for testing .
    :return: the gain of the attribute .
    """
    global yes_table_count, no_table_count, map_attributes
    regular_entropy = calculate_entropy(yes_table_count, no_table_count)
    result_attr_type = {}
    map_aie = {}
    x = map_attributes
    for type_of_attr in map_attributes[node.name_of_attr]:
        yes_result = 0
        no_result = 0
        # cal for each type of the attribute his yes_count and no_count.
        examples = []
        for example in train_data:
            if example[node.index] == type_of_attr:
                examples.append(example)
        for example in examples:
            if example[-1] == "yes":
                yes_result += 1
            else:
                no_result += 1
        # aie == average information entropy.
        map_aie[type_of_attr] = (yes_result + no_result) / (yes_table_count + no_table_count)
        result_attr_type[type_of_attr] = calculate_entropy(yes_result, no_result)
    aie = 0
    for type_of_attr in result_attr_type.keys():
        aie = aie + (map_aie[type_of_attr] * result_attr_type[type_of_attr])
    node.gain = regular_entropy - aie
    return node


def choose_best_attribute(attributes, examples):
    """
    :param attributes: all the  attributes.
    :param examples: our data.
    :return: pick the bst attribute.
    """
    attr_node_list = list()
    # for each attributes set a node and cal the gain of his.
    for attr in attributes:
        node = Node()
        # set index for the spesific attribiute.
        node.set_setting(attr)
        attr_node_list.append(gain(examples, node))
    # pick the best attributes.
    max = -1
    max_node = Node()
    for node in attr_node_list:
        if node.gain > max:
            max = node.gain
            max_node = node
    return max_node


def dtl(train_data, attributes, default):
    """
    :param train_data: our data examples.
    :param attributes: all the attributes.
    :param default: default.
    :return: it's recusion function, in the end we return tree .
    """

    global yes_table_count, no_table_count, map_attributes
    yes_table_count = len([data for data in train_data if data[-1] == "yes"])
    no_table_count = len([data for data in train_data if data[-1] == "no"])
    if len(train_data) == 0:
        return default
    if same_class_equal(train_data):
        tree = Node()
        tree.class_type = train_data[0][-1]
        return tree  # the classification(yes or no)
    if len(attributes) == 0:
        return default
    tree = choose_best_attribute(attributes, train_data)
    for value_type in map_attributes[tree.name_of_attr]:
        examples_per_value = [x for x in train_data if x[tree.index] == value_type]
        sub_attributes = copy.deepcopy(attributes)
        sub_attributes.remove(tree.name_of_attr)
        sub_tree = dtl(examples_per_value, sub_attributes, default)
        tree.add_subtree(value_type, sub_tree)
    return tree


def start_algorithm(data, folds, attributes):
    """

    :param attributes: our attributes.
    :param folds: our data examples.
    :return: how much accuracy is the algorithm
    """
    global ATTRIBUTES, map_attributes
    ATTRIBUTES = attributes[:-1]
    # create and write the big tree .
    big_tree = create_and_write_big_tree(data)
    big_tree.write_to_file(0)
    list_of_results = list()
    global yes_table_count, no_table_count
    for train_data, test_data in folds:
        map_attributes = {}
        # without can_eat row (yes/no row)
        attributes = ATTRIBUTES
        # set the default.
        yes_table_count = len([data for data in train_data if data[-1] == "yes"])
        no_table_count = len([data for data in train_data if data[-1] == "no"])
        if yes_table_count > no_table_count:
            default = Node()
            default.class_type = "yes"
        else:
            default = Node()
            default.class_type = "no"
        setMapAttributes(train_data)
        tree = dtl(train_data, attributes, default)
        list_of_results.append(tree.predict(test_data, default))
    predict = sum(list_of_results) / len(list_of_results)
    return round(predict * 100, 2)


def create_and_write_big_tree(data):
    """
    :param data: our data examples
    :return: the big tree of all the data .
    """
    global yes_table_count, no_table_count
    yes_table_count = len([data for data in data if data[-1] == "yes"])
    no_table_count = len([data for data in data if data[-1] == "no"])
    if yes_table_count > no_table_count:
        default = Node()
        default.class_type = "yes"
    else:
        default = Node()
        default.class_type = "no"
    attributes = ATTRIBUTES
    setMapAttributes(data)
    big_tree = dtl(data, attributes, default)
    return big_tree
