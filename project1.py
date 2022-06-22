from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def hinge_loss_single(feature_vector, label, theta, theta_0):
    z = label * (np.dot(feature_vector, theta) + theta_0)
    if z > 1:
        return 0
    else:
        return 1 - z


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    hloss = np.array([])
    for i in range(len(feature_matrix)):
        il = hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
        hloss = np.append(hloss, il)
    return np.mean(hloss)


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    z = label * (np.dot(feature_vector, current_theta) + current_theta_0)
    if z > 0:
        return (current_theta, current_theta_0)
    else:
        theta = current_theta + (label * feature_vector)
        theta_0 = current_theta_0 + label
    return (theta, theta_0)



def perceptron(feature_matrix, labels, T):
    n = len(feature_matrix[0])
    theta = np.zeros(n)
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
    return (theta, theta_0)


def average_perceptron(feature_matrix, labels, T):
    n = len(feature_matrix[0])
    n2 = len(feature_matrix)
    theta, theta_s = np.zeros(n), np.zeros(n)
    theta_0, theta_0_s = 0, 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            pt = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta, theta_0 = pt
            theta_s += theta
            theta_0_s += theta_0
    theta_s = theta_s / (n2*T)
    theta_0_s = theta_0_s / (n2*T)
    return (theta_s, theta_0_s)


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    z = label * (np.dot(feature_vector, current_theta) + current_theta_0)
    if z > 1:
        theta = (1-eta*L)*current_theta
        return (theta, current_theta_0)
    else:
        theta = (1-eta*L)*current_theta + eta*(label * feature_vector)
        theta_0 = current_theta_0 + eta*label
    return (theta, theta_0)


def pegasos(feature_matrix, labels, T, L):
    n = len(feature_matrix[0])
    n2 = len(feature_matrix)
    a = 0
    theta = np.zeros(n)
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            a += 1
            eta = 1/np.sqrt(a)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta_0)
    return (theta, theta_0)

# Part II


def classify(feature_matrix, theta, theta_0):
    res = np.array([])
    for i in feature_matrix:
        if np.dot(i, theta) + theta_0 > 0:
            res = np.append(res, 1)
        else:
            res = np.append(res, -1)
    return res


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    val_preds = classify(val_feature_matrix, theta, theta_0)
    train_preds = classify(train_feature_matrix, theta, theta_0)
    val_acc = accuracy(val_preds, val_labels)
    train_acc = accuracy(train_preds, train_labels)
    return (train_acc, val_acc)



def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    f = open("stopwords.txt", "r")
    f = open("stopwords.txt", "r")
    data = f.read()
    to_avoid = data.replace('\n', ' ').split(" ")
    f.close()
    dictionary = {} # maps word to unique index
    for text in texts:
            word_list = extract_words(text)
            for word in word_list:
                if word not in dictionary and word not in to_avoid:
                    dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
