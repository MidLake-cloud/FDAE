import numpy as np


def ovo_voting(decision_ovo, n_classes):
    predictions = np.zeros(len(decision_ovo))
    class_pos, class_neg = ovo_class_combinations(n_classes)

    counter = np.zeros([len(decision_ovo), n_classes])

    for p in range(len(decision_ovo)):
        for i in range(len(decision_ovo[p])):
            if decision_ovo[p, i] > 0:
                counter[p, class_pos[i]] += 1
            else:
                counter[p, class_neg[i]] += 1

        predictions[p] = np.argmax(counter[p])

    return predictions, counter


def ovo_class_combinations(n_classes):
    class_pos = []
    class_neg = []
    for c1 in range(n_classes - 1):
        for c2 in range(c1 + 1, n_classes):
            class_pos.append(c1)
            class_neg.append(c2)

    return class_pos, class_neg


# Each classifier adds it value for both classes (+/-)
# Then the class with largest number of votes is the prediction
def ovo_voting_both(decision_ovo, n_classes):
    predictions = np.zeros(len(decision_ovo))
    class_pos, class_neg = ovo_class_combinations(n_classes)

    counter = np.zeros([len(decision_ovo), n_classes])

    for p in range(len(decision_ovo)):
        for i in range(len(decision_ovo[p])):
            counter[p, class_pos[i]] += decision_ovo[p, i]
            counter[p, class_neg[i]] -= decision_ovo[p, i]

        predictions[p] = np.argmax(counter[p])

    return predictions, counter


def ovo_voting_exp(decision_ovo, n_classes):
    predictions = np.zeros(len(decision_ovo))
    class_pos, class_neg = ovo_class_combinations(n_classes)

    counter = np.zeros([len(decision_ovo), n_classes])

    for p in range(len(decision_ovo)):
        for i in range(len(decision_ovo[p])):
            counter[p, class_pos[i]] += 1 / (1 + np.exp(-decision_ovo[p, i]))
            counter[p, class_neg[i]] += 1 / (1 + np.exp(decision_ovo[p, i]))

        predictions[p] = np.argmax(counter[p])

    return predictions, counter