import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def wrapper(x_generator, sample_size, p1=0, p2=0):
    try:
        return x_generator(p1, p2, size=sample_size)
    except:
        try:
            return x_generator(p1, size=sample_size)
        except:
            x_generator(size=sample_size)


def compute_bias_variance(regressor, dependence_fun,
                          x_generator=np.random.uniform,
                          noise_generator=np.random.uniform,
                          sample_size=300, samples_num=300,
                          objects_num=200, seed=1234):

    samples = np.empty((samples_num, sample_size))
    noise = np.empty((samples_num, sample_size))

    for i in range(samples_num):
        samples[i] = x_generator(size=sample_size)
        noise[i] = noise_generator(size=sample_size)

    objects = x_generator(size=objects_num)
    mean_noise = noise_generator(size=objects_num).mean()

    bias, variance = compute_bias_variance_fixed_samples(regressor,
                                                         dependence_fun,
                                                         samples,
                                                         objects, noise,
                                                         mean_noise)

    return bias, variance


def compute_bias_variance_fixed_samples(regressor, dependence_fun,
                                        samples, objects, noise, mean_noise):

    E_y_x = dependence_fun(objects) + mean_noise

    y_predict = np.zeros((samples.shape[0], objects.shape[0]))

    for i in range(samples.shape[0]):
        regressor.fit(samples[i][:, np.newaxis],
                      dependence_fun(samples[i]) + noise[i])
        y_predict[i] = regressor.predict(objects[:, np.newaxis])

    E_X_mu_X = y_predict.mean(axis=0)
    bias = ((E_X_mu_X - E_y_x) ** 2).mean()

    variance = ((y_predict - E_X_mu_X) ** 2).mean(axis=0).mean()
    return bias, variance


def H(R, target_vector):
    s_r = np.sum(R, axis=1)
    p1 = np.sum(np.multiply(target_vector, R), axis=1) / s_r
    p0 = 1 - p1
    return (1 - p0 ** 2 - p1 ** 2) * s_r


def find_best_split(feature_vector, target_vector):
    if np.all(feature_vector == feature_vector[0]):
        return -np.inf, -np.inf, -np.inf, -np.inf
    sorted_feat = np.unique(feature_vector)
    thresholds = (sorted_feat[1:] + sorted_feat[:-1]) / 2.0
    thr_vector = thresholds[:, np.newaxis]
    R_l = np.less(feature_vector, thr_vector)
    R_r = np.greater(feature_vector, thr_vector)
    ginis = - (H(R_l, target_vector) + H(R_r, target_vector))
    ginis = ginis / len(feature_vector)
    gini_best_ix = np.argmax(ginis)
    gini_best = ginis[gini_best_ix]
    threshold_best = thresholds[gini_best_ix]
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None,
                 min_samples_split=None, min_samples_leaf=None):
        if any(list(map(lambda x: x != "real" and x != "categorical",
                        feature_types))):
            raise ValueError("There is an unknown feature type")
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        if min_samples_leaf is None:
            self._min_samples_leaf = 1
        else:
            self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if len(sub_y) == 0:
            node["type"] = "terminal"
            node["class"] = 0
            print("0")
            return
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        if self._max_depth is not None and depth > self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        if self._min_samples_split is not None:
            if sub_X.shape[0] < self._min_samples_split:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["depth"] = depth

        feature_best, threshold_best, gini_best, split = None, None, None, None
        split_best = None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0],
                                             sorted(ratio.items(),
                                             key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories,
                                          list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x],
                                                   sub_X[:, feature])))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            split = feature_vector < threshold

            if self._min_samples_leaf is not None:
                if np.sum(split) < self._min_samples_leaf or \
                   np.sum(1 - split) < self._min_samples_leaf:
                    continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split_best = split

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] <
                                                     threshold,
                                              categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split_best], sub_y[split_best],
                       node["left_child"], node["depth"] + 1)
        self._fit_node(sub_X[np.logical_not(split_best)],
                       sub_y[np.logical_not(split_best)],
                       node["right_child"], node["depth"] + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_best = node["feature_split"]
        if self._feature_types[feature_best] == "real":
            threshold_best = node["threshold"]
            if x[feature_best] < threshold_best:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_best] == "categorical":
            threshold_best = node["categories_split"]
            if x[feature_best] in threshold_best:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
