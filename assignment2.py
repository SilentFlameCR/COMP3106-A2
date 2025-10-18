# Implementation: read CSV [class,length,weight,speed], learn Gaussian params per class
# and priors from the dataset, then classify a new snake via Naïve Bayes with
# independent Gaussian likelihoods.
# returns (most_likely_class, [P(anaconda), P(cobra), P(python)]) for a given [L, W, S].

from math import log, sqrt, pi, exp
import csv

def mean(values):
    return sum(values) / len(values) if values else 0.0

def std(values, mu):
    # population std (ddof = 1)
    # using epsilon floor to avoid zero variance issues
    n = len(values)
    if not values:
        return 1.0
    var = sum((x - mu) * (x - mu) for x in values) / (n - 1)
    return sqrt(var) if var > 0.0 else 1e-6

def log_gaussian_pdf(x, mu, sigma):
    sigma = max(sigma, 1e-6)
    return -0.5 * log(2.0 * pi) - log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

def parse_csv(dataset_filepath, classes=None):
    """
    Reads CSV with rows: label,length,weight,speed
    Returns:
      per_class object with key snake type and inside length, weight and speed
        formatted like: {class: {"length": [...], "weight": [...], "speed": [...]}}
      total_count: int
    """
    if classes is None:
        classes = ["anaconda", "cobra", "python"]

    per_class = {c: {"length": [], "weight": [], "speed": []} for c in classes}
    total_count = 0

    with open(dataset_filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue
            label = row[0].strip()
            if label not in per_class:
                # ignore unknowns
                continue
            try:
                length = float(row[1])
                weight = float(row[2])
                speed  = float(row[3])
            except ValueError:
                # skip error line
                continue
            per_class[label]["length"].append(length)
            per_class[label]["weight"].append(weight)
            per_class[label]["speed"].append(speed)
            total_count += 1

    return per_class, total_count

def naive_bayes_classifier(dataset_filepath, snake_measurements):
    # dataset_filepath is the full file path to a CSV file containing the dataset
    # snake_measurements is a list of [length, weight, speed] measurements for a snake

    # most_likely_class is a string indicating the most likely class, either "anaconda", "cobra", or "python"
    # class_probabilities is a three element list indicating the probability of each class in the order [anaconda probability, cobra probability, python probability]
    ORDER = ["anaconda", "cobra", "python"]

    per_class, total_count = parse_csv(dataset_filepath, classes=ORDER)
    # print(per_class)
    # print('-------')
    # print(total_count)
    if total_count == 0:
        raise ValueError("Dataset appears is empty.")

    # learn priors & Gaussian params
    priors = {}
    params = {}
    for c in ORDER:
        n_c = len(per_class[c]["length"])
        if n_c == 0:
            priors[c] = 1e-12
            params[c] = {
                "mu":    {"length": 0.0, "weight": 0.0, "speed": 0.0},
                "sigma": {"length": 1e-6, "weight": 1e-6, "speed": 1e-6},
            }
            continue

        priors[c] = n_c / total_count
        mu_len = mean(per_class[c]["length"])
        mu_wt  = mean(per_class[c]["weight"])
        mu_spd = mean(per_class[c]["speed"])

        sig_len = std(per_class[c]["length"], mu_len)
        sig_wt  = std(per_class[c]["weight"], mu_wt)
        sig_spd = std(per_class[c]["speed"], mu_spd)

        params[c] = {
            "mu":    {"length": mu_len, "weight": mu_wt, "speed": mu_spd},
            "sigma": {"length": sig_len, "weight": sig_wt, "speed": sig_spd},
        }

    try:
        x_len, x_wt, x_spd = [float(v) for v in snake_measurements]
    except Exception as e:
        raise ValueError("snake_measurements must be [length, weight, speed].") from e

    # log-posteriors
    log_posts = []
    for c in ORDER:
        mu = params[c]["mu"]
        sg = params[c]["sigma"]
        lp = log(max(priors[c], 1e-18))
        lp += log_gaussian_pdf(x_len, mu["length"], sg["length"])
        lp += log_gaussian_pdf(x_wt,  mu["weight"], sg["weight"])
        lp += log_gaussian_pdf(x_spd, mu["speed"],  sg["speed"])
        log_posts.append(lp)

    # normalize (log-sum-exp)
    m = max(log_posts)
    exps = [exp(lp - m) for lp in log_posts]
    s = sum(exps)
    class_probabilities = [e / s for e in exps]

    most_idx = max(range(len(ORDER)), key=lambda i: class_probabilities[i])
    most_likely_class = ORDER[most_idx]

    # print(f"The class probabilities are: {class_probabilities} and the most likely class is {most_likely_class}")
    
    return most_likely_class, class_probabilities

# naive_bayes_classifier('./Examples/Example0/dataset.csv', [390, 28, 13])