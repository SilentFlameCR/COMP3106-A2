
def naive_bayes_classifier(dataset_filepath, snake_measurements):
  # dataset_filepath is the full file path to a CSV file containing the dataset
  # snake_measurements is a list of [length, weight, speed] measurements for a snake

  # most_likely_class is a string indicating the most likely class, either "anaconda", "cobra", or "python"
  # class_probabilities is a three element list indicating the probability of each class in the order [anaconda probability, cobra probability, python probability]
  return most_likely_class, class_probabilities