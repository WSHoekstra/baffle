# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


class BinaryClassificationReport():
    '''
    Generates a binary classification report.
    Takes 2 dataframes or np arrays, one containing the ground truth labels
    as binary labels by observation and class,
    and the other containing predictions
    as predicted probability by observation and class.
    '''
    def generate_report(self, labels, predictions):
        if isinstance(labels, pd.DataFrame):
            labels = labels['label'].values
        labels = labels.round().astype(int)
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions['probability'].values
        predictions = predictions.round().astype(int)

        if ( len(predictions) != len(labels) ):
            raise ValueError('labels and predictions need to be of equal length')

        n = len(predictions)
        prevalence_n = len([x for x in labels if x == 1])
        prevalence = round(len([x for x in labels if x == 1]) / len(labels),2)


        detection_n = len([x for x in predictions if x == 1])
        detection_rate = round(detection_n / len(predictions),2)

        conf_matrix = confusion_matrix(labels, predictions)
        n_true_positives = conf_matrix[1,1]
        n_true_negatives = conf_matrix[0,0]
        n_false_positives = conf_matrix[0,1]
        n_false_negatives = conf_matrix[1,0]

        accuracy = (n_true_positives + n_true_negatives) / n

        baseline_accuracy = max([prevalence, 1 - prevalence])

        positive_predictive_value = (n_true_positives) / (n_true_positives + n_false_positives)
        # specificity = positive_predictive_value
        negative_predictive_value = (n_true_negatives) / (n_true_negatives + n_false_negatives)

        sensitivity = (n_true_positives / (prevalence_n) )

        balanced_accuracy =  round(((sensitivity + positive_predictive_value) / 2),2)

        return {'n' : n,
                'prevalence_n' : prevalence_n,
                'prevalence' : prevalence,
                'detection_n' : detection_n,
                'detection_rate' : detection_rate,
                'true_positives' : n_true_positives,
                'true_negatives' : n_true_negatives,
                'false_positives' : n_false_positives,
                'false_negatives' : n_false_negatives,
                'baseline_accuracy' : baseline_accuracy,
                'accuracy' : accuracy,
                'positive_predictive_value_aka_precision' : positive_predictive_value,
                'negative_predictive_value' : negative_predictive_value,
                'sensitivity_aka_recall' : sensitivity,
                'balanced_accuracy' : balanced_accuracy}

    def __generate_mock_labels(self):
        return pd.DataFrame(data={
            'label': [0, 1, 1, 1, 1, 1]})

    def __generate_mock_predictions(self):
        return pd.DataFrame(data={
            'probability': [0.34, 0.7, 0.4, 0.8, 0.1, 0.0]})
            # 2 x TP,
            # 1 x TN,
            # 3 FN,
            # 0 FP


    def __init__(self, predictions, labels, testmode=False):
        self.predictions = predictions
        self.labels = labels
        self.__testmode = testmode

        # override inputs in test mode
        if self.__testmode:
            self.predictions = self.__generate_mock_predictions()
            self.labels = self.__generate_mock_labels()

        # type checking
        if not isinstance(self.predictions, pd.DataFrame) and not isinstance(self.predictions, np.ndarray):
            raise TypeError('predictions must be a pandas dataframe or a numpy array')
        if not isinstance(self.labels, pd.DataFrame):
            raise TypeError('labels must be a pandas dataframe or a numpy array')

        self.confusion_statistics = self.generate_report(self.labels, self.predictions)



# test = BinaryClassificationReport(None, None, testmode=True)
# test.confusion_statistics