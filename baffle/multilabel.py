# -*- coding: utf-8 -*-

import pandas as pd


class MultiLabelClassificationReport():
    '''
    Generates a multi label classification report.
    Takes 2 dataframes, one containing the ground truth labels  
    as binary labels by observation and class,
    and the other containing predictions 
    as predicted probability by observation and class.    
    '''
    
    def determine_classification_outcome_type(self, cols):
        '''
        Compares a prediction column to a label column.
    
        Args
            cols: a dataframe with 2 columns (predictions, labels)
    
        Returns the classification result:
            'TP' for True Positive 
            'FP' for false positive 
            'TN' for true negative 
            'FN' for false negative
            '''
        prediction = cols[0]
        label = cols[1]
        if prediction == 1 and label == 1:
            return 'TP'
        elif prediction == 1 and label == 0:
            return 'FP'
        elif prediction == 0 and label == 0:
            return 'TN'
        elif prediction == 0 and label == 1:
            return 'FN'
        else:
            ValueError('Expecting int 0 or 1 +\
                       for both prediction and label, +\
                       got prediction = {}, label = {} '.format(
                       prediction, 
                       label)
                       )
    
    
    def generate_classification_metrics(self):  
        '''
        Generates some row level classification metrics to summarize later.
        Assumes to be passed a dataframe of predictions, and a dataframe of labels.
        Will combine and compare them and calculate error rates, 
        classification correctness (assuming probability cutoff of 0.5) and
        classification types ('TP', 'TN', 'FP', 'FN').
    
        '''
        labels_vs_predictions = pd.merge(self.predictions, self.labels, how='left', on=['observation', 'class'])
        labels_vs_predictions['error'] =  labels_vs_predictions['label'] - labels_vs_predictions['probability']
        labels_vs_predictions['absolute_error'] =  abs(labels_vs_predictions['label'] - labels_vs_predictions['probability'])
        labels_vs_predictions['label_pred'] = labels_vs_predictions['probability'].round().astype(int)
        labels_vs_predictions['classification_outcome_type'] = labels_vs_predictions[['label_pred', 'label']].apply(
                    self.determine_classification_outcome_type, 
                    axis=1)       
        labels_vs_predictions['classification_correct'] = (labels_vs_predictions['label'] == labels_vs_predictions['label_pred']).astype(int)
        return labels_vs_predictions
        
    
    def generate_class_confusion_statistics(self, decimals=3):
        '''
        Generates confusion statistics per class.
        '''
        labels_vs_predictions = self.rowlevel_confusion_statistics
        class_confusion_statistics_list = []        
        for outcome in labels_vs_predictions['class'].unique():
            classdf = labels_vs_predictions[ labels_vs_predictions['class'] == outcome]
            class_confusion_statistics_list.append(self.generate_confusion_statistics(labels_vs_predictions = classdf, 
                                                                                      class_name = outcome))            
        class_confusion_statistics = pd.concat(class_confusion_statistics_list)        
        return class_confusion_statistics
    

    def generate_confusion_statistics(self, labels_vs_predictions = None, decimals=3, class_name = None):      
        
        if not class_name:
            class_name = 'overall'
            
        if not isinstance(labels_vs_predictions, pd.DataFrame):
            if not labels_vs_predictions:
                self.generate_classification_metrics()
            labels_vs_predictions = self.rowlevel_confusion_statistics
        sensitivity_numerator = labels_vs_predictions[labels_vs_predictions.classification_outcome_type.isin(['TP'])].shape[0]
        sensitivity_denominator = labels_vs_predictions[labels_vs_predictions.classification_outcome_type.isin(['TP', 'FN'])].shape[0]
        sensitivity = 0.0 if not sensitivity_denominator else (sensitivity_numerator / sensitivity_denominator)
            
        specificity_numerator = labels_vs_predictions[labels_vs_predictions.classification_outcome_type.isin(['TN'])].shape[0]
        specificity_denominator = labels_vs_predictions[labels_vs_predictions.classification_outcome_type.isin(['TN', 'FP'])].shape[0]
        specificity = 0.0 if not specificity_denominator else (specificity_numerator / specificity_denominator)
           
        positive_predictive_value_numerator = labels_vs_predictions[ (labels_vs_predictions['label_pred'] == 1) & (labels_vs_predictions['label'] == 1) ].shape[0]
        positive_predictive_value_denominator = labels_vs_predictions[labels_vs_predictions['label_pred'] == 1].shape[0]                                    
        positive_predictive_value =  0.0 if not positive_predictive_value_denominator else (positive_predictive_value_numerator / positive_predictive_value_denominator)
            
        negative_predictive_value_numerator = labels_vs_predictions[ (labels_vs_predictions['label_pred'] == 0) & (labels_vs_predictions['label'] == 0) ].shape[0]
        negative_predictive_value_denominator = labels_vs_predictions[labels_vs_predictions['label_pred'] == 0].shape[0]
        negative_predictive_value =  0.0 if not negative_predictive_value_denominator else (negative_predictive_value_numerator / negative_predictive_value_denominator)
        
        
        # generate overall confusion statistics
        confusion_statistics = {'class' : class_name,
                'n': labels_vs_predictions.shape[0],
                'prevalence' : labels_vs_predictions.label.mean(),
                'prevalence_n' : sum(labels_vs_predictions['label'] == 1),
                'detection_prevalence' : round(labels_vs_predictions.label_pred.mean(), decimals),
                'detection_prevalence_n' : sum(labels_vs_predictions.label_pred),
                'true_positive_n' : sum(labels_vs_predictions['classification_outcome_type'] == 'TP'),
                'false_positive_n' : sum(labels_vs_predictions['classification_outcome_type'] == 'FP'),
                'true_negative_n' : sum(labels_vs_predictions['classification_outcome_type'] == 'TN'),
                'false_negative_n' : sum(labels_vs_predictions['classification_outcome_type'] == 'FN'),
                'accuracy' : round(labels_vs_predictions.classification_correct.mean(), decimals),
                'sensitivity' : round(sensitivity, decimals),
                'specificity' : round(specificity, decimals),                                          
                'positive_predictive_value' : round(positive_predictive_value, decimals),
                'negative_predictive_value' : round(negative_predictive_value, decimals)
                }  
        
        
        confusion_statistics = pd.DataFrame.from_records(confusion_statistics, index=[0])
        confusion_statistics = confusion_statistics[['class', 
                                                                         'n',
                                                                         'prevalence',
                                                                         'prevalence_n', 
                                                                         'detection_prevalence', 
                                                                         'detection_prevalence_n',
                                                                         'true_positive_n',
                                                                         'false_positive_n',
                                                                         'true_negative_n',
                                                                         'false_negative_n',
                                                                         'accuracy', 
                                                                         'sensitivity',
                                                                         'specificity',
                                                                         'positive_predictive_value',
                                                                         'negative_predictive_value'
                                                                         ]]
        
        return confusion_statistics

    def __generate_mock_predictions(self):
        predictions = pd.DataFrame(data={
            'observation': [1, 1, 2, 2], 
            'class': [1, 2, 1, 2], 
            'probability': [0.34, 0.7, 0.4, 0.8]})
        predictions = predictions[['observation', 'class', 'probability']]
        return predictions

    def __generate_mock_labels(self):
        labels = pd.DataFrame(data={
                'observation': [1, 1, 2, 2], 
                'class': [1, 2, 1, 2], 
                'label': [0, 1, 1, 1]})
        labels = labels[['observation', 'class', 'label']]
        return labels
        
    def __init__(self, predictions, labels, testmode=False):        
        self.predictions = predictions
        self.labels = labels        
        self.__testmode = testmode
        
        # override inputs in test mode
        if self.__testmode:
            self.predictions = self.__generate_mock_predictions()
            self.labels = self.__generate_mock_labels()
           
                
        # type checking
        if not isinstance(self.predictions, pd.DataFrame):
            raise TypeError('predictions must be a pandas dataframe')        
        if not isinstance(self.labels, pd.DataFrame):
            raise TypeError('labels must be a pandas dataframe')
        
        self.rowlevel_confusion_statistics = self.generate_classification_metrics()
        self.overall_confusion_statistics = self.generate_confusion_statistics()
        self.class_confusion_statistics = self.generate_class_confusion_statistics()
           
# test = MultiLabelClassificationReport(None, None, testmode=True)