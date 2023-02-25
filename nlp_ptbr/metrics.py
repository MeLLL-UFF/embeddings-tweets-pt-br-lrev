# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

import datasets
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, log_loss


# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {Text Classification Metrics for sentiment analysis of tweets in portuguese},
authors={Fernando Pereira Carneiro},
year={2021}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
The set of metrics computed in the scope of the experiments conducted by 
Fernando Carneiro as part of its masters degree at Universidade Federal Fluminense.
For this study it is computed the metrics accuracy, f1, precision, recall as well as the confusion matrix
for a number of nlp tasks in Portuguese.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    "acc": Accuracy
    "f1": F1 score
    "precision": Precision
    "recall": Recall
    "tn": True Negative Count
    "fp": False Positive Count
    "fn": False Negative Count
    "tp": True Positive Count
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"
mc_metrics = [(precision_score, 'precision'), (recall_score, 'recall'), (f1_score, 'f1')]

def get_multiclass_metrics(metric_values, metric_name, labels=None):
    if labels is None:
        return({f'{metric_name}_{str(i)}': metric_values[i] for i in range(metric_values.shape[0])})
    else:
        return({f'{metric_name}_{labels[i]}': metric_values[i] for i in range(metric_values.shape[0])})

def counts_from_confusion(confusion, labels=None):
    """
    Obtain TP, FN FP, and TN for each class in the confusion matrix
    """

    counts_list = dict()

    # Iterate through classes and store the counts
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]

        fn_mask = np.zeros(confusion.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = int(np.sum(np.multiply(confusion, fn_mask)))

        fp_mask = np.zeros(confusion.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = int(np.sum(np.multiply(confusion, fp_mask)))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = int(np.sum(np.multiply(confusion, tn_mask)))
        
        if labels is None:
            counts_list[i] = ({f'tp_{i}': tp, f'fn_{i}': fn, f'fp_{i}': fp, f'tn_{i}': tn})
        else:
            counts_list[labels[i]] = ({f'tp_{labels[i]}': tp, f'fn_{labels[i]}': fn, f'fp_{labels[i]}': fp, f'tn_{labels[i]}': tn})

    return counts_list

def get_multiclass_metrics(metric_values, metric_name, labels=None):
    if labels is None:
        return({f'{metric_name}_{str(i)}': metric_values[i] for i in range(metric_values.shape[0])})
    else:
        return({f'{metric_name}_{labels[i]}': metric_values[i] for i in range(metric_values.shape[0])})

def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


def acc_and_others(predictions, references, average='weighted', normalize=True, sample_weight=None, labels=None, pos_label=1, predictions_proba=None, compute_loss=True, labels_names=None):
    counts = {f'references_{c}':0 for c in labels}
    counts.update({f'predictions_{c}':0 for c in labels})
    
    unique_references, counts_references = np.unique(references, return_counts=True)
    references_counts = dict(zip([f'references_{c}' for c in unique_references], counts_references))
    
    unique_predictions, counts_predictions = np.unique(predictions, return_counts=True)
    predictions_counts = dict(zip([f'predictions_{c}' for c in unique_predictions], counts_predictions))
    
    counts.update(references_counts)
    counts.update(predictions_counts)
    
    acc = float(accuracy_score(y_true=references, y_pred=predictions, normalize=normalize, sample_weight=sample_weight))
    f1 = f1_score(y_true=references, y_pred=predictions, average=average, labels=labels, pos_label=pos_label, sample_weight=sample_weight)
    precision = precision_score(y_true=references, y_pred=predictions, average=average, labels=labels, pos_label=pos_label, sample_weight=sample_weight)
    recall = recall_score(y_true=references, y_pred=predictions, average=average, labels=labels, pos_label=pos_label, sample_weight=sample_weight)
    #auc = roc_auc_score(y_true=references, y_score=predictions, sample_weight=sample_weight, labels=labels, average=average)
    
    if len(labels) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true=references, y_pred=predictions, normalize=None, sample_weight=sample_weight, labels=labels).ravel()
        scores = {
                "acc": acc,
                "f1": float(f1) if f1.size == 1 else f1,
                "precision": float(precision) if precision.size == 1 else precision,
                "recall": float(recall) if recall.size == 1 else recall,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                #"auc": float(auc) if auc.size == 1 else auc,
        }
    else:
        cm = confusion_matrix(y_true=references, y_pred=predictions, normalize=None, sample_weight=sample_weight, labels=labels)
        scores = {
                "acc": acc,
                "f1": float(f1) if f1.size == 1 else f1,
                "precision": float(precision) if precision.size == 1 else precision,
                "recall": float(recall) if recall.size == 1 else recall,
        }
        
        cm_counts = counts_from_confusion(cm)
        for k, v in cm_counts.items():
            scores.update(v)
        
        multiclass_metrics = {
            name: get_multiclass_metrics(
                func(y_true=references, y_pred=predictions, average=None, sample_weight=sample_weight, labels=labels, pos_label=1), name, labels=labels) for func, name in mc_metrics}
        
        for k, v in multiclass_metrics.items():
            scores.update(v)
            
    if compute_loss:
        if predictions_proba is None:
            #print('')
            #print(f'Labels: {labels}')
            #print(f'True: {references}')
            #print(f'Predicted: {predictions}')
            #scores['loss'] = float(log_loss(y_true=references, y_pred=predictions, labels=labels))
            scores['loss'] = None
        else:
            scores['loss'] = float(log_loss(y_true=references, y_pred=predictions_proba))
    
    scores.update(counts)
    
    return scores

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class UFFMetric(datasets.Metric):
    """TODO: Short description of my metric."""
    
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                'predictions': datasets.Value('int64'),
                'references': datasets.Value('int64'),
            }),
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=['https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score',
                            'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html',
                            'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score',
                            'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score',
                            'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix',
                            'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss'
                           ],
            format='numpy'
        )

    def _compute(self, predictions, references, average='weighted', normalize=True, sample_weight=None, labels=None, pos_label=1, predictions_proba=None, compute_loss=True, labels_names=None):
        return acc_and_others(predictions, references, average=average, normalize=normalize, sample_weight=sample_weight, predictions_proba=predictions_proba, labels=labels, compute_loss=compute_loss, labels_names=labels_names)
