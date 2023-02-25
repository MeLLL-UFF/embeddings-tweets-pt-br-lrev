from nlp_ptbr.base import *
from nlp_ptbr.data import SentimentDatasets
from nlp_ptbr.experiment import MaskedLMFineTuningExperiment
from nlp_ptbr.preprocessing import NORMALIZE_BERTWEET_STRIP_SPACES

import numpy as np
import pandas as pd
import argparse, os, gc, torch
from tabulate import tabulate
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=UndefinedMetricWarning)
simplefilter("ignore", category=UserWarning)
simplefilter("ignore", category=FutureWarning)

import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=str, dest='classes', default='mix', help='Binary or multiclass.', choices=['multiclass', 'binary', 'mix'])
    parser.add_argument('--selected_transformers', dest='selected_transformers', type=str, nargs="+", default=['xlmt-base'])
    parser.add_argument('--experiment_type', type=str, dest='experiment_type', default='feature extraction', help='Experiment type.', choices=['feature extraction', 'fine-tuning downstream'])
    parser.add_argument('--experiment_name', type=str, dest='experiment_name', default='contextual', help='Experiment name.')
    parser.add_argument('--results_from', type=str, dest='results_from', default='fine-tuning mlm', help='Directory where results file has been saved.')
    parser.add_argument('--results_dir', type=str, dest='results_dir', default='embeddings-ft', help='Directory where results must be saved.')
    parser.add_argument('--log_dir', type=str, dest='log_dir', default='embeddings-ft', help='Directory where logs must be saved.')
    parser.add_argument('--log_level', type=str, dest='log_level', default='WARNING', help='Log level.')
    parser.add_argument('--seed', type=int, dest='seed', default=2017, help='Seed.')
    parser.add_argument('--folds', type=int, dest='folds', default=10, help='Number of Kfolds to evaluate.')
    parser.add_argument('--nn_models', dest='nn_models', action='store_true', help='FLAG: Use NN models as classifiers.')
    parser.add_argument('--save_classifiers', dest='save_classifiers', action='store_true', help='FLAG: Save trained classifiers.')
    parser.add_argument('--disable_tqdm', dest='disable_tqdm', action='store_true', help='FLAG: Disable progress bar.')
    parser.add_argument('--only_these_folds', type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=256, help='Batch size.')
    parser.add_argument('--strategy', type=str, dest='strategy', default='loo', help='Strategy on how to continue mlm training.', choices=['loo', 'alldata', 'indata'])
    parser.add_argument('--eval_type', type=str, dest='eval_type', default='each', help='Run on each datasets, on all or both.', choices=['each', 'all', 'both'])
    parser.add_argument('--train_epochs', type=int, dest='train_epochs', default=3, help='Number of train epochs.')
    parser.add_argument('--local_files_only', dest='local_files_only', action='store_true', help='FLAG: Use only local files Hugging Face.')
    
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    #field = args.field
    #datasets_path = args.datasets_path
    classes = args.classes
    #datasets_index = args.datasets_index
    selected_transformers = args.selected_transformers
    #model_name = args.model_name
    experiment_type = args.experiment_type
    experiment_name = args.experiment_name
    results_from = args.results_from
    results_dir = args.results_dir
    log_dir = args.log_dir
    log_level = args.log_level
    seed = args.seed
    folds = args.folds
    nn_models = args.nn_models
    save_classifiers = args.save_classifiers
    disable_tqdm = args.disable_tqdm
    only_these_folds = args.only_these_folds
    batch_size = args.batch_size
    strategy = args.strategy
    eval_type = args.eval_type
    train_epochs = args.train_epochs
    local_files_only = args.local_files_only
    
    wandb.init(
        entity="bertweetbr",
        notes=f'Contextual word-embeddings experiment for short text classification from fine-tuned models strategy {strategy}: {classes}',
        # Set the project where this run will be logged
        project=f"{experiment_type}-from-finetuned-mlm",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        #name=f'{experiment_name}-{wandb.util.generate_id()}',
        name=f'{experiment_name}-{classes}-from-{strategy}',
        # Track hyperparameters and run metadata
        config={
            'experiment_type': f"{experiment_type}-from-finetuned-mlm-{strategy}",
            'results_from': args.results_from,
            'classes': args.classes,
            'strategy': args.strategy,
            'folds': args.folds,
            'seed': args.seed},
        job_type='contextual-embeddings-evaluation',
        tags=['contextual', 'transformers', args.classes, args.experiment_type, f"{experiment_type}-from-finetuned-mlm-{strategy}"])
    
    print(f'\nINICIO AVALIAÇÃO DE MODELOS A PARTIR DE FINE-TUNING MLM.')
    
    if experiment_type == 'feature extraction':
        results = MaskedLMFineTuningExperiment.run_contextual_on_finetuned_models(
            results_from=results_from, results_dir=results_dir, strategy = strategy, classes = classes, disable_tqdm=disable_tqdm, batch_size=batch_size, save=False,
            selected_transformers=selected_transformers, seed=seed, folds=folds, experiment_type=experiment_type, local_files_only=local_files_only)
    elif experiment_type == 'fine-tuning downstream':
        results, results_epochs = MaskedLMFineTuningExperiment.run_contextual_on_finetuned_models(
            results_from=results_from, results_dir=results_dir, strategy = strategy, classes = classes, disable_tqdm=disable_tqdm, batch_size=batch_size, save=False,
            selected_transformers=selected_transformers, seed=seed, folds=folds, train_epochs=train_epochs, eval_type=eval_type, experiment_type=experiment_type, local_files_only=local_files_only)
    
    print(f'\nFIM AVALIAÇÃO DE MODELOS A PARTIR DE FINE-TUNING MLM.')
    
    group_by=['model', 'field', 'dataset', 'split', 'experiment', 'origin', 'classifier']
    
    artifact = wandb.Artifact(name=f'{experiment_name}-{classes}-{strategy}-results', type='results')
    if experiment_type == 'fine-tuning downstream':
        artifact.add(wandb.Table(dataframe=results_epochs), "epochs")
    artifact.add(wandb.Table(dataframe=results), "folds")
    artifact.add(wandb.Table(dataframe=results.drop(labels=['fold'], axis=1).groupby(group_by).agg('mean').reset_index()), "consolidated")
    wandb.log_artifact(artifact)
    
    wandb.finish()
    
if __name__ == '__main__':
    main()