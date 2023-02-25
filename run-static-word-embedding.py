from nlp_ptbr.base import *
from nlp_ptbr.data import SentimentDatasets
from nlp_ptbr.experiment import StaticEmbeddingsExperiment, ContextualEmbeddingsExperiment, Experiment, EmbeddingsExperiment
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
    parser.add_argument('--field', type=str, dest='field', default='tweet', help='What fields on which the experiments will run.', choices=['tweet', 'tweet_normalized', 'both'])
    parser.add_argument('--datasets_path', type=str, dest='datasets_path', default=None, help='Datasets path.')
    parser.add_argument('--classes', type=str, dest='classes', default='multiclass', help='Binary or multiclass.', choices=['multiclass', 'binary'])
    parser.add_argument('--datasets_index', type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument('--selected_word_embeddings', dest='selected_word_embeddings', type=str, nargs="+", default=['fasttext'])
    parser.add_argument('--model_name', type=str, dest='model_name', default='fasttext', help='Embedding model name.')
    parser.add_argument('--experiment_type', type=str, dest='experiment_type', default='feature extraction', help='Experiment type.', choices=['feature extraction', 'fine-tuning downstream'])
    parser.add_argument('--experiment_name', type=str, dest='experiment_name', default='static', help='Experiment name.')
    parser.add_argument('--results_dir', type=str, dest='results_dir', default='embeddings', help='Directory where results must be saved.')
    parser.add_argument('--log_dir', type=str, dest='log_dir', default='embeddings', help='Directory where logs must be saved.')
    parser.add_argument('--log_level', type=str, dest='log_level', default='WARNING', help='Log level.')
    parser.add_argument('--seed', type=int, dest='seed', default=2017, help='Seed.')
    parser.add_argument('--folds', type=int, dest='folds', default=10, help='Number of Kfolds to evaluate.')
    #parser.add_argument('--n_classes', type=int, dest='n_classes', default=2, help='Number of classes.')
    parser.add_argument('--nn_models', dest='nn_models', action='store_true', help='FLAG: Use NN models as classifiers.')
    parser.add_argument('--tokenizer_type', type=str, dest='tokenizer_type', default='hugging_face', help='Type of custom tokenizer to train.')
    parser.add_argument('--num_tokens', type=int, dest='num_tokens', default=30000, help='Number of tokens of the dictionary of tokenizer to train.')
    parser.add_argument('--save_classifiers', dest='save_classifiers', action='store_true', help='FLAG: Save trained classifiers.')
    parser.add_argument('--disable_tqdm', dest='disable_tqdm', action='store_true', help='FLAG: Disable progress bar.')
    parser.add_argument('--eval_type', type=str, dest='eval_type', default='each', help='Run on each datasets, on all or both.', choices=['each', 'all', 'both'])
    parser.add_argument('--only_these_folds', type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=32, help='Batch size.')
    parser.add_argument('--do', type=str, dest='do', default='evaluate', help='What to do: evaluate, train or both.', choices=['evaluate', 'train', 'both'])
    
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    field = args.field
    datasets_path = args.datasets_path
    classes = args.classes
    datasets_index = args.datasets_index
    selected_word_embeddings = args.selected_word_embeddings
    model_name = args.model_name
    experiment_type = args.experiment_type
    experiment_name = args.experiment_name
    results_dir = args.results_dir
    log_dir = args.log_dir
    log_level = args.log_level
    seed = args.seed
    folds = args.folds
    nn_models = args.nn_models
    tokenizer_type = args.tokenizer_type
    num_tokens = args.num_tokens
    save_classifiers = args.save_classifiers
    disable_tqdm = args.disable_tqdm
    eval_type = args.eval_type
    only_these_folds = args.only_these_folds
    batch_size = args.batch_size
    do = args.do
    
    if field == 'both':
        cols = [('tweet', True), ('tweet_normalized', False)]
    elif field == 'tweet':
        cols = [('tweet', True)]
    else:
        cols = [('tweet_normalized', False)]
    
    datasets = SentimentDatasets(datasets_path=datasets_path, normalize_funcs=NORMALIZE_BERTWEET_STRIP_SPACES, normalize = True, remove_duplicates=False, remove_long_sentences=False, classes=classes)
    
    if do == 'both' or do == 'evaluate':

        print(f'\nINICIO AVALIAÇÃO DE MODELOS.')

        wandb.init(
            entity="bertweetbr",
            notes=f'Static word-embeddings experiment for short text classification: {classes}',
            # Set the project where this run will be logged
            project=f"{experiment_type}",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #name=f'{experiment_name}-{wandb.util.generate_id()}',
            name=f'{experiment_name}-{classes}',
            # Track hyperparameters and run metadata
            config={
                'experiment_type': args.experiment_type,
                'datasets_path': args.datasets_path,
                'classes': args.classes,
                'datasets_index': args.datasets_index,
                'field': args.field,
                'folds': args.folds,
                'seed': args.seed},
            job_type='static-embeddings-evaluation',
            tags=['static', 'word-embedding', args.classes, args.experiment_type, args.field])
        
        all_results = []
        
        for col, _ in cols:
            for k, v in StaticEmbeddingsExperiment.word_embeddings.items():
                if k in selected_word_embeddings:
                    print(f'Text column: {col} - Checkpoint: {k}')
                    
                    experiment = StaticEmbeddingsExperiment(
                        model_name=v['embedding_name'],
                        experiment_type = experiment_type,
                        experiment_name=f"{experiment_name}-{v['embedding_name']}",
                        results_dir=results_dir,
                        log_dir=log_dir,
                        log_level=log_level,
                        seed = seed,
                        folds=folds,
                        num_classes=datasets.n_classes,
                        nn_models=nn_models,
                        **v,
                        tokenizer_type=tokenizer_type,
                        num_tokens=num_tokens,
                        sequence_length='MAX',
                        save_classifiers=save_classifiers
                    )
                    
                    print(f'\nINÍCIO: SENTIMENT ANALYSIS - {experiment.experiment_name}')
                    experiment.logger.info(f'\nINÍCIO: SENTIMENT ANALYSIS - {experiment.experiment_name}')

                    params = vars(args)
                    experiment.logger.debug('\n' + tabulate({"Parâmetro": list(params.keys()), "Valor": list(params.values())}, headers="keys", tablefmt="psql", floatfmt=".2f"))
                    
                    experiment.logger.debug('')
                    
                    results, elapsed_time, dataset_eval_times = experiment.run_cv_datasets_evaluation(
                        eval_type=eval_type, datasets=datasets, index=datasets_index, text_col=col, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, batch_size=batch_size, save=True)
                    
                    all_results.append(StaticEmbeddingsExperiment.results_to_pandas(
                        results, origin='pre-trained', experiment_type=experiment_type, text_col=col, model_name=StaticEmbeddingsExperiment.word_embeddings[k]['embedding_name']))
                    
                    experiment.logger.debug('\n' + tabulate(dataset_eval_times, headers="keys", tablefmt="psql", floatfmt=".2f"))

                    print(f'FIM: SENTIMENT ANALYSIS - {experiment.experiment_name}')
                    experiment.logger.info(f'FIM: SENTIMENT ANALYSIS - {experiment.experiment_name}')
            
                    del experiment
                    gc.collect()
                    torch.cuda.empty_cache()
                    
        print(f'\nFIM AVALIAÇÃO DE MODELOS.')
        
        group_by=['model', 'field', 'dataset', 'split', 'experiment', 'origin', 'classifier']
        
        all_results_df = pd.concat(all_results, ignore_index=True)
        
        EmbeddingsExperiment.save_results(
            all_results_df,
            folder=os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_dir),
            excel_file_name=f'{experiment_name}-{classes}.xlsx',
            detail_sheet_name=f'{folds}-Fold CV',
            group_by=group_by)
        
        artifact = wandb.Artifact(name=f'{experiment_name}-{classes}-results', type='results')
        artifact.add(wandb.Table(dataframe=all_results_df), "folds")
        artifact.add(wandb.Table(dataframe=all_results_df.drop(labels=['fold'], axis=1).groupby(group_by).agg('mean').reset_index()), "consolidated")
        wandb.log_artifact(artifact)  
        
        wandb.finish()
    
    if do == 'both' or do == 'train':
        
        print(f'\nINICIO TREINAMENTO DE MODELOS.')

        for col, _ in cols:
            for k, v in StaticEmbeddingsExperiment.word_embeddings.items():
                if k in selected_word_embeddings:
                    print(f'Text column: {col} - Checkpoint: {k}')

                    experiment = StaticEmbeddingsExperiment(
                        model_name=v['embedding_name'],
                        experiment_type = experiment_type,
                        experiment_name=f"{experiment_name}-{v['embedding_name']}",
                        results_dir=results_dir,
                        log_dir=log_dir,
                        log_level=log_level,
                        seed = seed,
                        folds=folds,
                        num_classes=datasets.n_classes,
                        nn_models=nn_models,
                        **v,
                        tokenizer_type=tokenizer_type,
                        num_tokens=num_tokens,
                        sequence_length='MAX',
                        save_classifiers=save_classifiers
                    )

                    # Training on All Data Concatenated
                    experiment.run_downstream_finetuning(datasets=datasets, index=datasets_index, text_col=col)
        
        print(f'\nFIM TREINAMENTO DE MODELOS.')
            
if __name__ == '__main__':
    main()