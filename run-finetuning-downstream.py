from nlp_ptbr.base import *
from nlp_ptbr.data import SentimentDatasets
from nlp_ptbr.experiment import ContextualEmbeddingsExperiment, Experiment, FineTuningSentimentAnalysisExperiment
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
    parser.add_argument('--selected_transformers', dest='selected_transformers', type=str, nargs="+", default=['mbert'])
    parser.add_argument('--experiment_type', type=str, dest='experiment_type', default='fine-tuning downstream', help='Experiment type.', choices=['feature extraction', 'fine-tuning downstream'])
    parser.add_argument('--experiment_name', type=str, dest='experiment_name', default='fine-tuning downstream', help='Experiment name.')
    parser.add_argument('--results_dir', type=str, dest='results_dir', default='fine-tuning downstream', help='Directory where results must be saved.')
    parser.add_argument('--log_dir', type=str, dest='log_dir', default='fine-tuning downstream', help='Directory where logs must be saved.')
    parser.add_argument('--log_level', type=str, dest='log_level', default='WARNING', help='Log level.')
    parser.add_argument('--seed', type=int, dest='seed', default=2017, help='Seed.')
    parser.add_argument('--folds', type=int, dest='folds', default=10, help='Number of Kfolds to evaluate.')
    parser.add_argument('--local_files_only', dest='local_files_only', action='store_true', help='FLAG: Use only local files Hugging Face.')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=32, help='Batch size.')
    parser.add_argument('--train_epochs', type=int, dest='train_epochs', default=3, help='Number of train epochs.')
    parser.add_argument('--save_classifiers', dest='save_classifiers', action='store_true', help='FLAG: Save trained classifiers.')
    parser.add_argument('--disable_tqdm', dest='disable_tqdm', action='store_true', help='FLAG: Disable progress bar.')
    parser.add_argument('--eval_type', type=str, dest='eval_type', default='each', help='Run on each datasets, on all or both.', choices=['each', 'all', 'both'])
    parser.add_argument('--do', type=str, dest='do', default='evaluate', help='What to do: evaluate, train or both.', choices=['evaluate', 'train', 'both'])
    return parser.parse_args()


def main():
    args = parse_args()
    
    field = args.field
    datasets_path = args.datasets_path
    classes = args.classes
    datasets_index = args.datasets_index
    selected_transformers = args.selected_transformers
    experiment_type = args.experiment_type
    experiment_name = args.experiment_name
    results_dir = args.results_dir
    log_dir = args.log_dir
    log_level = args.log_level
    seed = args.seed
    folds = args.folds
    local_files_only = args.local_files_only
    batch_size = args.batch_size
    train_epochs = args.train_epochs
    disable_tqdm = args.disable_tqdm
    eval_type = args.eval_type
    do = args.do
    save_classifiers = args.save_classifiers
    
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
            notes=f'Downstream fine-tuning of transformers language models: {classes}',
            # Set the project where this run will be logged
            project=f"{experiment_type}",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #name=f'{experiment_name}-{wandb.util.generate_id()}',
            name=f'{experiment_name}-{classes}',
            # Track hyperparameters and run metadata
            config={
                'experiment_type': args.experiment_type,
                'models': selected_transformers,
                'datasets_path': datasets.datasets_path,
                'classes': args.classes,
                'datasets_names': datasets.sorted_datasets_names,
                'field': args.field,
                'folds': args.folds,
                'seed': args.seed},
            job_type='contextual-finetuning-downstream',
            tags=['contextual', 'transformers', args.classes, args.experiment_type, args.field])
        
        all_results = []
        all_results_epochs = []
        
        for col, normalization in cols:
            for k, v in ContextualEmbeddingsExperiment.checkpoints.items():
                if k in selected_transformers:
                    print(f'Text column: {col} - Checkpoint: {k}')
                    # The effect of normalization param only applies to BERTweet and BERTweet.BR
                    finetuning_experiment = FineTuningSentimentAnalysisExperiment(
                        **v,
                        experiment_type = experiment_type,
                        results_dir=results_dir,
                        log_dir=log_dir,
                        log_level=log_level,
                        seed = seed,
                        folds=folds,
                        local_files_only=local_files_only,
                        save_classifiers=save_classifiers,
                        normalization=normalization)
                    
                    print(f'\nINÍCIO: SENTIMENT ANALYSIS - {finetuning_experiment.experiment_name}')
                    finetuning_experiment.logger.info(f'\nINÍCIO: SENTIMENT ANALYSIS - {finetuning_experiment.experiment_name}')
                    
                    params = vars(args)
                    finetuning_experiment.logger.debug('\n' + tabulate({"Parâmetro": list(params.keys()),"Valor": list(params.values())}, headers="keys", tablefmt="psql", floatfmt=".2f"))
                    
                    finetuning_experiment.logger.debug('')
                    
                    # Downstream Fine-Tuning Cross-Validation For Each Dataset Individually and All Concatenated
                    results, elapsed_time, dataset_eval_times = finetuning_experiment.run_cv_datasets_evaluation(
                        eval_type=eval_type, datasets=datasets, index=datasets_index, tensorboard = False, num_epochs=train_epochs, batch_size=batch_size, text_col=col, disable_tqdm=disable_tqdm)
                    
                    all_results.append(
                        FineTuningSentimentAnalysisExperiment.results_to_pandas(
                            results, origin='pre-trained', experiment_type=experiment_type, text_col=col, model_name=ContextualEmbeddingsExperiment.checkpoints[k]['model_name'], experiment='cv', last_step_only=True))
                    all_results_epochs.append(
                        FineTuningSentimentAnalysisExperiment.results_to_pandas(
                        results, origin='pre-trained', experiment_type=experiment_type, text_col=col, model_name=ContextualEmbeddingsExperiment.checkpoints[k]['model_name'], experiment='cv', last_step_only=False))
                    
                    finetuning_experiment.logger.debug('\n' + tabulate(dataset_eval_times, headers="keys", tablefmt="psql", floatfmt=".2f"))
                    
                    print(f'FIM: SENTIMENT ANALYSIS - {finetuning_experiment.experiment_name}')
                    finetuning_experiment.logger.info(f'FIM: SENTIMENT ANALYSIS - {finetuning_experiment.experiment_name}')
    
        del finetuning_experiment
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f'\nFIM AVALIAÇÃO DE MODELOS.')
        
        group_by=['model', 'field', 'dataset', 'split', 'experiment', 'origin', 'classifier']
        
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_epochs_df = pd.concat(all_results_epochs, ignore_index=True)
        
        FineTuningSentimentAnalysisExperiment.save_results_consolidated(
            (all_results_df, all_results_epochs_df),
            folder=os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_dir),
            excel_file_name=f'{experiment_name}-{classes}.xlsx',
            detail_sheet_name=f'{folds}-Fold CV',
            epoch_detail_sheet_name=f'{folds}-Fold CV-{train_epochs}-epoch',
            group_by=group_by)
        
        artifact = wandb.Artifact(name=f'ft-downstream-{classes}-results', type='results')
        artifact.add(wandb.Table(dataframe=all_results_epochs_df), "epochs")
        artifact.add(wandb.Table(dataframe=all_results_df), "folds")
        artifact.add(wandb.Table(dataframe=all_results_df.drop(labels=['fold'], axis=1).groupby(group_by).agg('mean').reset_index()), "consolidated")
        wandb.log_artifact(artifact)
        
        wandb.finish()
    
    if do == 'both' or do == 'train':
        
        print(f'\nINICIO TREINAMENTO DE MODELOS - FINE-TUNING DOWNSTREAM TASK.')
        
        for col, normalization in cols:
            for k, v in ContextualEmbeddingsExperiment.checkpoints.items():
                if k in selected_transformers:
                    print(f'Text column: {col} - Checkpoint: {k}')
                    # The effect of normalization param only applies to BERTweet and BERTweet.BR
                    finetuning_experiment = FineTuningSentimentAnalysisExperiment(
                        **v,
                        experiment_type = experiment_type,
                        results_dir=results_dir,
                        log_dir=log_dir,
                        log_level=log_level,
                        seed = seed,
                        folds=folds,
                        local_files_only=local_files_only,
                        save_classifiers=save_classifiers,
                        normalization=normalization)
                    
                    print(f'\nINÍCIO: SENTIMENT ANALYSIS - {finetuning_experiment.experiment_name}')
                    finetuning_experiment.logger.info(f'\nINÍCIO: SENTIMENT ANALYSIS - {finetuning_experiment.experiment_name}')
                    
                    params = vars(args)
                    finetuning_experiment.logger.debug('\n' + tabulate({"Parâmetro": list(params.keys()),"Valor": list(params.values())}, headers="keys", tablefmt="psql", floatfmt=".2f"))

                    finetuning_experiment.logger.debug('')
                    
                    # Downstream Fine-Tuning on All Data Concatenated
                    results, elapsed_time = finetuning_experiment.run_downstream_finetuning(
                        datasets=datasets, index=datasets_index, tensorboard=True, batch_size=batch_size, num_epochs=train_epochs, text_col=col, disable_tqdm=disable_tqdm, save=False)

                    print(f'FIM: SENTIMENT ANALYSIS - {finetuning_experiment.experiment_name}')
                    finetuning_experiment.logger.info(f'FIM: SENTIMENT ANALYSIS - {finetuning_experiment.experiment_name}')
        
        print(f'\nFIM TREINAMENTO DE MODELOS - FINE-TUNING DOWNSTREAM TASK.')
    
    
if __name__ == '__main__':
    main()