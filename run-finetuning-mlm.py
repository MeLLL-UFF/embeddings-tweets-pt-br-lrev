from nlp_ptbr.base import *
from nlp_ptbr.data import SentimentDatasets
from nlp_ptbr.experiment import ContextualEmbeddingsExperiment, Experiment, EmbeddingsExperiment, MaskedLMFineTuningExperiment
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
    parser.add_argument('--classes', type=str, dest='classes', default='multiclass', help='Binary or multiclass.', choices=['multiclass', 'binary', 'mix'])
    parser.add_argument('--datasets_index', type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument('--selected_transformers', dest='selected_transformers', type=str, nargs="+", default=['mbert'])
    parser.add_argument('--experiment_type', type=str, dest='experiment_type', default='fine-tuning mlm', help='Experiment type.', choices=['feature extraction', 'fine-tuning downstream', 'fine-tuning mlm'])
    parser.add_argument('--experiment_name', type=str, dest='experiment_name', default='fine-tuning mlm', help='Experiment name.')
    parser.add_argument('--results_dir', type=str, dest='results_dir', default='fine-tuning mlm', help='Directory where results must be saved.')
    parser.add_argument('--log_dir', type=str, dest='log_dir', default='fine-tuning mlm', help='Directory where logs must be saved.')
    parser.add_argument('--log_level', type=str, dest='log_level', default='WARNING', help='Log level.')
    parser.add_argument('--seed', type=int, dest='seed', default=2017, help='Seed.')
    parser.add_argument('--folds', type=int, dest='folds', default=10, help='Number of Kfolds to evaluate.')
    parser.add_argument('--local_files_only', dest='local_files_only', action='store_true', help='FLAG: Use only local files Hugging Face.')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=32, help='Batch size.')
    parser.add_argument('--train_epochs', type=int, dest='train_epochs', default=3, help='Number of train epochs.')
    parser.add_argument('--disable_tqdm', dest='disable_tqdm', action='store_true', help='FLAG: Disable progress bar.')
    parser.add_argument('--strategy', type=str, dest='strategy', default='loo', help='Strategy on how to continue mlm training.', choices=['loo', 'alldata', 'indata'])
    parser.add_argument('--tokenizer_slow', dest='tokenizer_slow', action='store_true', help='FLAG: Use Slow Tokenizer.')
    parser.add_argument('--only_these_folds', type=int, nargs="+", default=None)
    parser.add_argument('--max_length', type=int, dest='max_length', help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.')
    parser.add_argument('--log_wandb', dest='log_wandb', action='store_true', help='FLAG: Use WandB to log experiment.')
    
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
    strategy = args.strategy
    tokenizer_slow = args.tokenizer_slow
    only_these_folds = args.only_these_folds
    max_length = args.max_length
    log_wandb = args.log_wandb
    
    if field == 'both':
        cols = [('tweet', True), ('tweet_normalized', False)]
    elif field == 'tweet':
        cols = [('tweet', True)]
    else:
        cols = [('tweet_normalized', False)]
    
    datasets = SentimentDatasets(datasets_path=datasets_path, normalize_funcs=NORMALIZE_BERTWEET_STRIP_SPACES, normalize = True, remove_duplicates=False, remove_long_sentences=False, classes=classes)
    
    print(f'\nTREINAMENTO CONTINUADO DOS MODELOS.')
    
    if log_wandb:
        wandb.init(
            entity="bertweetbr",
            notes=f'MLM fine-tuning of transformers language models: {strategy}-{classes}',
            # Set the project where this run will be logged
            project=f"{experiment_type}",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #name=f'{experiment_name}-{wandb.util.generate_id()}',
            name=f'{experiment_name}-{strategy}-{classes}',
            # Track hyperparameters and run metadata
            config={
                'experiment_type': args.experiment_type,
                'models': selected_transformers,
                'datasets_path': datasets.datasets_path,
                'strategy': args.strategy,
                'classes': args.classes,
                'datasets_names': datasets.sorted_datasets_names,
                'field': args.field,
                'folds': args.folds,
                'seed': args.seed},
            job_type='contextual-finetuning-downstream',
            tags=['contextual', 'transformers', args.classes, args.experiment_type, args.strategy, args.field])

    all_results = []

    for col, normalization in cols:
        for k, v in ContextualEmbeddingsExperiment.checkpoints.items():
            if k in selected_transformers:
                print(f'Text column: {col} - Checkpoint: {k}')
                # The effect of normalization param only applies to BERTweet and BERTweet.BR
                mlm_finetuning_experiment = MaskedLMFineTuningExperiment(
                    **v,
                    experiment_type = experiment_type,
                    results_dir=results_dir,
                    log_dir=log_dir,
                    log_level=log_level,
                    seed = seed,
                    folds=folds,
                    local_files_only=local_files_only,
                    normalization=normalization,
                    use_slow_tokenizer=tokenizer_slow,
                    max_length=max_length)

                print(f'\nINÍCIO: SENTIMENT ANALYSIS - {mlm_finetuning_experiment.experiment_name}')
                mlm_finetuning_experiment.logger.info(f'\nINÍCIO: SENTIMENT ANALYSIS - {mlm_finetuning_experiment.experiment_name}')

                params = vars(args)
                mlm_finetuning_experiment.logger.debug('\n' + tabulate({"Parâmetro": list(params.keys()),"Valor": list(params.values())}, headers="keys", tablefmt="psql", floatfmt=".2f"))

                mlm_finetuning_experiment.logger.debug('')
                
                if strategy == 'loo':
                    results = mlm_finetuning_experiment.run_mlm_finetunig_loo(datasets, index=datasets_index, text_col=col, batch_size=batch_size, train_epochs=train_epochs)
                    
                    all_results.append(pd.DataFrame(results))
                elif strategy == 'alldata':
                    results, results_df, elapsed_time, dataset_eval_times = mlm_finetuning_experiment.run_mlm_alldata_datasets(
                        datasets, index=datasets_index, text_col=col, only_these_folds=only_these_folds, batch_size=batch_size, train_epochs=train_epochs)
                    
                    all_results.append(results_df)
                    
                    mlm_finetuning_experiment.logger.debug('\n' + tabulate(dataset_eval_times, headers="keys", tablefmt="psql", floatfmt=".2f"))
                elif strategy == 'indata':
                    results, results_df, elapsed_time, dataset_eval_times = mlm_finetuning_experiment.run_mlm_indata_datasets(
                        datasets, index=datasets_index, text_col=col, only_these_folds=only_these_folds, batch_size=batch_size, train_epochs=train_epochs)
                    
                    all_results.append(results_df)
                    
                    mlm_finetuning_experiment.logger.debug('\n' + tabulate(dataset_eval_times, headers="keys", tablefmt="psql", floatfmt=".2f"))
                else:
                    print('Unknown strategy. Please specify a valid one.')

                print(f'FIM: SENTIMENT ANALYSIS - {mlm_finetuning_experiment.experiment_name}')
                mlm_finetuning_experiment.logger.info(f'FIM: SENTIMENT ANALYSIS - {mlm_finetuning_experiment.experiment_name}')

    del mlm_finetuning_experiment
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'\nFIM TREINAMENTO CONTINUADO DOS MODELOS.')
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    MaskedLMFineTuningExperiment.save_results(
        all_results_df, os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_dir, strategy), f'{experiment_type}-{strategy}-{datasets.classes}.xlsx', sheet_name=strategy)
    
    if log_wandb:
        artifact = wandb.Artifact(name=f'ft-mlm-{strategy}-{classes}-results', type='results')
        artifact.add(wandb.Table(dataframe=all_results_df), "consolidated")
        wandb.log_artifact(artifact)

        wandb.finish()
    
if __name__ == '__main__':
    main()