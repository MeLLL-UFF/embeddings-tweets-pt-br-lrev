from nlp_ptbr.base import *
from nlp_ptbr.data import *
from nlp_ptbr.finetuning import CustomFineTuningForSequenceClassification
from nlp_ptbr.preprocessing import get_hugging_face_tokenizer, NORMALIZE_BERTWEET_STRIP_SPACES

import torch
import re, os, gc, logging, copy, math, time, random
import datasets, transformers
from tqdm.auto import tqdm
from tabulate import tabulate
from datetime import timedelta
from sklearn.base import clone
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from datasets import ClassLabel, Value
import pandas as pd
import numpy as np
from datasets import load_metric
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from joblib import dump, load
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader
import fasttext.util
from gensim.models import KeyedVectors
from nltk.tokenize import TweetTokenizer
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed,
)


def get_checkpoints_stats(series, checkpoints, normalization=False, local_files_only=True, most_common=None, raw_series=None):
    stats = []
    for k, checkpoint in checkpoints.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['checkpoint'], local_files_only=local_files_only, normalization=normalization) 
        stats.append(corpus_tokenizer_stats(series, tokenizer, name=k, most_common=most_common, raw_series=raw_series))
        
    return pd.concat(stats)


def get_datasets_tokenizers_stats(dataset_paths, classes, normalization = False, text_col='tweet_normalized', normalize=False, normalize_funcs=None, selected_checkpoints=None, local_files_only=True, most_common=None, raw_text_col='tweet'):
    l = []
    for f in dataset_paths:
        df = SentimentDatasets.read_and_normalize(f, normalize=normalize, normalize_funcs=normalize_funcs, classes=classes)
        if selected_checkpoints is None:
            selected_checkpoints = ContextualEmbeddingsExperiment.checkpoints
        else:
            selected_checkpoints = {k:v for k, v in ContextualEmbeddingsExperiment.checkpoints.items() if k in selected_checkpoints}
        stats = get_checkpoints_stats(df[text_col], selected_checkpoints, local_files_only=local_files_only, normalization=normalization, most_common=None, raw_series=df[raw_text_col])
        stats = stats.reset_index()
        stats.insert(loc=0, column='dataset', value=get_base_name(f))
        l.append(stats)
    return pd.concat(l, ignore_index=True)


def get_embeddings_stats(series, word_embeddings, most_common=None):
    stats = []
    for k, embedding in word_embeddings.items():
        tokenizer = TweetTokenizer()
        if embedding['load_method'] == 'gensim':
            model = KeyedVectors.load_word2vec_format(embedding['embedding_file_name'])
        else:
            model = fasttext.load_model(embedding['embedding_file_name'])
        stats.append(corpus_tokenizer_stats_from_gensim(series, model, tokenizer, name=k, load_method=embedding['load_method'], most_common=most_common))
        
    return pd.concat(stats)


def get_datasets_embeddings_stats(dataset_paths, classes, text_col='tweet_normalized', normalize=False, normalize_funcs=None, selected_embeddings=['glove', 'word2vec'], most_common=None):
    l = []
    for f in dataset_paths:
        df = SentimentDatasets.read_and_normalize(f, normalize=normalize, normalize_funcs=normalize_funcs, classes=classes)
        if selected_embeddings is None:
            selected_embeddings = StaticEmbeddingsExperiment.word_embeddings
        else:
            selected_embeddings = {k:v for k, v in StaticEmbeddingsExperiment.word_embeddings.items() if k in selected_embeddings}
        stats = get_embeddings_stats(df[text_col], selected_embeddings, most_common=most_common)
        stats = stats.reset_index()
        stats.insert(loc=0, column='dataset', value=get_base_name(f))
        l.append(stats)
    return pd.concat(l, ignore_index=True)

    
class PreTrainedEmbedding:
    def __init__(self, name, length, file_name, ignore_lines=0):
        self.name = name
        self.length = length
        self.file_name = file_name
        #self.base_dir = base_dir
        self.path = self.file_name
        self.ignore_lines = ignore_lines
        self.embedding_num_words = None
        self.embeddings_index = None
        self.embedding_matrix = None
        self.hits = None
        self.misses = None
    
    @staticmethod
    def get_embeddings_index(pre_trained_embeddings_file, ignore_lines=0):
        fin = open(pre_trained_embeddings_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for il in range(ignore_lines):
            # Fasttext: The first line of the file contains the number of words in the vocabulary and the size of the vectors. 
            n, d = map(int, fin.readline().split())
        embeddings_index = {}
        for line in fin:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
        return embeddings_index
    
    @staticmethod
    def get_embedding_matrix(num_tokens, embedding_dim, word_index, embeddings_index):
        hits = 0
        misses = 0
        missed_tokens = []

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros. This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                missed_tokens.append(word)
                misses += 1
        return embedding_matrix, hits, misses, missed_tokens
    
    def generate_embeddings_index(self):
        self.embeddings_index = PreTrainedEmbedding.get_embeddings_index(self.path, ignore_lines=self.ignore_lines)
        self.embedding_num_words = len(self.embeddings_index)
    
    def generate_embedding_matrix(self, vocab_size, word_index):
        if self.embeddings_index is None:
            self.generate_embeddings_index()
        return PreTrainedEmbedding.get_embedding_matrix(num_tokens=vocab_size, embedding_dim=self.length, word_index=word_index, embeddings_index=self.embeddings_index)


class SentimentExperiment(Experiment):
    def __init__(self, experiment_name, experiment_type, model_name, results_dir='sentiment', log_dir='sentiment', log_level='WARNING', seed = 2017, folds=10, num_classes=2):
        super().__init__(experiment_name=experiment_name, experiment_type=experiment_type, model_name=model_name, results_dir=results_dir, log_dir=log_dir, log_level=log_level, seed = seed, folds=folds)
        self.num_classes = num_classes
    
    def __str__ (self):
        me = {k:v for k, v in self.__dict__.items() if type(v) in [str, bool, int, float]}
        return(tabulate({"Parâmetro": list(me.keys()),"Valor": list(me.values())}, headers="keys", tablefmt="psql", floatfmt=".2f"))
    
    @staticmethod
    def fit_and_evaluate_scikit(model, train_features, train_y, val_features=None, val_y=None, return_train_score=False, log_func=None, save=False, model_file_name='filename.joblib', labels_names=None):
        train_metrics = load_metric('nlp_ptbr/metrics.py')
        val_metrics = load_metric('nlp_ptbr/metrics.py')
        
        fit_start_time = time.monotonic()
        model.fit(train_features, train_y)
        fit_end_time = time.monotonic()
        
        if val_features is not None:
            evaluate_start_time = time.monotonic()
            prediction_val_proba = None
            prediction_val = model.predict(val_features)

            if hasattr(model, 'predict_proba'):
                prediction_val_proba = model.predict_proba(val_features)
            val_metrics.add_batch(predictions=prediction_val, references=val_y)
            val_scores = val_metrics.compute(labels=np.unique(val_y), predictions_proba=prediction_val_proba, labels_names=labels_names)
            evaluate_end_time = time.monotonic()

            evaluate_time = evaluate_end_time - evaluate_start_time
        
        train_scores = None
        
        if return_train_score:
            prediction_train_proba = None
            prediction_train = model.predict(train_features)
            if hasattr(model, 'predict_proba'):
                prediction_train_proba = model.predict_proba(train_features)
            train_metrics.add_batch(predictions=prediction_train, references=train_y)
            train_scores = train_metrics.compute(labels=np.unique(train_y), predictions_proba=prediction_train_proba, labels_names=labels_names)
        
        if save:
            dump(model, model_file_name)

        fit_time = fit_end_time - fit_start_time
        
        return model, None if val_features is None else val_scores, train_scores, fit_time, None if val_features is None else evaluate_time


class EmbeddingsExperiment(SentimentExperiment):
    def __init__(
        self, experiment_name, model_name, results_dir='embeddings', log_dir='embeddings', log_level='WARNING', seed = 2017, folds=10, num_classes=2, nn_models=False, sequence_length='MAX',
        experiment_type = 'feature extraction', save_classifiers=False):
            
            super().__init__(experiment_name=experiment_name, experiment_type=experiment_type, model_name=model_name, results_dir=results_dir, log_dir=log_dir, log_level=log_level, seed = seed, folds=folds, num_classes=num_classes)
            self.nn_models = nn_models
            self.sequence_length = sequence_length
            #self.experiment_type = experiment_type
            self.save_classifiers = save_classifiers
            # Experiment Models
            self.experiment_models = dict()
            #https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451
            # Do we need to scale features coming from pre-trained word-embedding?
            self.append_models("lr", make_pipeline(MinMaxScaler(feature_range=(0,1)), LogisticRegression(solver='liblinear', max_iter=2000, dual=False, C=0.9, class_weight='balanced', random_state=self.seed, n_jobs=-1)))
            #self.append_models("lr1", make_pipeline(MinMaxScaler(feature_range=(0,1)), LogisticRegression(solver='saga', max_iter=2000, dual=False, random_state=self.seed, n_jobs=-1)))
            #self.append_models("lsvm", make_pipeline(MinMaxScaler(feature_range=(0,1)), LinearSVC(max_iter=2000, dual=False, random_state=self.seed)))
            ##self.append_models("bsvm", make_pipeline(MinMaxScaler(feature_range=(0,1)), BaggingClassifier(base_estimator=SVC(max_iter=-1, class_weight='balanced', C=0.9, probability=True, random_state=self.seed), random_state=self.seed, max_samples=0.25, bootstrap=True, warm_start=False, n_estimators=10, n_jobs=-1)))
            ##self.append_models("svm", make_pipeline(MinMaxScaler(feature_range=(0,1)), SVC(max_iter=2000, class_weight='balanced', C=0.9, probability=True, random_state=self.seed)))
            #self.append_models("xgb", make_pipeline(MinMaxScaler(feature_range=(0,1)), xgb.XGBClassifier(n_estimators=75, max_depth=3, learning_rate=0.01, eval_metric='logloss', colsample_bytree=0.75, subsample=0.75, n_jobs=-1, use_label_encoder=False, verbosity=0, random_state=self.seed)))
    
    def append_models(self, name, model):
        self.experiment_models[name] = {"model": model}
    
    # Embeddings Experiment
    # Evaluate the model in Each Dataset Using CV Strategy
    def run_cv_datasets_evaluation(self, datasets, index=[], text_col='tweet', label_col = 'class', disable_tqdm=False, fine_tuning_model_language=False, only_these_folds=None, batch_size=128, save=True, eval_type='each'):
        start_time = time.monotonic()
        self.logger.debug('')
        self.logger.debug(f'Início de experimento {self.experiment_name}')

        datasets_files = datasets.get_datasets_path_by_index(index=index)
        
        self.logger.debug(f'Modelos serão avaliados em {len(datasets_files)} datasets: {datasets_files}.')
        
        # If MIX then two results should be outputed
        if datasets.classes == 'mix':
            print(datasets.classes)

        results = {}
        if eval_type == 'each' or eval_type == 'both':
            results = {get_base_name(x): [] for x in datasets_files}
        if eval_type == 'all' or eval_type == 'both':
            results['all'] = []
        
        dataset_eval_times = {'datasets': [], 'elapsed_time': []}

        if eval_type == 'both' or eval_type == 'each':
        
            pbar_datasets = tqdm(enumerate(datasets.get_datasets_by_index(index=index)), total=len(datasets_files), desc=self.experiment_name, disable=disable_tqdm, leave=False)
            
            for i, (f, df) in pbar_datasets:
                start_time_eval = time.monotonic()
                try:
                    pbar_datasets.set_description(f'{i+1}/{len(datasets_files)} : {f}')
                    self.logger.debug(f'Executando experimento com o campo {text_col} do dataset {f}.')
                    
                    dataset_stats = SentimentDatasets.get_df_stats(df, name=f, text_col=text_col)
                    self.logger.debug(f'{f}:\n' + tabulate(dataset_stats.iloc[:, :14], headers='keys', tablefmt='psql', floatfmt=".2f"))
                    
                    max_sequence_length = self.sequence_length
                    if isinstance(self.sequence_length, str):
                        if self.sequence_length == 'MAX':
                            max_sequence_length = int(dataset_stats['max'].values[0])
                    if self.sequence_length is None:
                        max_sequence_length = int(dataset_stats['max'].values[0])
                    
                    self.logger.debug(f'Início da execução do {self.cv.n_splits}-Fold CV com {len(self.experiment_models)} modelos : {list(self.experiment_models.keys())}.')
                    
                    results[f].extend(self.get_cv_run(dataset_file_or_dataframe=df, sequence_length=max_sequence_length, batch_size=batch_size, text_col=text_col, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, dataset_base_name=f, classes=datasets.classes))
                    
                    self.logger.debug(f'Término da execução do {self.cv.n_splits}-Fold CV com {len(self.experiment_models)} modelos : {list(self.experiment_models.keys())}.')

                except Exception as e:
                    torch.cuda.empty_cache()
                    self.logger.exception(e)
                    del df
                    gc.collect()
                    continue
                    
                end_time_eval = time.monotonic()
                dataset_eval_times['datasets'].append(f)
                dataset_eval_times['elapsed_time'].append(timedelta(seconds=end_time_eval - start_time_eval))
        
        if eval_type == 'all' or eval_type == 'both':
            start_time_eval = time.monotonic()
            concatenated_dataset, label_encoder = datasets.get_concatenated_datasets(datasets_files=datasets_files)
            dataset_stats = SentimentDatasets.get_df_stats(concatenated_dataset, name='all', text_col=text_col)
            self.logger.debug(f'All:\n' + tabulate(dataset_stats.iloc[:, :14], headers='keys', tablefmt='psql', floatfmt=".2f"))

            max_sequence_length = self.sequence_length
            if isinstance(self.sequence_length, str):
                if self.sequence_length == 'MAX':
                    max_sequence_length = int(dataset_stats['max'].values[0])
            if self.sequence_length is None:
                max_sequence_length = int(dataset_stats['max'].values[0])

            results['all'].extend(self.get_cv_run(dataset_file_or_dataframe=concatenated_dataset, sequence_length=max_sequence_length, batch_size=batch_size, text_col=text_col, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, dataset_base_name='concatenated', classes=datasets.classes))
            end_time_eval = time.monotonic()
            dataset_eval_times['datasets'].append('concatenated')
            dataset_eval_times['elapsed_time'].append(timedelta(seconds=end_time_eval - start_time_eval))
        
        if save:
            base_path = os.path.join(self.results_dir, self.model_name, 'feature extraction', f'{self.folds}-fold cv')
            results_df = StaticEmbeddingsExperiment.results_to_pandas(results, experiment_type=self.experiment_type, splits = ['train', 'test'], text_col=text_col, model_name=self.model_name)
            
            EmbeddingsExperiment.save_results(
                results_df,
                folder=base_path,
                excel_file_name=f'{text_col}-{self.experiment_name}-{self.folds}-fold-{batch_size}-batch_size.xlsx',
                detail_sheet_name=f'{self.folds}-Fold CV')
        
        end_time = time.monotonic()
        elapsed_time = timedelta(seconds=end_time-start_time)
        
        self.logger.debug('\n' + tabulate(dataset_eval_times, headers="keys", tablefmt="psql", floatfmt=".2f"))
        
        self.logger.debug(f'Fim da avaliação dos modelos. Modelos avaliados em estratégia {self.folds}-Fold CV em {elapsed_time}.')
        
        return results, elapsed_time, dataset_eval_times
    
    
    def get_cv_run(self, dataset_file_or_dataframe, text_col, label='class', only_these_folds=None, batch_size=128, sequence_length=64, disable_tqdm=False, dataset_base_name=None, classes='binary', ready_outputs=None):
        def save_history(history, outputs, dataset='train', fold=None, classifier=None):
            for m in outputs.score.keys():
                history[dataset][m].append(outputs.score[m])

            history[dataset]['time'].append(outputs.time)
            if outputs.loss is not None:
                history[dataset]['loss'].append(outputs.loss)

            if fold is not None:
                if 'fold' not in history[dataset]:
                    history[dataset]['fold'] = []
                history[dataset]['fold'].append(fold)
            if classifier is not None:
                if 'classifier' not in history[dataset]:
                    history[dataset]['classifier'] = []
                history[dataset]['classifier'].append(classifier)
        
        # Here we provide the matrix features already processed from an arbitrary set of rows. Useful for all-data where we concatenate rows in a different arrange other than original datasets
        if ready_outputs is not None:
            fold, embeddings_output, label_encoder = ready_outputs
            
            train_matrix = embeddings_output.train_matrix
            val_matrix = embeddings_output.val_matrix
            train_y = embeddings_output.train_y
            val_y = embeddings_output.val_y
            
            self.logger.debug(f'Início do Treinamento e avaliação dos modelos {list(self.experiment_models.keys())} no fold {fold}.')
            self.logger.debug(f'{1} folds selecionados: {[fold]}')
            
            folds_results = []
            
            self.logger.debug(f'Fold 1 de 1: Fold {fold} de {[fold]} - Treino: {train_matrix.shape[0]} - Validação: {val_matrix.shape[0]}')
            self.logger.debug(f'Início do Treinamento e avaliação dos modelos {list(self.experiment_models.keys())} no fold {fold}.')
            
            history = {'train': {m: [] for m in get_metrics_names(len(label_encoder.classes_))}, 'test': {m: [] for m in get_metrics_names(len(label_encoder.classes_))}}

            if self.save_classifiers:
                model_path_ = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, classes, text_col, dataset_base_name, str(fold))
                os.makedirs(model_path_, exist_ok=True)

                if embeddings_output.fold_tokenizer is not None:
                    tokenizer_file_name = os.path.join(model_path_, 'tokenizer.json')
                    embeddings_output.fold_tokenizer.save(tokenizer_file_name)
                
                if embeddings_output.embedding_matrix is not None:
                    embeddings_file_name = os.path.join(model_path_, 'embeddings.npy')
                    np.save(embeddings_file_name, embeddings_output.embedding_matrix, allow_pickle=False)
                    with open(embeddings_file_name, 'wb') as f:
                        np.save(f, embeddings_output.embedding_matrix)
                
                label_encoder_file_name = os.path.join(model_path_, 'label_encoder.joblib')
                dump(label_encoder, label_encoder_file_name)
            
            for model_name, model in self.experiment_models.items():
                self.logger.debug(f'Treinando e avaliando modelo {model_name} no fold {fold}.')
                
                model_file_name = None
                if self.save_classifiers:
                    model_file_name = os.path.join(model_path_, f'{model_name}.joblib')
                
                m = clone(model['model'])
                _, val_scores, train_scores, fit_time, evaluate_time = SentimentExperiment.fit_and_evaluate_scikit(
                    model=m, train_features=train_matrix, train_y=train_y, val_features=val_matrix, val_y=val_y, return_train_score=True, log_func=self.logger.debug, save=self.save_classifiers, model_file_name=model_file_name, labels_names=label_encoder.classes_)
                
                train_outputs = EpochOutput(train_scores, fit_time, None)
                val_outputs = EpochOutput(val_scores, evaluate_time, None)
                save_history(history, train_outputs, 'train', fold=fold, classifier=model_name)
                save_history(history, val_outputs, 'test', fold=fold, classifier=model_name)
                self.logger.debug(f'Fim do treinamento e avaliação do modelo {model_name} no fold {fold}.')
                
            self.logger.debug(f'Término do Treinamento e avaliação dos modelos {list(self.experiment_models.keys())} no fold {fold}.')
            folds_results.append(history)
            
        else: # Here a dataset is given and we are going to processes all k-folds
            n_folds = self.cv.n_splits if only_these_folds is None else len(only_these_folds)
            list_of_folds = list(range(self.cv.n_splits)) if only_these_folds is None else only_these_folds

            if isinstance(self, ContextualEmbeddingsExperiment):
                gen = self.get_embeddings_for_cv_generator(
                    dataset_file_or_dataframe=dataset_file_or_dataframe, text_col=text_col, label=label, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, batch_size=batch_size, classes=classes)
            else:
                #sequence_length=None, remove_duplicates=False, remove_long_sentences=False, long_sentences=64
                gen = self.get_embeddings_for_cv_generator(
                    dataset_file_or_dataframe=dataset_file_or_dataframe, text_col=text_col, label=label, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, batch_size=batch_size, classes=classes, sequence_length=sequence_length)
            
            pbar_folds = tqdm(enumerate(gen), total=n_folds, leave=False, desc="Fold", disable=disable_tqdm)
            
            self.logger.debug(f'{n_folds} folds selecionados: {list_of_folds}')
            
            folds_results = []
            
            for i, (fold, embeddings_output, label_encoder) in pbar_folds:
                train_matrix = embeddings_output.train_matrix
                val_matrix = embeddings_output.val_matrix
                train_y = embeddings_output.train_y
                val_y = embeddings_output.val_y

                pbar_folds.set_description(f'Fold {i+1} de {n_folds}')
                self.logger.debug(f'Fold {i+1} de {n_folds}: Fold {fold} de {list_of_folds} - Treino: {train_matrix.shape[0]} - Validação: {val_matrix.shape[0]}')
                self.logger.debug(f'Início do Treinamento e avaliação dos modelos {list(self.experiment_models.keys())} no fold {fold}.')

                history = {'train': {m: [] for m in get_metrics_names(len(label_encoder.classes_))}, 'test': {m: [] for m in get_metrics_names(len(label_encoder.classes_))}}

                if self.save_classifiers:
                    model_path_ = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, classes, text_col, dataset_base_name, str(fold))
                    os.makedirs(model_path_, exist_ok=True)
                    
                    if embeddings_output.fold_tokenizer is not None:
                        tokenizer_file_name = os.path.join(model_path_, 'tokenizer.json')
                        embeddings_output.fold_tokenizer.save(tokenizer_file_name)
                    
                    if embeddings_output.embedding_matrix is not None:
                        embeddings_file_name = os.path.join(model_path_, 'embeddings.npy')
                        np.save(embeddings_file_name, embeddings_output.embedding_matrix, allow_pickle=False)
                        with open(embeddings_file_name, 'wb') as f:
                            np.save(f, embeddings_output.embedding_matrix)
                    
                    label_encoder_file_name = os.path.join(model_path_, 'label_encoder.joblib')
                    dump(label_encoder, label_encoder_file_name)
                
                for model_name, model in self.experiment_models.items():
                    self.logger.debug(f'Treinando e avaliando modelo {model_name} no fold {fold}.')
                    
                    model_file_name = None
                    if self.save_classifiers:
                        model_file_name = os.path.join(model_path_, f'{model_name}.joblib')

                    m = clone(model['model'])
                    _, val_scores, train_scores, fit_time, evaluate_time = SentimentExperiment.fit_and_evaluate_scikit(
                        model=m, train_features=train_matrix, train_y=train_y, val_features=val_matrix, val_y=val_y, return_train_score=True, log_func=self.logger.debug, save=self.save_classifiers, model_file_name=model_file_name, labels_names=label_encoder.classes_)

                    train_outputs = EpochOutput(train_scores, fit_time, None)
                    val_outputs = EpochOutput(val_scores, evaluate_time, None)
                    save_history(history, train_outputs, 'train', fold=fold, classifier=model_name)
                    save_history(history, val_outputs, 'test', fold=fold, classifier=model_name)
                    self.logger.debug(f'Fim do treinamento e avaliação do modelo {model_name} no fold {fold}.')

                self.logger.debug(f'Término do Treinamento e avaliação dos modelos {list(self.experiment_models.keys())} no fold {fold}.')
                
                folds_results.append(history)
                
        return folds_results
                
    @staticmethod
    def save_results(
        results,
        folder,
        excel_file_name,
        detail_sheet_name,
        sort_by=['model', 'field', 'dataset', 'split', 'fold', 'experiment', 'origin', 'classifier'],
        group_by = ['model', 'field', 'dataset', 'split', 'experiment', 'origin', 'classifier']):
        
        if sort_by is not None:
            results_df = results.sort_values(by=sort_by)
        
        os.makedirs(folder, exist_ok=True)
        excel_file = os.path.join(folder, excel_file_name)
        
        with pd.ExcelWriter(excel_file, datetime_format='hh:mm:ss') as writer:
            results_df.drop(labels=['fold'], axis=1).groupby(group_by).agg('mean').reset_index().to_excel(writer, sheet_name='consolidated', float_format='%.4f', index=False)
            results_df.to_excel(writer, sheet_name=detail_sheet_name, float_format='%.4f', index=False)
            results_df.drop(labels=['fold'], axis=1).groupby(group_by).agg('std').reset_index().to_excel(writer, sheet_name='consolidated_std', float_format='%.4f', index=False)

    
    @staticmethod
    def results_to_pandas(res, origin='pre-trained', experiment_type = 'feature extraction', splits = ['train', 'test'], model_name='fasttext', text_col='tweet'):
        def to_pandas_cv(res):
            datasets = []
            for split in splits:
                dfs = []
                for i, fold in enumerate(res):
                    df = pd.DataFrame(fold[split])
                    if 'classifier' in df.columns:
                        df.insert(0, 'classifier', df.pop("classifier"))
                    if 'fold' in df.columns:
                        df.insert(0, 'fold', df.pop("fold"))
                    dfs.append(df)
                df = pd.concat(dfs, ignore_index=True)
                df.insert(0, "split", split)
                datasets.append(df)
            return pd.concat(datasets, ignore_index=True)

        datasets = []
        for k, v in res.items():
            if len(v) > 0:
                tmp=to_pandas_cv(v)
                tmp.insert(2, "origin", origin)
                tmp.insert(2, "experiment", experiment_type)
                tmp.insert(0, "dataset", k)
                tmp.insert(0, "field", text_col)
                tmp.insert(0, "model", model_name)
                datasets.append(tmp)

        return pd.concat(datasets, ignore_index=True)


    def load_pipeline(self, classes, text_col, dataset_base_name, classifier, fold):
        if fold is None:
            dataset_base_name = f'{self.model_name}-{StaticEmbeddingsExperiment.GLOBAL_CLASSIFIER_SUFFIX}'
            model_path = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, classes, text_col, dataset_base_name)
        else:
            model_path = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, classes, text_col, dataset_base_name, str(fold)) 
        clf = load(os.path.join(model_path, os.path.join(model_path, f'{classifier}.joblib')))
        label_encoder = load(os.path.join(model_path, os.path.join(model_path, 'label_encoder.joblib')))

        return clf, label_encoder

    @staticmethod
    def get_sentences_as_df(sentences, text_col='tweet', label='class', label_encoder=None):
        if isinstance(sentences, list):
            df_dict = {text_col: sentences, label: label_encoder.inverse_transform([random.randrange(0, 3) for i in range(len(sentences))])}
        elif isinstance(sentences, str):
            df_dict = {text_col: [sentences], label: label_encoder.inverse_transform([random.randrange(0, 3) for i in range(1)])}

        if isinstance(sentences, pd.DataFrame):
            df= sentences.loc[:, [text_col, label]]
        else:
            df = pd.DataFrame(df_dict)

        # Label is expected to be strings thus label_encoder.inverse_transform
        return df

    
class StaticEmbeddingsExperiment(EmbeddingsExperiment):
    GLOBAL_CLASSIFIER_SUFFIX = 'sentiment-analysis-ptbr'
    word_embeddings = {
        'fasttext': {'embedding_name': 'fasttext', 'embedding_length':300, 'embedding_file_name': os.path.join(Experiment.UFF_SENTIMENT_EMBEDDINGS, 'fasttext', 'cc.pt.300.vec'), 'embedding_ignore_lines':1, 'load_method': 'txt'},
        'fasttext_bin': {'embedding_name': 'fasttext', 'embedding_length':300, 'embedding_file_name': os.path.join(Experiment.UFF_SENTIMENT_EMBEDDINGS, 'fasttext', 'cc.pt.300.bin'), 'embedding_ignore_lines':1, 'load_method': 'bin'},
        'glove': {'embedding_name': 'glove', 'embedding_length':300, 'embedding_file_name': os.path.join(Experiment.UFF_SENTIMENT_EMBEDDINGS, 'glove', 'glove_s300.txt'), 'embedding_ignore_lines':1, 'load_method': 'gensim'},
        'word2vec': {'embedding_name': 'word2vec', 'embedding_length':300, 'embedding_file_name': os.path.join(Experiment.UFF_SENTIMENT_EMBEDDINGS, 'word2vec', 'cbow_s300.txt'), 'embedding_ignore_lines':1, 'load_method': 'gensim'}
    }
    
    def __init__(
        self, experiment_name, model_name, results_dir, log_dir, log_level, seed, folds, num_classes, nn_models, sequence_length, save_classifiers, embedding_name, embedding_length, embedding_file_name,
        embedding_ignore_lines, tokenizer_type, num_tokens, load_method='bin', experiment_type = 'feature extraction'):
        
        super().__init__(
            experiment_name=experiment_name, model_name=model_name, results_dir=results_dir, log_dir=log_dir, log_level=log_level,
            seed = seed, folds=folds, num_classes=num_classes, nn_models=nn_models, sequence_length=sequence_length, experiment_type=experiment_type, save_classifiers=save_classifiers)
        
        self.embedding_name = embedding_name
        self.embedding_length = embedding_length
        self.embedding_file_name = embedding_file_name
        
        self.load_method = load_method
        
        if self.load_method == 'bin': # only for fasttext
            self.initialize_embedding_from_bin()
        elif self.load_method == 'gensim':
            self.initialize_embedding_from_gensim()
        else:
            self.embedding_ignore_lines = embedding_ignore_lines
            self.tokenizer_type = tokenizer_type
            self.num_tokens = num_tokens
            
            self.initialize_tokenizer()
            self.initialize_embedding()
    
    def initialize_tokenizer(self):
        self.tokenizer, self.trainer = get_hugging_face_tokenizer(vocab_size=self.num_tokens)
    
    def initialize_embedding(self):
        self.pretrained_word_embedding = PreTrainedEmbedding(name=self.embedding_name, length=self.embedding_length, file_name=self.embedding_file_name, ignore_lines=self.embedding_ignore_lines)
        self.pretrained_word_embedding.generate_embeddings_index()
        
        if self.nn_models:
            self.add_nn_models()
    
    def initialize_embedding_from_bin(self):
        self.pretrained_word_embedding = fasttext.load_model(self.embedding_file_name)
        self.tokenizer=None
        self.trainer=None
        
    def initialize_embedding_from_gensim(self):
        self.tokenizer = TweetTokenizer()
        self.pretrained_word_embedding = KeyedVectors.load_word2vec_format(self.embedding_file_name)
        self.trainer=None    
    
    
    @staticmethod
    def get_predictions_old(sentences, labels, model_name, experiment_type, dataset_base_name, text_col, classifier, fold):
        
        def load_pipeline(model_name, experiment_type, dataset_base_name, text_col, classifier, fold):
            model_path = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, model_name, experiment_type, dataset_base_name, text_col, str(fold)) 
            tokenizer = Tokenizer.from_file(os.path.join(model_path, os.path.join(model_path, f'{dataset_base_name}_{text_col}_fold-{fold}_tokenizer.json')))
            with open(os.path.join(model_path, os.path.join(model_path, f'{dataset_base_name}_{text_col}_fold-{fold}_embeddings.npy')), 'rb') as f:
                embedding_matrix = np.load(f)
            clf = load(os.path.join(model_path, os.path.join(model_path, f'{dataset_base_name}_{text_col}_{classifier}_fold-{fold}.joblib')))
            label_encoder = load(os.path.join(model_path, os.path.join(model_path, f'{dataset_base_name}_{text_col}_fold-{fold}_label_encoder.joblib')))

            return tokenizer, embedding_matrix, clf, label_encoder
        
        def run_pipeline(sentences, labels, tokenizer, embedding_matrix, classifier, label_encoder, batch_size=64, collate_fn=collate_batch, drop_last=False, custom_tokenizer=True):
            embeddings = nn.EmbeddingBag.from_pretrained(embeddings=torch.FloatTensor(embedding_matrix), mode='mean', freeze=True).to(device)
            tokenized_sentences = tokenizer.encode_batch(sentences)
            data_loader = get_data_loader(tokenized_sentences, labels, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=drop_last, custom_tokenizer=custom_tokenizer)
            features = torch.tensor(()).to(device)
            with torch.no_grad():
                for idx, (text, label, length) in enumerate(data_loader):
                    features = torch.cat((features, embeddings(text)), 0)
            torch.cuda.empty_cache()

            predictions = classifier.predict(X=features.cpu().numpy())
            predictions_proba = None
            if hasattr(classifier, 'predict_proba'):
                predictions_proba = classifier.predict_proba(X=features.cpu().numpy())

            return label_encoder.inverse_transform(predictions), predictions, predictions_proba
        
        tokenizer, embedding, clf, label_encoder = load_pipeline(model_name=model_name, experiment_type=experiment_type, dataset_base_name=dataset_base_name, text_col=text_col, classifier=classifier, fold=fold)
        encoded_predictions, predictions, predictions_proba =  run_pipeline(sentences=sentences, labels=labels, tokenizer=tokenizer, embedding_matrix=embedding, classifier=clf, label_encoder=label_encoder)
        
        inference_outputs = InferenceOutput(encoded_predictions, predictions, predictions_proba)
        
        return inference_outputs


    def get_input_matrix(self, sentences):
        if self.load_method == 'bin':
            sentences_matrix = sentences.apply(self.pretrained_word_embedding.get_sentence_vector).to_list()
        elif self.load_method == 'gensim':
            sentences_matrix = sentences.apply(lambda x: get_sentence_vector_for_gensim(self.pretrained_word_embedding, self.tokenizer, x)[0]).to_list()
        
        return sentences_matrix


    def get_predictions(self, sentences, classes, text_col, dataset_base_name, classifier, fold=None, label='class'):
        def run_pipeline(sentences_matrix, classifier, label_encoder):
            predictions = classifier.predict(X=sentences_matrix)
            predictions_proba = None
            if hasattr(classifier, 'predict_proba'):
                predictions_proba = classifier.predict_proba(X=sentences_matrix)

            return predictions, predictions_proba, label_encoder.inverse_transform(predictions)
        
        clf, label_encoder = self.load_pipeline(dataset_base_name=dataset_base_name, text_col=text_col, classifier=classifier, fold=fold, classes=classes)
        sentences_df = EmbeddingsExperiment.get_sentences_as_df(sentences, text_col=text_col, label=label, label_encoder=label_encoder)
        sentences_matrix = self.get_input_matrix(sentences=sentences_df[text_col])
        encoded_predictions, predictions, predictions_proba = run_pipeline(sentences_matrix=sentences_matrix, classifier=clf, label_encoder=label_encoder)

        inference_outputs = InferenceOutput(encoded_predictions, predictions, predictions_proba)

        return inference_outputs
    

    #StaticEmbeddingsExperiment
    def get_embeddings_for_cv_generator(
        self, dataset_file_or_dataframe, text_col='tweet', label='class',
        disable_tqdm=False, only_these_folds=None, batch_size=64, drop_last=False, classes='binary', sequence_length=None, remove_duplicates=False, remove_long_sentences=False, long_sentences=64):
        
        # When using fasttext txt file and a vocabulary is trained from scratch for every dataset/fold
        def custom_tokenized_features_matrix_for_cv_generator(
            dataset_file_or_dataframe, tokenizer, trainer, sequence_length, cv, pretrained_word_embedding,
            text_col='tweet', label='class', remove_duplicates=False, remove_long_sentences=False, long_sentences=64, disable_tqdm=False, only_these_folds=None, batch_size=64, drop_last=False, classes='binary'):

            for fold, train_loader, train_y, val_loader, val_y, fold_tokenizer, label_encoder in SentimentDatasets.custom_tokenized_dataloader_for_cv_generator(
                dataset_file_or_dataframe=dataset_file_or_dataframe, tokenizer=tokenizer, trainer=trainer, sequence_length=sequence_length,
                cv=cv, text_col=text_col, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, remove_duplicates=remove_duplicates,
                remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, batch_size=batch_size, drop_last=drop_last, classes=classes):

                embedding_matrix, hits, misses, missed_tokens = pretrained_word_embedding.generate_embedding_matrix(vocab_size=len(fold_tokenizer.get_vocab()), word_index=fold_tokenizer.get_vocab())
                embeddings = nn.EmbeddingBag.from_pretrained(embeddings=torch.FloatTensor(embedding_matrix), mode='mean', freeze=True).to(device)

                train_features = torch.tensor(()).to(device)
                with torch.no_grad():
                    for idx, (text, label, length) in enumerate(train_loader):
                        train_features = torch.cat((train_features, embeddings(text)), 0)

                torch.cuda.empty_cache()

                val_features = torch.tensor(()).to(device)
                with torch.no_grad():
                    for idx, (text, label, length) in enumerate(val_loader):
                        val_features = torch.cat((val_features, embeddings(text)), 0)

                torch.cuda.empty_cache()
                
                embeddings_outputs = EmbeddingsOutput(train_features.cpu().numpy(), train_y, val_features.cpu().numpy(), val_y, fold_tokenizer, embedding_matrix)
                
                yield fold, embeddings_outputs, label_encoder
        
        # When working with bin file from fasttext (only fasttext)
        def from_bin_features_matrix_for_cv_generator(
            dataset_file_or_dataframe, cv, pretrained_word_embedding,
            text_col='tweet', label='class', remove_duplicates=False, remove_long_sentences=False, long_sentences=64, disable_tqdm=False, only_these_folds=None, batch_size=64, drop_last=False, classes='binary'):

            for fold, (train_text, train_y, val_text, val_y, label_encoder, train_index, test_index) in SentimentDatasets.raw_for_cv_generator(
                dataset_file_or_dataframe=dataset_file_or_dataframe, cv=cv, text_col=text_col, label=label,
                remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, classes=classes):

                train_features = pd.Series(train_text).apply(pretrained_word_embedding.get_sentence_vector)
                val_features = pd.Series(val_text).apply(pretrained_word_embedding.get_sentence_vector)
                
                embeddings_outputs = fold, EmbeddingsOutput(np.array(train_features.to_list()), train_y, np.array(val_features.to_list()), val_y, None, None), label_encoder
                
                yield embeddings_outputs
        
        # When using pre-trained word-embeddings through gensim intergface. (NILC Portuguese word-embeddings)
        def from_gensim_features_matrix_for_cv_generator(
            dataset_file_or_dataframe, cv, pretrained_word_embedding, tokenizer,
            text_col='tweet', label='class', remove_duplicates=False, remove_long_sentences=False, long_sentences=64, disable_tqdm=False, only_these_folds=None, batch_size=64, drop_last=False, classes='binary'):
            
            for fold, (train_text, train_y, val_text, val_y, label_encoder, train_index, test_index) in SentimentDatasets.raw_for_cv_generator(
                dataset_file_or_dataframe=dataset_file_or_dataframe, cv=cv, text_col=text_col, label=label,
                remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, classes=classes):
                
                train_features = pd.Series(train_text).apply(lambda x: get_sentence_vector_for_gensim(pretrained_word_embedding, tokenizer, x)[0])
                val_features = pd.Series(val_text).apply(lambda x: get_sentence_vector_for_gensim(pretrained_word_embedding, tokenizer, x)[0])
                
                embeddings_outputs = fold, EmbeddingsOutput(np.array(train_features.to_list()), train_y, np.array(val_features.to_list()), val_y, None, None), label_encoder
                
                yield embeddings_outputs
        
        
        if self.load_method == 'bin': # Word embedding from bin file. Only for fasttext
            for embeddings_outputs in from_bin_features_matrix_for_cv_generator(
                dataset_file_or_dataframe=dataset_file_or_dataframe, cv=self.cv, pretrained_word_embedding=self.pretrained_word_embedding, text_col=text_col, label=label,
                remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, disable_tqdm=disable_tqdm,
                only_these_folds=only_these_folds, batch_size=batch_size, drop_last=drop_last, classes=classes):
                
                yield embeddings_outputs
                
        elif self.load_method == 'gensim':
            for embeddings_outputs in from_gensim_features_matrix_for_cv_generator(
                dataset_file_or_dataframe=dataset_file_or_dataframe, cv=self.cv, pretrained_word_embedding=self.pretrained_word_embedding, tokenizer=self.tokenizer, text_col=text_col, label=label,
                remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, disable_tqdm=disable_tqdm,
                only_these_folds=only_these_folds, batch_size=batch_size, drop_last=drop_last, classes=classes):
                
                yield embeddings_outputs
        else:
            for embeddings_outputs in custom_tokenized_features_matrix_for_cv_generator(
                dataset_file_or_dataframe=dataset_file_or_dataframe, tokenizer=self.tokenizer, trainer=self.trainer, sequence_length=sequence_length, cv=self.cv, pretrained_word_embedding=self.pretrained_word_embedding,
                text_col=text_col, label=label, remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, disable_tqdm=disable_tqdm,
                only_these_folds=only_these_folds, batch_size=batch_size, drop_last=drop_last, classes=classes):
                
                yield embeddings_outputs

    # Train the model with all labeled data
    def run_downstream_finetuning(self, datasets, index=[], text_col='tweet', label='class'):
        self.logger.debug(f'Início treinamento.')
        start_time = time.monotonic()
        
        datasets_files = datasets.get_datasets_path_by_index(index=index)
        
        self.logger.debug(f'Modelo será treinado a partir de {len(datasets_files)} datasets: {datasets_files}.')
        
        concatenated_dataset, label_encoder = datasets.get_concatenated_datasets(datasets_files=datasets_files)
        
        if self.load_method == 'bin': # Word embedding from bin file. Only for fasttext
            train_features = concatenated_dataset[text_col].apply(self.pretrained_word_embedding.get_sentence_vector)
        elif self.load_method == 'gensim':
            train_features = concatenated_dataset[text_col].apply(lambda x: get_sentence_vector_for_gensim(self.pretrained_word_embedding, self.tokenizer, x)[0])

        train_y = concatenated_dataset[label].to_list()

        _, embeddings_output, label_encoder = None, EmbeddingsOutput(np.array(train_features.to_list()), train_y, None, None, None, None), label_encoder
        
        train_matrix = embeddings_output.train_matrix
        train_y = embeddings_output.train_y
        
        model_path_ = os.path.join(
            Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, datasets.classes, text_col, f'{self.model_name}-{StaticEmbeddingsExperiment.GLOBAL_CLASSIFIER_SUFFIX}')
        os.makedirs(model_path_, exist_ok=True)
        label_encoder_file_name = os.path.join(model_path_, 'label_encoder.joblib')
        dump(label_encoder, label_encoder_file_name)
        
        for model_name, model in self.experiment_models.items():
            model_file_name = os.path.join(model_path_, f'{model_name}.joblib')
            print(f'Treinando e salvando modelo {model_name}.')
            m = clone(model['model'])
            _, val_scores, train_scores, fit_time, evaluate_time = SentimentExperiment.fit_and_evaluate_scikit(
                model=m, train_features=train_matrix, train_y=train_y, val_features=None, val_y=None, return_train_score=True, log_func=self.logger.debug, save=self.save_classifiers, model_file_name=model_file_name, labels_names=label_encoder.classes_)

        end_time = time.monotonic()
        self.logger.debug(f'Término treinamento.')

class ContextualEmbeddingsExperiment(EmbeddingsExperiment):
    BERT_LAYERS_RANGE = (9, 12)
    
    checkpoints = {
        'xlmr-base': {'model_name': 'xlmr-base', 'experiment_name':'xlmr-base-sentiment-analysis-ptbr', 'checkpoint':'xlm-roberta-base'},
        'xlmt-base': {'model_name': 'xlmt-base', 'experiment_name':'xlmt-base-sentiment-analysis-ptbr', 'checkpoint':'cardiffnlp/twitter-xlm-roberta-base'},
        'mbert': {'model_name': 'mbert', 'experiment_name':'mbert-sentiment-analysis-ptbr', 'checkpoint':'bert-base-multilingual-cased'},
        'bertimbau-base': {'model_name': 'bertimbau-base', 'experiment_name':'bertimbau-base-sentiment-analysis-ptbr', 'checkpoint':'neuralmind/bert-base-portuguese-cased'},
        'bertweetbr': {'model_name': 'bertweetbr', 'experiment_name': 'bertweetbr-sentiment-analysis-ptbr', 'checkpoint': 'melll-uff/bertweetbr'},
        # DAPT Models
        'mbert-dapt': {'model_name': 'mbert-dapt', 'experiment_name':'mbert-dapt-sentiment-analysis-ptbr', 'checkpoint': os.path.join(SentimentExperiment.UFF_CACHE_HOME, 'dapt', 'mbert', 'checkpoint-1000000')},
        'bertimbau-base-dapt': {'model_name': 'bertimbau-base-dapt', 'experiment_name':'bertimbau-base-dapt-sentiment-analysis-ptbr', 'checkpoint': os.path.join(SentimentExperiment.UFF_CACHE_HOME, 'dapt', 'bertimbau', 'checkpoint-1000000')},
        'xlmr-base-dapt': {'model_name': 'xlmr-base-dapt', 'experiment_name':'xlmr-base-dapt-sentiment-analysis-ptbr', 'checkpoint': os.path.join(SentimentExperiment.UFF_CACHE_HOME, 'dapt', 'xlmr', 'checkpoint-2000000')}
    }
    
    def __init__(self,
                 experiment_name,
                 model_name,
                 results_dir,
                 log_dir,
                 log_level,
                 seed,
                 folds,
                 num_classes,
                 nn_models,
                 sequence_length,
                 save_classifiers,
                 experiment_type = 'feature extraction',
                 checkpoint = 'bert-base-multilingual-cased',
                 tokenizer_checkpoint=None,
                 fine_tuned_checkpoint=None,
                 use_slow_tokenizer = False,
                 normalization = False,
                 local_files_only=False,
                 max_length=None):
        
        super().__init__(experiment_name=experiment_name, model_name=model_name, results_dir=results_dir, log_dir=log_dir, log_level=log_level, seed = seed, folds=folds, num_classes=num_classes, nn_models=nn_models,
                         sequence_length=sequence_length, experiment_type=experiment_type, save_classifiers=save_classifiers)
        
        self.checkpoint = checkpoint
        self.fine_tuned_checkpoint = fine_tuned_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.use_slow_tokenizer = use_slow_tokenizer
        # if working with raw Tweets, normalization should be set to True, otherwise False (default, as we normaly run experiments on already normalized tweets
        self.normalization = normalization
        self.local_files_only=local_files_only
        
        self.initialize_hf_model()
        
        if max_length is None:
            self.max_length = list(self.tokenizer.max_model_input_sizes.values())[0]
        else:
            self.max_length = max_length
            
        #self.logger.warning(f'{experiment_name} started.')
    
    def initialize_hf_model(self):
        self.init_bert_tokenizer_and_model()
        self.bert_model = self.bert_model.to(device)
        self.bert_model.eval()
        
        if self.nn_models:
            self.add_nn_models()
    
    def init_bert_tokenizer_and_model(self):
        self.tokenizer, self.bert_model = self.get_bert_tokenizer_and_model()
    
    def get_bert_tokenizer_and_model(self):
        if self.fine_tuned_checkpoint is None:
            bert_model = AutoModel.from_pretrained(self.checkpoint, output_hidden_states = True, local_files_only=self.local_files_only)
        else:
            bert_model = AutoModel.from_pretrained(self.fine_tuned_checkpoint, output_hidden_states = True)
        
        tokenizer_checkpoint = self.checkpoint if self.tokenizer_checkpoint is None else self.tokenizer_checkpoint

        bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=not self.use_slow_tokenizer, local_files_only=self.local_files_only, normalization=self.normalization)
        #bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

        return bert_tokenizer, bert_model
    

    def get_features_from_tokenized_dataloader_(self, dataloader, label_out_name='labels', layers_range=(9, 12)):
        features = torch.tensor(()).to(device)
        labels = torch.tensor(()).to(device)
        for idx, batch in enumerate(dataloader):
            labels = torch.cat((labels, batch[label_out_name].to(device)), 0)
            batch = {k: v.to(device) for k, v in batch.items() if k != label_out_name}
            with torch.no_grad():
                output = torch.sum(torch.stack(self.bert_model(**batch)['hidden_states'][layers_range[0]:layers_range[1]+1], dim=0), dim=0)
                features = torch.cat((features, output[:,0,:]), 0)
        return features, labels
    

    def get_features_from_tokenized_dataloader(self, train_dataloader, eval_dataloader, label_out_name='labels', layers_range=(9, 12)):
        train_features = torch.tensor(()).to(device)
        train_labels = torch.tensor(()).to(device)
        for idx, batch in enumerate(train_dataloader):
            train_labels = torch.cat((train_labels, batch[label_out_name].to(device)), 0)
            batch = {k: v.to(device) for k, v in batch.items() if k != label_out_name}
            with torch.no_grad():
                output = torch.sum(torch.stack(self.bert_model(**batch)['hidden_states'][layers_range[0]:layers_range[1]+1], dim=0), dim=0)
                train_features = torch.cat((train_features, output[:,0,:]), 0)

        val_features = torch.tensor(()).to(device)
        val_labels = torch.tensor(()).to(device)
        for idx, batch in enumerate(eval_dataloader):
            val_labels = torch.cat((val_labels, batch[label_out_name].to(device)), 0)
            batch = {k: v.to(device) for k, v in batch.items() if k != label_out_name}
            with torch.no_grad():
                output = torch.sum(torch.stack(self.bert_model(**batch)['hidden_states'][layers_range[0]:layers_range[1]+1], dim=0), dim=0)
                val_features = torch.cat((val_features, output[:,0,:]), 0)

        embeddings_outputs = EmbeddingsOutput(train_features.cpu().numpy(), train_labels.cpu().numpy().astype(int), val_features.cpu().numpy(), val_labels.cpu().numpy().astype(int), None, None)

        return embeddings_outputs
    

    def get_embeddings_for_cv_generator(
        self, dataset_file_or_dataframe, text_col='tweet', label='class',
        disable_tqdm=False, only_these_folds=None, batch_size=64, drop_last=False, classes='binary', label_out_name='labels', layers_range=(9, 12)):
        
        for i, (fold, (train_dataloader, eval_dataloader), label_encoder) in enumerate(
            SentimentDatasets.dataloader_for_cv_generator(
                tokenizer=self.tokenizer, dataset_file_or_dataframe=dataset_file_or_dataframe, cv=self.cv, batch_size=batch_size,
                text_col=text_col, disable_tqdm=disable_tqdm, max_length=self.max_length, only_these_folds=only_these_folds, label_out_name=label_out_name)):
            
            embeddings_outputs = self.get_features_from_tokenized_dataloader(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, label_out_name=label_out_name, layers_range=layers_range)

            yield fold, embeddings_outputs, label_encoder
    
    # Train the model with all labeled data
    def run_downstream_finetuning(self, datasets, index=[], text_col='tweet', label='class', batch_size=64):
        self.logger.debug(f'Início treinamento.')
        start_time = time.monotonic()
        
        datasets_files = datasets.get_datasets_path_by_index(index=index)

        self.logger.debug(f'Modelo será treinado a partir de {len(datasets_files)} datasets: {datasets_files}.')

        raw_dataset, label_encoder = datasets.raw_hf_train_dataset(datasets_files=datasets_files, text_col=text_col, drop_label_column=False)
        
        train_dataloader = SentimentDatasets.get_tokenized_dataloader(raw_datasets=raw_dataset, tokenizer=self.tokenizer, batch_size=batch_size, max_length=self.max_length, text_col=text_col)
        
        train_features, train_labels = self.get_features_from_tokenized_dataloader_(train_dataloader)

        _, embeddings_output, label_encoder = None, EmbeddingsOutput(train_features.cpu().numpy(), train_labels.cpu().numpy().astype(int), None, None, None, None), label_encoder

        train_matrix = embeddings_output.train_matrix
        train_y = embeddings_output.train_y
        
        model_path_ = os.path.join(
            Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, datasets.classes, text_col, f'{self.model_name}-{StaticEmbeddingsExperiment.GLOBAL_CLASSIFIER_SUFFIX}')
        os.makedirs(model_path_, exist_ok=True)
        label_encoder_file_name = os.path.join(model_path_, 'label_encoder.joblib')
        dump(label_encoder, label_encoder_file_name)

        for model_name, model in self.experiment_models.items():
            model_file_name = os.path.join(model_path_, f'{model_name}.joblib')
            print(f'Treinando e salvando modelo {model_name}.')
            m = clone(model['model'])
            _, val_scores, train_scores, fit_time, evaluate_time = SentimentExperiment.fit_and_evaluate_scikit(
                model=m, train_features=train_matrix, train_y=train_y, val_features=None, val_y=None, return_train_score=True, log_func=self.logger.debug, save=self.save_classifiers, model_file_name=model_file_name, labels_names=label_encoder.classes_)
        
        end_time = time.monotonic()
        self.logger.debug(f'Término treinamento.')


    def get_input_matrix(self, df, text_col, label, drop_label_column=False, label_encoder=None):
        raw_dataset, _ = SentimentDatasets.get_raw_hf_train_dataset(train_df=df, text_col=text_col, label=label, drop_label_column=drop_label_column, label_encoder=label_encoder)
        train_dataloader = SentimentDatasets.get_tokenized_dataloader(raw_datasets=raw_dataset, tokenizer=self.tokenizer, batch_size=64, max_length=self.max_length, text_col=text_col)
        train_features, train_labels = self.get_features_from_tokenized_dataloader_(train_dataloader)
        _, embeddings_output, _ = None, EmbeddingsOutput(train_features.cpu().numpy(), train_labels.cpu().numpy().astype(int), None, None, None, None), label_encoder

        sentences_matrix = embeddings_output.train_matrix
        sentences_y = embeddings_output.train_y

        return sentences_matrix, sentences_y


    def get_predictions(self, sentences, classes, text_col, dataset_base_name, classifier, fold=None, label='class'):
        def run_pipeline(sentences_matrix, classifier, label_encoder):
            predictions = classifier.predict(X=sentences_matrix)
            predictions_proba = None
            if hasattr(classifier, 'predict_proba'):
                predictions_proba = classifier.predict_proba(X=sentences_matrix)

            return predictions, predictions_proba, label_encoder.inverse_transform(predictions)
        
        clf, label_encoder = self.load_pipeline(dataset_base_name=dataset_base_name, text_col=text_col, classifier=classifier, fold=fold, classes=classes)
        sentences_df = EmbeddingsExperiment.get_sentences_as_df(sentences, text_col=text_col, label=label, label_encoder=label_encoder)
        sentences_matrix, sentences_y = self.get_input_matrix(df=sentences_df, text_col=text_col, label=label, label_encoder=label_encoder)
        predictions, predictions_proba, encoded_predictions = run_pipeline(sentences_matrix=sentences_matrix, classifier=clf, label_encoder=label_encoder)
        
        inference_outputs = InferenceOutput(encoded_predictions, predictions, predictions_proba)

        return inference_outputs


class FineTuningSentimentAnalysisExperiment(Experiment):
    GLOBAL_CLASSIFIER_SUFFIX = 'sentiment-analysis-ptbr'
    def __init__(self,
                 model_name='mbert', experiment_name='mbert-sentiment-analysis-ptbr', checkpoint = 'bert-base-multilingual-cased', tokenizer_checkpoint= None,
                 local_files_only=False, results_dir='contextual', log_dir='contextual', log_level='WARNING', folds=10, seed=2017, normalization=False, max_length=None, experiment_type='fine-tuning downstream', save_classifiers=False):
        
        super().__init__(experiment_name=experiment_name, experiment_type=experiment_type, model_name=model_name, results_dir=results_dir, log_dir=log_dir, log_level=log_level, seed = seed, folds=folds)
        self.checkpoint = checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.local_files_only = local_files_only
        self.normalization = normalization
        self.save_classifiers = save_classifiers
        
        tokenizer_checkpoint = self.checkpoint if self.tokenizer_checkpoint is None else self.tokenizer_checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, local_files_only=self.local_files_only, normalization=self.normalization)

        if max_length is None:
            self.max_length = list(self.tokenizer.max_model_input_sizes.values())[0]
        else:
            self.max_length = max_length
            
        self.local_files_only = local_files_only
    
    @staticmethod
    def results_to_pandas(res, experiment='cv', last_step_only=True, logging_strategy="epoch", origin='pre-trained', experiment_type = 'fine-tuning downstream', splits = ['train', 'test'], model_name='mbert', text_col='tweet'):
        def to_pandas_cv(res, last_step_only=True, logging_strategy="epoch"):
            datasets = []
            for split in splits:
                if last_step_only:
                    df = [pd.DataFrame(fold[split]).tail(1) for fold in res]
                    df = pd.concat(df, ignore_index=True).reset_index().rename(columns={'index':'fold'})
                else:
                    dfs = []
                    for i, fold in enumerate(res):
                        df = pd.DataFrame(fold[split]).reset_index().rename(columns={'index':logging_strategy})
                        df.insert(0, "fold", i)
                        dfs.append(df)
                    df = pd.concat(dfs, ignore_index=True)
                df.insert(0, "split", split)
                datasets.append(df)
            return pd.concat(datasets, ignore_index=True)

        def to_pandas_loo(res, last_step_only=True, logging_strategy="epoch"):
            datasets = []
            for split in splits:
                if last_step_only:
                    df=pd.DataFrame(v[split]).tail(1)
                else:
                    df=pd.DataFrame(v[split]).reset_index().rename(columns={'index':logging_strategy})  
                df.insert(0, "split", split)
                datasets.append(df)
            return pd.concat(datasets, ignore_index=True)
        
        datasets = []
        for k,v in res.items():
            if len(v) > 0:
                if experiment == 'cv':
                    tmp=to_pandas_cv(v, last_step_only=last_step_only, logging_strategy=logging_strategy)
                elif experiment == 'loo':
                    tmp=to_pandas_loo(v, last_step_only=last_step_only, logging_strategy=logging_strategy)
                tmp.insert(2, "classifier", "1-layer nn")
                tmp.insert(2, "origin", origin)
                tmp.insert(2, "experiment", experiment_type)
                tmp.insert(0, "dataset", k)
                tmp.insert(0, "field", text_col)
                tmp.insert(0, "model", model_name)
                datasets.append(tmp)

        return pd.concat(datasets, ignore_index=True)
    
    def save_results(
        self, results, experiment_folder, experiment_subfolder, excel_file_name, detail_sheet_name, epoch_detail_sheet_name, experiment='cv', logging_strategy='epoch', splits = ['train', 'test'], text_col='tweet',
        sort_by=['model', 'field', 'dataset', 'split', 'fold', 'experiment', 'origin', 'classifier'],
        group_by = ['model', 'field', 'dataset', 'split', 'experiment', 'origin', 'classifier']):
        
        results_df = FineTuningSentimentAnalysisExperiment.results_to_pandas(
            results, experiment=experiment, last_step_only=True, logging_strategy=logging_strategy, splits = splits, text_col=text_col, model_name=self.model_name)
        results_df_epochs = FineTuningSentimentAnalysisExperiment.results_to_pandas(
            results, experiment=experiment, last_step_only=False, logging_strategy=logging_strategy, splits = splits, text_col=text_col, model_name=self.model_name)
        
        if sort_by is not None:
            results_df = results_df.sort_values(by=sort_by)
        
        os.makedirs(self.results_dir, exist_ok=True)
        base_path = os.path.join(self.results_dir, self.model_name, experiment_folder, experiment_subfolder)
        os.makedirs(base_path, exist_ok=True)

        excel_file = os.path.join(base_path, excel_file_name)
        
        with pd.ExcelWriter(excel_file, datetime_format='hh:mm:ss') as writer:
            results_df.drop(labels=['fold'], axis=1).groupby(group_by).agg('mean').reset_index().to_excel(writer, sheet_name='consolidated', float_format='%.4f', index=False)
            results_df.to_excel(writer, sheet_name=detail_sheet_name, float_format='%.4f', index=False)
            results_df.drop(labels=['fold'], axis=1).groupby(group_by).agg('std').reset_index().to_excel(writer, sheet_name='consolidated_std', float_format='%.4f', index=False)

            results_df_epochs.to_excel(writer, sheet_name=epoch_detail_sheet_name, float_format='%.4f', index=False)
    
    
    @staticmethod
    def save_results_consolidated(
        results,
        folder,
        excel_file_name,
        detail_sheet_name,
        epoch_detail_sheet_name,
        sort_by=['model', 'field', 'dataset', 'split', 'fold', 'experiment', 'origin', 'classifier'],
        group_by = ['model', 'field', 'dataset', 'split', 'experiment', 'origin', 'classifier']):
        
        if sort_by is not None:
            results_df = results[0].sort_values(by=sort_by)
            results_epochs_df = results[1].sort_values(by=sort_by)
        
        os.makedirs(folder, exist_ok=True)
        excel_file = os.path.join(folder, excel_file_name)
        
        with pd.ExcelWriter(excel_file, datetime_format='hh:mm:ss') as writer:
            results_df.drop(labels=['fold'], axis=1).groupby(group_by).agg('mean').reset_index().to_excel(writer, sheet_name='consolidated', float_format='%.4f', index=False)
            results_df.to_excel(writer, sheet_name=detail_sheet_name, float_format='%.4f', index=False)

            results_epochs_df.to_excel(writer, sheet_name=epoch_detail_sheet_name, float_format='%.4f', index=False)
            results_df.drop(labels=['fold'], axis=1).groupby(group_by).agg('std').reset_index().to_excel(writer, sheet_name='consolidated_std', float_format='%.4f', index=False)
    
    # Evaluate the model LOO at dataset level
    def run_loo_datasets_evaluation(self, datasets, index=[], num_epochs=1, tensorboard_points = 3, min_log = 1000, tensorboard = True, logging_strategy="epoch", batch_size=64, save=True, text_col='tweet', disable_tqdm=False, empty_steps=10, max_length=None):
        
        self.logger.debug(f'Início da avaliação dos modelos na estratégia leave-one-out.')
        start_time = time.monotonic()
        
        datasets_files = datasets.get_datasets_path_by_index(index=index)
        
        self.logger.debug(f'Modelos serão avaliados em {len(datasets_files)} datasets: {datasets_files}.')
        
        results = {get_base_name(x): [] for x in datasets_files}
        
        dataset_eval_times = {'datasets': [], 'elapsed_time': []}
        
        for i, raw_datasets, train_datasets, test_datasets in datasets.raw_datasets_hf_generator(datasets_files=datasets_files, text_col=text_col):
            start_time_eval = time.monotonic()
            
            self.logger.debug(f'Dataset de validação {i+1}/{len(train_datasets)+1}: {test_datasets[0]}')
            self.logger.debug(f'Datasets de treinamento {len(train_datasets)}: {train_datasets}')
            
            self.logger.debug(f'Instanciando Hugging Face AutoModel e AutoTokenizer {self.model_name}: {self.checkpoint}')
            # Instantiate language model for every validation dataset
            custom_finetuning = CustomFineTuningForSequenceClassification(model_name=self.model_name, name=self.experiment_name, checkpoint=self.checkpoint, tokenizer_checkpoint=self.tokenizer_checkpoint, local_files_only=self.local_files_only, num_labels=datasets.n_classes)
            
            self.logger.debug(f'Obtendo tokenized Dataloader para dataset de treino em batches de {batch_size}')
            train_dataloader, eval_dataloader = SentimentDatasets.get_tokenized_dataloader(
                raw_datasets=raw_datasets, tokenizer=custom_finetuning.tokenizer, batch_size=batch_size, max_length=max_length, text_col=text_col)
            
            num_training_steps = num_epochs * len(train_dataloader)
            self.logger.debug(f'Base de treinamento possui {len(train_dataloader)} batches de {batch_size} sentenças.')
            self.logger.debug(f'Base de validação possui {len(eval_dataloader)} batches de {batch_size} sentenças.')
            self.logger.debug(f'Treinando modelo em {num_epochs} épocas em um total de {num_training_steps} de passos.')

            # Acho que deveria ser salvo aqui
            # Falta fold e dataset_name
            model_path_ = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, text_col)
            
            results[test_datasets[0]] = custom_finetuning.fit(
                train_dataloader, eval_dataloader, metric='f1', greater=True, logging_strategy=logging_strategy,
                logging_steps=1, epochs=num_epochs, suffix=test_datasets[0], tensorboard=tensorboard, log_func=self.logger.debug, disable_tqdm=disable_tqdm, empty_steps=empty_steps, save=self.save_classifiers)
            
            del custom_finetuning
            gc.collect()
            torch.cuda.empty_cache()
            
            end_time_eval = time.monotonic()
            
            dataset_eval_times['datasets'].append(test_datasets[0])
            dataset_eval_times['elapsed_time'].append(timedelta(seconds=end_time_eval - start_time_eval))

        if save:
            self.save_results(results, experiment_folder='finetuning-downstream', experiment_subfolder='loo',
                              excel_file_name=f'{text_col}-{self.experiment_name}-loo-{num_epochs}-epoch-{batch_size}-batch_size.xlsx',
                              detail_sheet_name='loo', epoch_detail_sheet_name=f'loo-{num_epochs}-epoch', experiment='loo', text_col=text_col)
        
        end_time = time.monotonic()
        
        elapsed_time = timedelta(seconds=end_time-start_time)
        
        self.logger.debug(f'Fim da avaliação dos modelos. Modelos avaliados na estratégia leave-one-out em {elapsed_time}.')
        
        return results, elapsed_time, dataset_eval_times
    
    def cv_evaluation(self, dataset_path, tokenizer, num_epochs=1, tensorboard_points = 3, min_log = 1000, tensorboard = True, logging_strategy="epoch", batch_size=64, suffix='Fold', text_col='tweet', disable_tqdm=False, empty_steps=10, max_length=None, num_classes=2, only_these_folds=None, ready_outputs=None, label_encoder=None, dataset_base_name=None):
        
        results = []
        
        if ready_outputs is not None:
            fold, train_dataloader, eval_dataloader, label_encoder = ready_outputs
            self.logger.debug(f'Fold: {fold}')
            # Instantiate language model for every fold
            self.logger.debug(f'Instanciando Hugging Face AutoModel e AutoTokenizer {self.model_name}: {self.checkpoint}')
            custom_finetuning = CustomFineTuningForSequenceClassification(
                model_name=self.model_name, name=self.experiment_name, checkpoint=self.checkpoint, tokenizer_checkpoint=self.tokenizer_checkpoint, local_files_only=self.local_files_only, labels=[c.upper() for c in label_encoder.classes_])
            
            num_training_steps = num_epochs * len(train_dataloader)
            self.logger.debug(f'Base de treinamento possui {len(train_dataloader)} batches de {batch_size} sentenças de tamanho máximo {max_length}.')
            self.logger.debug(f'Base de validação possui {len(eval_dataloader)} batches de {batch_size} sentenças de tamanho máximo {max_length}.')
            self.logger.debug(f'Treinando modelo em {num_epochs} épocas em um total de {num_training_steps} de passos.')

            model_path_ = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, dataset_base_name, text_col, str(fold))
            
            results.append(
                custom_finetuning.fit(train_dataloader, eval_dataloader, logging_strategy=logging_strategy,
                                      logging_steps=1, epochs=num_epochs, checkpoints_dir=model_path_,
                                      suffix=f'{suffix}-{fold}', tensorboard=tensorboard, log_func=self.logger.debug, disable_tqdm=disable_tqdm, empty_steps=empty_steps, save=self.save_classifiers))
            
            del custom_finetuning
            gc.collect()
            torch.cuda.empty_cache()
            
        else:
            for j, (fold, (train_dataloader, eval_dataloader), encoder) in enumerate(
                SentimentDatasets.dataloader_for_cv_generator(
                    tokenizer=tokenizer, dataset_file_or_dataframe=dataset_path, cv=self.cv, batch_size=batch_size, text_col=text_col, disable_tqdm=disable_tqdm, max_length=max_length, only_these_folds=only_these_folds)):
                
                self.logger.debug(f'Fold: {fold}')
                
                if label_encoder is None:
                    label_encoder = encoder

                # Instantiate language model for every fold
                self.logger.debug(f'Instanciando Hugging Face AutoModel e AutoTokenizer {self.model_name}: {self.checkpoint}')
                custom_finetuning = CustomFineTuningForSequenceClassification(
                    model_name=self.model_name, name=self.experiment_name, checkpoint=self.checkpoint, tokenizer_checkpoint=self.tokenizer_checkpoint, local_files_only=self.local_files_only, labels=[c.upper() for c in label_encoder.classes_])

                num_training_steps = num_epochs * len(train_dataloader)
                self.logger.debug(f'Base de treinamento possui {len(train_dataloader)} batches de {batch_size} sentenças de tamanho máximo {max_length}.')
                self.logger.debug(f'Base de validação possui {len(eval_dataloader)} batches de {batch_size} sentenças de tamanho máximo {max_length}.')
                self.logger.debug(f'Treinando modelo em {num_epochs} épocas em um total de {num_training_steps} de passos.')

                model_path_ = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, dataset_base_name, text_col, str(fold))

                results.append(
                    custom_finetuning.fit(train_dataloader, eval_dataloader, logging_strategy=logging_strategy,
                                          logging_steps=1, epochs=num_epochs, checkpoints_dir=model_path_,
                                          suffix=f'{suffix}-{fold}', tensorboard=tensorboard, log_func=self.logger.debug, disable_tqdm=disable_tqdm, empty_steps=empty_steps, save=self.save_classifiers))

                del custom_finetuning
                gc.collect()
                torch.cuda.empty_cache()
                
        return results
    
    # Fine-Tuning Downstream
    # Evaluate the model in Each Dataset Using CV Strategy Or/And All Datasets Concatenated
    def run_cv_datasets_evaluation(self, datasets, index=[], eval_type='each', num_epochs=1,
                                   tensorboard_points = 3, min_log = 1000, tensorboard = True, logging_strategy="epoch", batch_size=64, save=True, text_col='tweet', disable_tqdm=False, only_these_folds=None):
        
        self.logger.debug(f'Início avaliação em estratégia {self.folds}-Fold CV.')
        
        start_time = time.monotonic()
        
        datasets_files = datasets.get_datasets_path_by_index(index=index)
        
        self.logger.debug(f'Modelos serão avaliados em {len(datasets_files)} datasets: {datasets_files}.')
        
        results = {}
        
        if eval_type == 'each' or eval_type == 'both':
            results = {get_base_name(x): [] for x in datasets_files}
        if eval_type == 'all' or eval_type == 'both':
            results['all'] = []
        
        dataset_eval_times = {'datasets': [], 'elapsed_time': []}
        
        if eval_type == 'both' or eval_type == 'each':
            
            for f, df in datasets.get_datasets_by_index(index=index):
                start_time_eval = time.monotonic()
                base_name = f
                self.logger.debug(f'Avaliando dataset {base_name} em {num_epochs} épocas em batches de {batch_size} sentenças de tamanho máximo {self.max_length}.')
                results[base_name].extend(self.cv_evaluation(dataset_path=df, dataset_base_name=f, tokenizer=self.tokenizer, max_length=self.max_length, num_epochs=num_epochs, tensorboard_points = tensorboard_points, min_log = min_log, tensorboard = tensorboard, logging_strategy=logging_strategy, batch_size=batch_size, suffix=base_name, text_col=text_col, disable_tqdm=disable_tqdm, num_classes=datasets.n_classes, only_these_folds=only_these_folds))
                end_time_eval = time.monotonic()
                dataset_eval_times['datasets'].append(base_name)
                dataset_eval_times['elapsed_time'].append(timedelta(seconds=end_time_eval - start_time_eval))
        
        if eval_type == 'all' or eval_type == 'both':
            start_time_eval = time.monotonic()
            concatenated_dataset, label_encoder = datasets.get_concatenated_datasets(datasets_files=datasets_files)
            results['all'].extend(self.cv_evaluation(concatenated_dataset, dataset_base_name='concatenated', tokenizer=self.tokenizer, max_length=self.max_length, num_epochs=num_epochs, tensorboard_points = tensorboard_points, min_log = min_log, tensorboard = tensorboard, logging_strategy=logging_strategy, batch_size=batch_size, suffix='all', text_col=text_col, disable_tqdm=disable_tqdm, num_classes=datasets.n_classes, only_these_folds=only_these_folds, label_encoder=label_encoder))
            end_time_eval = time.monotonic()
            dataset_eval_times['datasets'].append('concatenated')
            dataset_eval_times['elapsed_time'].append(timedelta(seconds=end_time_eval - start_time_eval))
        
        if save:
            self.save_results(
                results, experiment_folder='finetuning-downstream', experiment_subfolder=f'{self.folds}-fold cv',
                excel_file_name=f'{text_col}-{self.experiment_name}-{self.folds}-fold-{num_epochs}-epoch-{batch_size}-batch_size.xlsx', detail_sheet_name=f'{self.folds}-Fold CV',
                epoch_detail_sheet_name=f'{self.folds}-Fold CV-{num_epochs}-epoch', experiment='cv', text_col=text_col)
        
        end_time = time.monotonic()
        elapsed_time = timedelta(seconds=end_time-start_time)
        self.logger.debug(f'Fim da avaliação dos modelos. Modelos avaliados em estratégia {self.folds}-Fold CV em {elapsed_time}.')
        
        return results, elapsed_time, dataset_eval_times
        
    # Train the model with all labeled data
    def run_downstream_finetuning(self, datasets, index=[], num_epochs=1, tensorboard_points = 3, min_log = 1000,
                                  tensorboard = True, batch_size=64, logging_strategy="epoch", save=True, text_col='tweet', disable_tqdm=False, empty_steps=10):
        
        self.logger.debug(f'Início fine-tuning.')
        start_time = time.monotonic()
        
        datasets_files = datasets.get_datasets_path_by_index(index=index)
        
        self.logger.debug(f'Modelo será treinado a partir de {len(datasets_files)} datasets: {datasets_files}.')
        self.logger.debug('Obtendo datasets no formato Hugging Face.')

        raw_dataset, label_encoder = datasets.raw_hf_train_dataset(datasets_files=datasets_files, text_col=text_col, drop_label_column=False)
        
        self.logger.debug(f'Instanciando Hugging Face AutoModel e AutoTokenizer {self.model_name}: {self.checkpoint}')
        custom_finetuning = CustomFineTuningForSequenceClassification(
            model_name=self.model_name, name=self.experiment_name, checkpoint=self.checkpoint, tokenizer_checkpoint=self.tokenizer_checkpoint, local_files_only=self.local_files_only, labels=[c.upper() for c in label_encoder.classes_])
        
        self.logger.debug(f'Obtendo tokenized Dataloader para dataset de treino em batches de {batch_size} e sentenças de até {self.max_length}.')
        train_dataloader = SentimentDatasets.get_tokenized_dataloader(raw_datasets=raw_dataset, tokenizer=custom_finetuning.tokenizer, batch_size=batch_size, max_length=self.max_length, text_col=text_col)
        
        num_training_steps = num_epochs * len(train_dataloader)
        
        self.logger.debug(f'Executando fine tuning do modelo em {num_epochs} épocas em um total de {num_training_steps} de passos.')
        
        model_path_ = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, self.model_name, self.experiment_type, datasets.classes, text_col, f'{self.model_name}-{FineTuningSentimentAnalysisExperiment.GLOBAL_CLASSIFIER_SUFFIX}')
        
        history = custom_finetuning.fit(
            train_dataloader, logging_strategy=logging_strategy, logging_steps=1, epochs=num_epochs, save=self.save_classifiers,
            checkpoints_dir=model_path_, suffix='all', tensorboard = tensorboard, log_func=self.logger.debug, disable_tqdm=disable_tqdm, empty_steps=empty_steps)
        
        self.tuned_model = custom_finetuning
        
        if save:
            self.save_results({'all': history}, experiment_folder='finetuning-downstream', experiment_subfolder='model-training',
                              excel_file_name=f'{text_col}-{self.experiment_name}-model-training-{num_epochs}-epoch-{batch_size}-batch_size.xlsx',
                              detail_sheet_name='model-training',
                              epoch_detail_sheet_name=f'model-training-{num_epochs}-epoch', experiment='loo', splits = ['train'], text_col=text_col)
        
        end_time = time.monotonic()
        
        elapsed_time = timedelta(seconds=end_time-start_time)
        
        self.logger.debug(f'Fim fine-tuning. Modelo treinado em {elapsed_time}.')
        
        return history, elapsed_time
    
    
# Based on https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py
# Old version of it still available at: https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py

class MaskedLMFineTuningExperiment(Experiment):
    def __init__(self,
                 model_name='mbert',
                 experiment_name='mbert-mlm-continued-training',
                 checkpoint = 'bert-base-multilingual-cased',
                 local_files_only=False,
                 results_dir='contextual',
                 log_dir='contextual',
                 log_level='WARNING',
                 folds=10,
                 seed=2017,
                 normalization=False,
                 max_length=None,
                 experiment_type='fine-tuning mlm',
                 use_slow_tokenizer = False):
        
        super().__init__(experiment_name=experiment_name, experiment_type=experiment_type, model_name=model_name, results_dir=results_dir, log_dir=log_dir, log_level=log_level, seed = seed, folds=folds)
        self.checkpoint = checkpoint
        self.local_files_only = local_files_only
        self.normalization = normalization
        self.use_slow_tokenizer = use_slow_tokenizer
        
        self.init_bert_tokenizer_and_model()
        
        if max_length is None:
            self.max_length = list(self.tokenizer.max_model_input_sizes.values())[0]
        else:
            self.max_length = max_length
            
        
    def init_bert_tokenizer_and_model(self):
        self.tokenizer, self.model = self.get_tokenizer_and_model()
    
    
    def get_tokenizer_and_model(self):
        config = AutoConfig.from_pretrained(self.checkpoint)
        
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=self.local_files_only, normalization=self.normalization, use_fast=not self.use_slow_tokenizer)

        model = AutoModelForMaskedLM.from_pretrained(
            self.checkpoint, local_files_only=self.local_files_only, config=config)
        
        model.resize_token_embeddings(len(tokenizer))
        
        return tokenizer, model
    
    
    def run(self,
            raw_datasets, max_seq_length=None, line_by_line = False, preprocessing_num_workers = None, overwrite_cache = False,
            mlm_probability = 0.15, per_device_train_batch_size = 8, per_device_eval_batch_size = 8, weight_decay = 0.0, learning_rate = 5e-5,
            gradient_accumulation_steps = 1, max_train_steps = None, num_train_epochs = 3, lr_scheduler_type = 'linear', num_warmup_steps = 0,
            output_dir='./tmp/test-mlm', print_sample_rows=False, disable_tqdm=True, pad_to_max_length=False):
        
        accelerator = Accelerator()
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        self.logger.info(f'GPU Count: {torch.cuda.device_count()}')
        self.logger.info(accelerator.state)
        
        # Setup logging, we only want one process per machine to log things on the screen.
        # accelerator.is_local_main_process is only True for one process per machine.
        self.logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            # If passed along, set the training seed now.
        if self.seed is not None:
            set_seed(self.seed)
        
        model = copy.deepcopy(self.model)
        tokenizer = copy.deepcopy(self.tokenizer)
        
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                self.logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
                )
                max_seq_length = 1024
        else:
            if max_seq_length > tokenizer.model_max_length:
                self.logger.warning(
                    f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(max_seq_length, tokenizer.model_max_length)
        
        if line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples["text"],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
        
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not overwrite_cache,
            )
            
            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result
            
            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                #batch_size=500,
                num_proc=preprocessing_num_workers,
                load_from_cache_file=not overwrite_cache,
            )

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        
        if print_sample_rows:
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                self.logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        
        
        # Data collator
        # This one will take care of randomly masking the tokens.
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader)


        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
        else:
            num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Train!
        total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Max sequence length = {max_seq_length}")
        self.logger.info(f"  Num examples = {len(train_dataset)}")
        self.logger.info(f"  Num Epochs = {num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        #progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar = tqdm(range(max_train_steps), disable=disable_tqdm)
        completed_steps = 0
        
        perplexities = []
        
        for epoch in range(num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= max_train_steps:
                    break

            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather(loss.repeat(per_device_eval_batch_size)))

            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]
            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")
                
            perplexities.append(perplexity)

            self.logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        if output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            
        return perplexities
    
    
    @staticmethod
    def save_results(results_df, folder, excel_file_name, sheet_name='loo'):
        os.makedirs(folder, exist_ok=True)
        excel_file = os.path.join(folder, excel_file_name)

        with pd.ExcelWriter(excel_file, datetime_format='hh:mm:ss') as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, float_format='%.4f', index=False)

    
    def run_mlm_finetunig_loo(self, datasets, index=[], text_col='tweet', label_col = 'class', disable_tqdm=False, batch_size=8, train_epochs=3):
        output_models = {'model': [], 'field': [], 'dataset': [], 'index': [], 'classes': [], 'trainsets': [], 'model_output': [], 'perplexities': [], 'elapsed_time': []}

        base_dir = os.path.join(Experiment.UFF_SENTIMENT_MODELS, self.experiment_type, 'loo', self.model_name, datasets.classes)
        
        self.logger.debug('')
        self.logger.debug(f'Início de experimento {self.experiment_name}')

        datasets_files = datasets.get_datasets_path_by_index(index=index)

        self.logger.debug(f'Modelos serão tunados considerando um conjunto de {len(datasets_files)} datasets: {datasets_files}.')
        
        for i, raw_datasets, train_datasets, test_datasets, label_encoder in datasets.raw_datasets_hf_generator(datasets_files=datasets_files, 
                                                                                                                text_col=text_col, label=label_col, label_out_name='labels', drop_label_column=True):
            start_time = time.monotonic()
            self.logger.debug(f'Executando treinamento continuado LOO com {train_datasets} como bases de treino e {test_datasets} como base de teste.')
            output_dir = os.path.join(base_dir, str(i))

            perplexities = self.run(
                raw_datasets, max_seq_length=self.max_length, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=train_epochs, output_dir=output_dir)
            
            output_models['model'].append(self.model_name)
            output_models['field'].append(text_col)
            output_models['dataset'].append(test_datasets[0])
            output_models['index'].append(i)
            output_models['classes'].append(datasets.classes)
            output_models['trainsets'].append(train_datasets)
            output_models['model_output'].append(output_dir)
            output_models['perplexities'].append(perplexities)
            end_time = time.monotonic()
            
            output_models['elapsed_time'].append(timedelta(seconds=end_time - start_time))
            
            torch.cuda.empty_cache()
            gc.collect()

        save_dir = os.path.join(base_dir)
        MaskedLMFineTuningExperiment.save_results(pd.DataFrame(output_models), save_dir, f'{self.model_name}-{datasets.classes}-loo.xlsx', sheet_name='loo')

        return output_models
    
    
    # In-Data model in Each Dataset
    def run_mlm_indata_datasets(self, datasets, index=[], text_col='tweet', label_col = 'class', label_out_name='labels', drop_label_column=True, disable_tqdm=False, only_these_folds=None, batch_size=128, train_epochs=3):
        start_time = time.monotonic()

        self.logger.debug('')
        self.logger.debug(f'Início de experimento {self.experiment_name}')

        datasets_files = datasets.get_datasets_path_by_index(index=index)

        self.logger.debug(f'Modelos serão avaliados em {len(datasets_files)} datasets: {datasets_files}.')

        results = {get_base_name(x): [] for x in datasets_files}
        dataset_eval_times = {'datasets': [], 'elapsed_time': []}
        pbar_datasets = tqdm(enumerate(datasets.get_datasets_by_index(index=index)), total=len(datasets_files), desc=self.experiment_name, disable=disable_tqdm, leave=False)
        dataset_eval_times = {'datasets': [], 'elapsed_time': []}

        base_dir = os.path.join(Experiment.UFF_SENTIMENT_MODELS, self.experiment_type, 'indata', self.model_name, datasets.classes)

        for i, (f, df) in pbar_datasets:
            start_time_eval = time.monotonic()

            base_dir_dataset = os.path.join(base_dir, f)

            try:
                pbar_datasets.set_description(f'{i+1}/{len(datasets_files)} : {f}')
                self.logger.debug(f'Executando finetuning mlm indata com o campo {text_col} do dataset {f}.')
                
                #dataset_stats = SentimentDatasets.get_df_stats(df, name=f, text_col=text_col)
                #self.logger.debug(f'{f}:\n' + tabulate(dataset_stats.iloc[:, :14], headers='keys', tablefmt='psql', floatfmt=".2f"))

                self.logger.debug(f'Início da execução do {self.cv.n_splits}-Fold CV finetuning mlm indata para dataset : {f}.')

                output_models = []

                for j, (fold, raw_datasets, label_encoder) in enumerate(
                    SentimentDatasets.raw_dataset_for_cv_generator(
                        dataset_file_or_dataframe=df, cv=self.cv, text_col=text_col, label=label_col, label_out_name=label_out_name,
                        disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, classes=datasets.classes, drop_label_column = drop_label_column)):

                    start_fold_eval = time.monotonic()

                    output_dir = os.path.join(base_dir_dataset, str(fold))

                    perplexities = self.run(
                        raw_datasets, max_seq_length=self.max_length, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=train_epochs, output_dir=output_dir)

                    end_fold_eval = time.monotonic()
                    
                    # se for passado um conjunto de índices aleatório teremos um resultado incorreto ao processar o arquivo de modelos fine-tunados. sem tempo de corrigir agora!
                    output_models.append({'model': self.model_name, 'field': text_col, 'dataset': f, 'index': i, 'fold': fold, 'classes': datasets.classes, 'model_output': output_dir, 'perplexities': perplexities, 'elapsed_time': timedelta(seconds=end_fold_eval - start_fold_eval)})

                results[f].extend(output_models)

                save_dir = os.path.join(base_dir_dataset)
                MaskedLMFineTuningExperiment.save_results(pd.DataFrame(output_models), save_dir, f'{self.model_name}-{f}-{datasets.classes}-indata.xlsx', sheet_name='indata')

                self.logger.debug(f'Término da execução do {self.cv.n_splits}-Fold CV finetuning mlm indata para dataset : {f}.')

            except Exception as e:
                torch.cuda.empty_cache()
                self.logger.exception(e)
                del df
                gc.collect()
                continue

            end_time_eval = time.monotonic()
            dataset_eval_times['datasets'].append(f)
            dataset_eval_times['elapsed_time'].append(timedelta(seconds=end_time_eval - start_time_eval))

        end_time = time.monotonic()
        elapsed_time = timedelta(seconds=end_time-start_time)

        self.logger.debug('\n' + tabulate(dataset_eval_times, headers="keys", tablefmt="psql", floatfmt=".2f"))

        self.logger.debug(f'Fim da avaliação dos modelos. Modelos avaliados em estratégia {self.folds}-Fold CV em {elapsed_time}.')

        results_df = pd.concat([pd.DataFrame(v) for v in results.values()], axis=0, ignore_index=True)

        save_dir = os.path.join(base_dir)
        MaskedLMFineTuningExperiment.save_results(results_df, save_dir, f'{self.model_name}-{datasets.classes}-indata.xlsx', sheet_name='indata')

        return results, results_df, elapsed_time, dataset_eval_times
    
    
    # All-Data model in Each Dataset
    def run_mlm_alldata_datasets(
        self, datasets, index=[], text_col='tweet_normalized', label='class', label_out_name='labels', disable_tqdm=False, only_these_folds=None, batch_size=128, train_epochs=3, drop_label_column = True):

        start_time = time.monotonic()

        self.logger.debug('')
        self.logger.debug(f'Início de experimento {self.experiment_name}')

        datasets_files = datasets.get_datasets_path_by_index(index=index)

        self.logger.debug(f'Modelos serão avaliados em {len(datasets_files)} datasets: {datasets_files}.')

        results = {get_base_name(x): [] for x in datasets_files}
        dataset_eval_times = {'datasets': [], 'elapsed_time': []}
        dataset_eval_times = {'datasets': [], 'elapsed_time': []}

        base_dir = os.path.join(Experiment.UFF_SENTIMENT_MODELS, self.experiment_type, 'alldata', self.model_name, datasets.classes)

        for i, train, test, train_datasets, test_datasets, label_encoder in datasets.dataset_loo_generator(datasets_files=datasets_files):
            start_time_eval = time.monotonic()
            f = test_datasets[0]

            self.logger.debug(f'Executando finetuning mlm alldata com o campo {text_col} dos dataset {test_datasets + train_datasets}.')
            base_dir_dataset = os.path.join(base_dir, f)

            n_folds = self.cv.n_splits if only_these_folds is None else len(only_these_folds)
            initial_list_of_folds = list(range(self.cv.n_splits)) if only_these_folds is None else only_these_folds

            self.logger.debug(f'Início da execução do {self.cv.n_splits}-Fold CV finetuning mlm alldata para dataset : {f}.')
            
            output_models = []

            for fold, (train_text, train_y, val_text, val_y, encoder, train_index, test_index) in datasets.fold_data_generator(X=test[text_col], y=test[label], cv=self.cv, only_these_folds=only_these_folds):
                model_output_dir = os.path.join(base_dir_dataset, f'fold-{str(fold)}')
                list_of_folds = initial_list_of_folds[:]
                list_of_folds.remove(fold)

                print(f'Trainset : {train_datasets} + folds {list_of_folds} of dataset {f}')
                print(f'Val set is fold {fold} of dataset {f}')

                df_concat = pd.concat([train.loc[:, [text_col, label]], pd.DataFrame({text_col: train_text, label: train_y})], axis=0, ignore_index=True)

                data_files = {}
                data_files['train'] = Dataset.from_pandas(pd.DataFrame({text_col: df_concat[text_col], label_out_name: df_concat[label]}))
                data_files['validation'] = Dataset.from_pandas(pd.DataFrame({text_col: val_text, label_out_name: val_y}))
                # name of the label col must be named 'label'
                raw_datasets=DatasetDict(data_files)

                new_features = raw_datasets["train"].features.copy()
                new_features[label_out_name] = ClassLabel(names=[c.upper() for c in label_encoder.classes_])
                raw_datasets = raw_datasets.cast(new_features)

                if drop_label_column:
                    raw_datasets=raw_datasets.remove_columns([label_out_name])

                start_fold_eval = time.monotonic()

                perplexities = self.run(
                    raw_datasets, max_seq_length=self.max_length, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=train_epochs, output_dir=model_output_dir)

                end_fold_eval = time.monotonic()

                output_models.append({
                    'model': self.model_name, 'field': text_col, 'dataset': f, 'index': i, 'fold': fold, 'classes': datasets.classes,
                    'trainset': f'{train_datasets} + folds {list_of_folds} of dataset {f}',
                    'testset': f'Fold {fold} of dataset {f}',
                    'model_output': model_output_dir, 'perplexities': perplexities,
                    'elapsed_time': timedelta(seconds=end_fold_eval - start_fold_eval)})

            results[f].extend(output_models)

            save_dir = os.path.join(base_dir_dataset)
            MaskedLMFineTuningExperiment.save_results(pd.DataFrame(output_models), save_dir, f'{self.model_name}-{f}-{datasets.classes}-alldata.xlsx', sheet_name='alldata')

            self.logger.debug(f'Término da execução do {self.cv.n_splits}-Fold CV finetuning mlm alldata para dataset : {f}.')

            end_time_eval = time.monotonic()
            dataset_eval_times['datasets'].append(f)
            dataset_eval_times['elapsed_time'].append(timedelta(seconds=end_time_eval - start_time_eval))

        end_time = time.monotonic()
        elapsed_time = timedelta(seconds=end_time-start_time)

        self.logger.debug('\n' + tabulate(dataset_eval_times, headers="keys", tablefmt="psql", floatfmt=".2f"))

        self.logger.debug(f'Fim da avaliação dos modelos. Modelos avaliados em estratégia {self.folds}-Fold CV em {elapsed_time}.')

        results_df = pd.concat([pd.DataFrame(v) for v in results.values()], axis=0, ignore_index=True)

        save_dir = os.path.join(base_dir)
        MaskedLMFineTuningExperiment.save_results(results_df, save_dir, f'{self.model_name}-{datasets.classes}-alldata.xlsx', sheet_name='alldata')

        return results, results_df, elapsed_time, dataset_eval_times
    
    @staticmethod
    def run_contextual_on_finetuned_models(
        results_from='fine-tuning mlm', results_dir='embeddings-ft', strategy = 'loo', classes = 'binary', experiment_type='feature extraction', disable_tqdm=False, batch_size=128, save=False, selected_transformers=None, seed=2017, folds=10, label_out_name='labels', layers_range=(9, 12), train_epochs=3, eval_type='each', local_files_only=False):

        results_path = os.path.join(ContextualEmbeddingsExperiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_from, strategy)
        results_file = os.path.join(results_path, f'{results_from}-{strategy}-{classes}.xlsx')
        results_df = pd.read_excel(results_file)
        
        if strategy == 'loo':
            results_list = [
                MlmOutput(row[0], row[1], 'class', row[2], row[3], row[4], None, row[5]) for row in zip(
                    results_df['model'], results_df['field'], results_df['classes'], results_df['dataset'], results_df['index'], results_df['model_output']) if os.path.exists(row[-1])]
        else:
            results_list = [
                MlmOutput(row[0], row[1], 'class', row[2], row[3], row[4], row[5], row[6]) for row in zip(
                    results_df['model'], results_df['field'], results_df['classes'], results_df['dataset'], results_df['index'], results_df['fold'], results_df['model_output']) if os.path.exists(row[-1])]
        
        datasets = SentimentDatasets(normalize_funcs=NORMALIZE_BERTWEET_STRIP_SPACES, normalize = True, remove_duplicates=False, remove_long_sentences=False, classes=classes)
        selected_transformers = list(ContextualEmbeddingsExperiment.checkpoints.keys()) if selected_transformers is None else selected_transformers
        
        if strategy == 'alldata':
            results_all = {m: {get_base_name(x): [] for x in datasets.get_datasets_path_by_index()} for m in selected_transformers}
        else:
            results_all = []
        
        for i, mlm in enumerate(results_list):
            print(f'{i+1} de {len(results_list)}')
            if mlm.model_name in selected_transformers:
                print(f'Target dataset is {mlm.dataset}')

                normalization = True if mlm.text_col == 'tweet' else False
                
                if experiment_type == 'feature extraction':
                    
                    if strategy == 'loo':
                        print(f'Extracting embeddings from {mlm.model_output} and evaluating classifiers on {mlm.classes} dataset {mlm.dataset} using {mlm.text_col}.')
                    else:
                        print(f'Target fold is {mlm.fold}')
                        print(f'Extracting embeddings from {mlm.model_output} and evaluating classifiers on fold {mlm.fold} of {mlm.classes} dataset {mlm.dataset} using {mlm.text_col}.')
                    
                    experiment = ContextualEmbeddingsExperiment(
                        model_name=mlm.model_name,
                        experiment_type=experiment_type,
                        experiment_name=f'embeddings-{mlm.model_name}-from-finetuned-mlm-{strategy}-{mlm.index}-{i}',
                        results_dir=os.path.join(results_dir, strategy),
                        log_dir=os.path.join(results_dir, strategy),
                        log_level='WARNING',
                        seed = seed,
                        folds=folds,
                        num_classes=datasets.n_classes,
                        nn_models=False,
                        checkpoint = mlm.model_output,
                        tokenizer_checkpoint=ContextualEmbeddingsExperiment.checkpoints[mlm.model_name]['checkpoint'],
                        sequence_length='MAX',
                        save_classifiers=False,
                        normalization=normalization,
                        local_files_only=local_files_only)
                elif experiment_type == 'fine-tuning downstream':
                    
                    if strategy == 'loo':
                        print(f'Finetuning {mlm.model_output} in {train_epochs} epochs on {mlm.classes} dataset {mlm.dataset} using {mlm.text_col}.')
                    else:
                        print(f'Target fold is {mlm.fold}')
                        print(f'Finetuning {mlm.model_output} in {train_epochs} epochs on fold {mlm.fold} of {mlm.classes} dataset {mlm.dataset} using {mlm.text_col}.')
                    
                    results_all_epochs = []
                    experiment = FineTuningSentimentAnalysisExperiment(
                        model_name=mlm.model_name,
                        experiment_type=experiment_type,
                        experiment_name=f'downstream-ft-{mlm.model_name}-from-finetuned-mlm-{strategy}-{mlm.index}-{i}',
                        results_dir=os.path.join(results_dir, strategy),
                        log_dir=os.path.join(results_dir, strategy),
                        log_level='WARNING',
                        seed = seed,
                        folds=folds,
                        checkpoint = mlm.model_output,
                        tokenizer_checkpoint=ContextualEmbeddingsExperiment.checkpoints[mlm.model_name]['checkpoint'],
                        normalization=normalization,
                        local_files_only=local_files_only)
                    
                if strategy == 'alldata':
                    train, val, label_encoder = datasets.get_df_for_alldata(
                        target_dataset=mlm.dataset, target_fold=mlm.fold, text_col=mlm.text_col, label=mlm.label)
                    
                    print(f'train: {len(train)} val: {len(val)}')
                    
                    raw_datasets, _ = datasets.raw_hf_train_dataset(
                        train_df=train, val_df=val, text_col=mlm.text_col, label=mlm.label, label_out_name=label_out_name, label_names=[c.upper() for c in label_encoder.classes_], drop_label_column=False)
                    
                    print(f'Generating dataloader...')
                    
                    train_dataloader, eval_dataloader = SentimentDatasets.get_tokenized_dataloader(
                        raw_datasets=raw_datasets, tokenizer=experiment.tokenizer, text_col=mlm.text_col, shuffle=True, batch_size=batch_size, max_length=experiment.max_length)
                    
                    if experiment_type == 'feature extraction':
                        print(f'Extracting features from {mlm.model_name}: {mlm.model_output}...')

                        features = experiment.get_features_from_tokenized_dataloader(
                            train_dataloader, eval_dataloader, label_out_name=label_out_name, layers_range=layers_range)

                        print(f'Training classifiers...')
                        results_all[mlm.model_name][mlm.dataset].extend(
                            experiment.get_cv_run(
                                dataset_file_or_dataframe=None,
                                text_col=mlm.text_col, label=mlm.label, only_these_folds=None, batch_size=batch_size, sequence_length=experiment.sequence_length,
                                disable_tqdm=disable_tqdm, dataset_base_name=None, classes=classes, ready_outputs=(mlm.fold, features, label_encoder)))
                    elif experiment_type == 'fine-tuning downstream':
                        print(f'Fitting the model...')
                        results_all[mlm.model_name][mlm.dataset].extend(experiment.cv_evaluation(
                            dataset_path=None, dataset_base_name=mlm.dataset, tokenizer=experiment.tokenizer, max_length=experiment.max_length, num_epochs=train_epochs, tensorboard = False, batch_size=batch_size, suffix=mlm.dataset, text_col=mlm.text_col, disable_tqdm=disable_tqdm, num_classes=datasets.n_classes, only_these_folds=None, ready_outputs=(mlm.fold, train_dataloader, eval_dataloader, label_encoder)))
                    print('')
                else:
                    if experiment_type == 'feature extraction':
                        results, elapsed_time, dataset_eval_times = experiment.run_cv_datasets_evaluation(
                            datasets=datasets, index=[mlm.index], text_col=mlm.text_col, disable_tqdm=disable_tqdm, only_these_folds=None if mlm.fold is None else [mlm.fold], batch_size=batch_size, save=save)
                        results_all.append(
                            ContextualEmbeddingsExperiment.results_to_pandas(
                                results, origin=f'fine-tuning-mlm {strategy}', experiment_type=experiment_type, text_col=mlm.text_col, model_name=mlm.model_name))
                    elif experiment_type == 'fine-tuning downstream':
                        results, elapsed_time, dataset_eval_times = experiment.run_cv_datasets_evaluation(
                            datasets=datasets, index=[mlm.index], text_col=mlm.text_col, disable_tqdm=disable_tqdm, only_these_folds=None if mlm.fold is None else [mlm.fold], batch_size=batch_size, save=save, tensorboard = False, num_epochs=train_epochs, eval_type=eval_type)
                        results_all.append(
                            FineTuningSentimentAnalysisExperiment.results_to_pandas(
                                results, origin=f'fine-tuning-mlm {strategy}', experiment_type=experiment_type, text_col=mlm.text_col, model_name=mlm.model_name, experiment='cv', last_step_only=True))
                        results_all_epochs.append(
                            FineTuningSentimentAnalysisExperiment.results_to_pandas(
                                results, origin=f'fine-tuning-mlm {strategy}', experiment_type=experiment_type, text_col=mlm.text_col, model_name=mlm.model_name, experiment='cv', last_step_only=False))
                del experiment
            else:
                print(f'Model {mlm.model_name} not selected for processing.')
            
            gc.collect()
            torch.cuda.empty_cache()
        
        group_by=['model', 'field', 'dataset', 'split', 'experiment', 'origin', 'classifier']
        
        if strategy == 'alldata':
            if experiment_type == 'feature extraction':
                all_results_df = pd.concat([EmbeddingsExperiment.results_to_pandas(
                    results_all[m], experiment_type=experiment_type, origin=f'fine-tuning-mlm {strategy}', splits = ['train', 'test'], text_col=mlm.text_col, model_name=m) for m in results_all.keys() if sum([len(i) for i in results_all[m].values()]) > 0], axis=0, ignore_index=True)
            elif experiment_type == 'fine-tuning downstream':
                all_results_df = pd.concat(
                    [FineTuningSentimentAnalysisExperiment.results_to_pandas(
                        results_all[m], experiment_type=experiment_type, origin=f'fine-tuning-mlm {strategy}', splits = ['train', 'test'], text_col=mlm.text_col, model_name=m, experiment='cv', last_step_only=True) for m in results_all.keys() if sum([len(i) for i in results_all[m].values()]) > 0], axis=0, ignore_index=True)

                all_results_epochs_df = pd.concat([
                    FineTuningSentimentAnalysisExperiment.results_to_pandas(
                        results_all[m], experiment_type=experiment_type, origin=f'fine-tuning-mlm {strategy}', splits = ['train', 'test'], text_col=mlm.text_col, model_name=m, experiment='cv', last_step_only=False) for m in results_all.keys() if sum([len(i) for i in results_all[m].values()]) > 0], axis=0, ignore_index=True)
        else:
            all_results_df = pd.concat(results_all, axis=0, ignore_index=True)
            if experiment_type == 'fine-tuning downstream':
                all_results_epochs_df = pd.concat(results_all_epochs, ignore_index=True)

        if experiment_type == 'feature extraction':
            print(all_results_df)
            print(os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_dir, strategy))
            EmbeddingsExperiment.save_results(
                all_results_df,
                folder=os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_dir, strategy),
                excel_file_name=f'{experiment_type}-fine-tuning-mlm-{strategy}-{classes}.xlsx',
                detail_sheet_name=f'{folds}-Fold CV',
                group_by=group_by)
            return all_results_df
        elif experiment_type == 'fine-tuning downstream':
            FineTuningSentimentAnalysisExperiment.save_results_consolidated(
                (all_results_df, all_results_epochs_df),
                folder=os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_dir, strategy),
                excel_file_name=f'{experiment_type}-fine-tuning-mlm-{strategy}-{classes}.xlsx',
                detail_sheet_name=f'{folds}-Fold CV',
                epoch_detail_sheet_name=f'{folds}-Fold CV-{train_epochs}-epoch',
                group_by=group_by)
            return all_results_df, all_results_epochs_df
