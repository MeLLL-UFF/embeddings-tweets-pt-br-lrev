# https://pytorch.org/docs/stable/notes/randomness.html
GLOBAL_SEED = 2017

import random
random.seed(GLOBAL_SEED)
import numpy as np
np.random.seed(GLOBAL_SEED)
import torch
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

import datasets
from datasets.utils.logging import set_verbosity_error

import os, logging, sys
import pandas as pd
import numpy as np
from datetime import datetime
from collections import namedtuple
from sklearn.model_selection import StratifiedKFold
from itertools import chain
from collections import Counter
# Disable Hugging Face Status Bar
set_verbosity_error()
datasets.logging.set_verbosity(datasets.logging.ERROR)
datasets.logging.disable_progress_bar()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

get_base_name = lambda x: os.path.basename(x).replace('-final.csv', '')

CustomModelOutput = namedtuple('CustomModelOutput', ['logits', 'loss'])
EpochOutput = namedtuple('EpochOutput', ['score', 'time', 'loss'])
MlmOutput = namedtuple('MlmOutput', ['model_name', 'text_col', 'label', 'classes', 'dataset', 'index', 'fold', 'model_output'])
InferenceOutput = namedtuple('InferenceOutput', ['predicted_labels', 'predictions', 'predictions_proba'])
EmbeddingsOutput = namedtuple('EmbeddingsOutput', ['train_matrix', 'train_y', 'val_matrix', 'val_y', 'fold_tokenizer', 'embedding_matrix'])


def get_logger(loglevel, log_dir, name = 'sentiment-analysis'):
    os.makedirs(log_dir, exist_ok=True)
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    formater = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formater)
    logging.getLogger(name).addHandler(console)
    
    # Add file rotating handler, with level DEBUG
    #os.makedirs(os.path.join(log_dir, datetime.today().strftime("%Y%m%d"), datetime.today().strftime("%H%M%S")), exist_ok=True)
    os.makedirs(os.path.join(log_dir, datetime.today().strftime("%Y%m%d")), exist_ok=True)
    #rotatingHandler = logging.FileHandler(filename=os.path.join(log_dir, datetime.today().strftime("%Y%m%d"), datetime.today().strftime("%H%M%S"), f'{name}.log'), encoding='utf-8')
    rotatingHandler = logging.FileHandler(filename=os.path.join(log_dir, datetime.today().strftime("%Y%m%d"), f'{name}-{datetime.today().strftime("%Y%m%d")}.log'), encoding='utf-8')
    rotatingHandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s]: %(message)s', datefmt='%H:%M:%S')
    rotatingHandler.setFormatter(formatter)
    logging.getLogger(name).addHandler(rotatingHandler)

    log = logging.getLogger(name)

    return log


def set_seed(seed_value=2017):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    
# *** http://mccormickml.com/2019/07/22/BERT-fine-tuning/ 
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# https://medium.com/analytics-vidhya/bert-word-embeddings-deep-dive-32f6214f02bf
def collate_batch(batch, bert_model=None, layers_range=(9, 12), only_cls=False):
    label_list, text_list, length_list = [], [], []
    
    for (_text, _attention, _label) in batch:
        label_list.append(_label)
        text_list.append(_text.numpy())
        length_list.append((_attention.numpy()==1).sum())
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    
    output = torch.tensor(np.array(text_list), dtype=torch.int64).to(device)
    
    if bert_model is not None:
        with torch.no_grad():
            output = torch.sum(torch.stack(bert_model(output)['hidden_states'][layers_range[0]:layers_range[1]+1], dim=0), dim=0)
            if only_cls:
                output = output[:, 0, :]
    
    return output, label_list.to(device), torch.tensor(np.array(length_list))


def get_data_loader(encoding, label, batch_size, collate_fn=None, shuffle=False, drop_last=True, custom_tokenizer=False):
    
    if custom_tokenizer:
        ids = torch.from_numpy(np.array([o.ids for o in encoding]))
        att = torch.from_numpy(np.array([o.attention_mask for o in encoding]))
    else:
        ids = encoding['input_ids']
        att = encoding['attention_mask']
    
    return DataLoader(
        TensorDataset(ids, att, torch.from_numpy(label)),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last)


def get_metrics_names(n_classes=2):
    if n_classes == 2:
        return ['loss', 'acc', 'f1', 'precision','recall', 'tn', 'fp', 'fn', 'tp', 'references_0', 'references_1', 'predictions_0', 'predictions_1', 'time']
    else:
        metrics_1 = ['loss', 'acc', 'f1', 'precision','recall']
        metrics_precision_recall_f1 = [f'{i}_{c}' for c in range(n_classes) for i in ['precision', 'recall', 'f1']]
        metrics_confusion_matrix = [f'{i}_{c}' for c in range(n_classes) for i in ['tp', 'fn', 'fp', 'tn']]
        metrics_2 = [f'{i}_{c}' for c in range(n_classes) for i in ['references', 'predictions']]
        return metrics_1 + metrics_precision_recall_f1 + metrics_confusion_matrix + metrics_2 + ['time']
    
    
def corpus_tokenizer_stats(series, tokenizer, detailed=False, name=None, most_common=None, raw_series=None):
    def get_oovs(x, tokenizer):
        list_of_space_separated_pieces = x.strip().split()
        ids = [tokenizer.encode(piece, add_special_tokens=False) for piece in list_of_space_separated_pieces]
        # Identify the tokens which are unknown according to the tokenizer's `unk_token_id`
        unk_indices = [i for i, encoded in enumerate(ids) if tokenizer.unk_token_id in encoded]
        # Retrieve the strings that were converted into unknown tokens
        unknown_strings = [piece for i, piece in enumerate(list_of_space_separated_pieces) if i in unk_indices]
    
        return unknown_strings

    # Where are the missing tokens indexes: df[have_missing].tweet_normalized.apply(tokenizer.encode).apply(lambda x: np.where(np.array(x) == tokenizer.unk_token_id)[0])
    encoded = series.apply(tokenizer.encode)
    
    oov = series.apply(lambda x: get_oovs(x, tokenizer))
    unique_oov = oov.apply(lambda x: len(set(x)))
    not_unique_oov = oov.apply(lambda x: len(x))
    counter = Counter(chain(*oov.to_list()))
    mcommon = counter.most_common(most_common)
    
    if detailed:
        counts_dict = {
            'has_missing': encoded.apply(lambda x: tokenizer.unk_token_id in x),
            'unknown': encoded.apply(lambda x: x.count(tokenizer.unk_token_id)),
            'tokenized_len': encoded.apply(len)}
        return pd.DataFrame(counts_dict)
    else:
        lens = encoded.apply(len)
        counts_dict = {
            'has_missing': encoded.apply(lambda x: tokenizer.unk_token_id in x).sum(),
            'unknown': encoded.apply(lambda x: x.count(tokenizer.unk_token_id)).sum(),
            'tokenized_min': lens.min(),
            'tokenized_len': lens.mean(),
            'tokenized_std': lens.std(),
            'tokenized_max': lens.max(),
            'most_missed': mcommon,
            'unique_oovs': len(counter.most_common()),
            'unique_oov_min': unique_oov.min(),
            'unique_oov_max': unique_oov.max(),
            'unique_oov_mean': unique_oov.mean(),
            'unique_oov_std': unique_oov.std(),
            'text_min_tokenized': list(set(series[lens == lens.min()].to_list())),
            'text_max_tokenized': list(set(series[lens == lens.max()].to_list())),
            'text_oov_max_abs': list(set(series[not_unique_oov == not_unique_oov.max()].to_list())) if not_unique_oov.max() > 0 else None,
            'text_oov_max_unique': list(set(series[unique_oov == unique_oov.max()].to_list())) if unique_oov.max() > 0 else None
            }
        if raw_series is not None:
            counts_dict['raw_text_min_tokenized'] = list(set(raw_series[lens == lens.min()].to_list()))
            counts_dict['raw_text_max_tokenized'] = list(set(raw_series[lens == lens.max()].to_list()))
            counts_dict['raw_text_oov_max_abs'] = list(set(raw_series[not_unique_oov == not_unique_oov.max()].to_list())) if not_unique_oov.max() > 0 else None
            counts_dict['raw_text_oov_max_unique'] = list(set(raw_series[unique_oov == unique_oov.max()].to_list())) if unique_oov.max() > 0 else None
        stats = pd.DataFrame.from_records([counts_dict], index=[name])
        stats.index.name = 'model'
        stats.insert(loc=0, column='vocab_size', value=tokenizer.vocab_size)
        stats.insert(loc=0, column='sentence_max_length', value=tokenizer.max_model_input_sizes[list(tokenizer.max_model_input_sizes.keys())[0]])
        return stats
    

def get_sentence_vector_for_gensim(model, tokenizer, sentence, embedding_dim=300, load_method='gensim'):
    sentence_tokens = tokenizer.tokenize(sentence)
    if load_method=='gensim':
        oov = [t for t in sentence_tokens if model.has_index_for(t) == False]
        sentence_vectors = [model.get_vector(t) for t in sentence_tokens if model.has_index_for(t)]
        if len(sentence_vectors) == 0:
            return np.zeros((embedding_dim)), oov, sentence_tokens
        return np.asarray(sentence_vectors).mean(axis=0), oov, sentence_tokens
    else: #load_method=='bin', here for simplicity we assume fasttext is using tweet tokenizer too in order to extract stats of missing tokens
        oov = [t for t in sentence_tokens if model.get_word_id(t) == -1]
        return model.get_sentence_vector(sentence), oov, sentence_tokens
    
    
def corpus_tokenizer_stats_from_gensim(series, model, tokenizer, detailed=False, name=None, most_common=None, load_method='gensim'):
    encoded = pd.DataFrame(series.apply(lambda x: get_sentence_vector_for_gensim(model, tokenizer, x, load_method=load_method)).to_list(), columns =['vec', 'oov', 'tokens'])
    lens = encoded.tokens.apply(len)
    unique_oov = encoded.oov.apply(lambda x: len(set(x)))

    counter = Counter(chain(*encoded.oov.to_list()))
    mcommon = counter.most_common(most_common)
    
    counts_dict = {
        'has_missing': encoded.oov.apply(lambda x: len(x) > 0).sum(),
        'unknown': encoded.oov.apply(lambda x: len(x)).sum(),
        'tokenized_min': lens.min(),
        'tokenized_len': lens.mean(),
        'tokenized_std': lens.std(),
        'tokenized_max': lens.max(),
        'most_missed': mcommon,
        'unique_oovs': len(counter.most_common()),
        'unique_oov_min': unique_oov.min(),
        'unique_oov_max': unique_oov.max(),
        'unique_oov_mean': unique_oov.mean(),
        'unique_oov_std': unique_oov.std()}
    
    stats = pd.DataFrame.from_records([counts_dict], index=[name])
    stats.index.name = 'model'
    if load_method=='gensim':
        stats.insert(loc=0, column='vocab_size', value=len(model))
    else:
        stats.insert(loc=0, column='vocab_size', value=len(model.words))
    stats.insert(loc=0, column='sentence_max_length', value=None)
    
    return stats


class Experiment:
    HF_HOME = os.getenv('HF_HOME', default = None)
    if HF_HOME is None:
        HF_HOME = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
    
    UFF_SENTIMENT_EMBEDDINGS = os.getenv('UFF_SENTIMENT_EMBEDDINGS', default = None)
    if UFF_SENTIMENT_EMBEDDINGS is None:
        UFF_SENTIMENT_EMBEDDINGS = os.path.join(os.path.expanduser('~'), '.cache', 'embeddings')
    
    UFF_CACHE_HOME = os.getenv('UFF_CACHE_HOME', default = None)
    if UFF_CACHE_HOME is None:
        UFF_CACHE_HOME = os.path.join(os.path.expanduser('~'), '.cache')
    
    UFF_SENTIMENT_HOME = os.getenv('UFF_SENTIMENT_HOME', default = None)
    
    if UFF_SENTIMENT_HOME is None:
        UFF_SENTIMENT_HOME = os.path.join(os.path.expanduser('~'), 'uff-sentiment-analysis-ptbr')
    
    UFF_SENTIMENT_OUTPUTS = os.path.join(UFF_SENTIMENT_HOME, 'outputs')
    UFF_SENTIMENT_OUTPUTS_LOGS = os.path.join(UFF_SENTIMENT_OUTPUTS, 'logs')
    UFF_SENTIMENT_OUTPUTS_RESULTS = os.path.join(UFF_SENTIMENT_OUTPUTS, 'results')
    UFF_SENTIMENT_OUTPUTS_CLASSIFIERS = os.path.join(UFF_SENTIMENT_OUTPUTS, 'classifiers')
    
    os.makedirs(UFF_SENTIMENT_HOME, exist_ok=True)
    os.makedirs(UFF_SENTIMENT_OUTPUTS, exist_ok=True)
    os.makedirs(UFF_SENTIMENT_OUTPUTS_LOGS, exist_ok=True)
    os.makedirs(UFF_SENTIMENT_OUTPUTS_RESULTS, exist_ok=True)
    os.makedirs(UFF_SENTIMENT_OUTPUTS_CLASSIFIERS, exist_ok=True)
    
    UFF_SENTIMENT_DATASETS = os.getenv('UFF_SENTIMENT_DATASETS', default = None)
    if UFF_SENTIMENT_DATASETS is None:
        UFF_SENTIMENT_DATASETS = os.path.join(UFF_SENTIMENT_HOME, 'datasets')
    
    UFF_SENTIMENT_MODELS = os.getenv('UFF_SENTIMENT_MODELS', default = None)
    if UFF_SENTIMENT_MODELS is None:
        UFF_SENTIMENT_MODELS = os.path.join(UFF_SENTIMENT_HOME, 'models')
    
    os.makedirs(UFF_SENTIMENT_MODELS, exist_ok=True)
    
    def __init__(self, experiment_name, experiment_type, model_name, results_dir='uff', log_dir='uff', log_level='WARNING', seed = 2017, folds=10):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.log_dir = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_LOGS, log_dir)
        self.results_dir = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_dir)
        self.log_level = log_level
        self.logger = get_logger(self.log_level, self.log_dir, name=self.experiment_name)
        self.seed = seed
        self.folds = folds
        self.cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)