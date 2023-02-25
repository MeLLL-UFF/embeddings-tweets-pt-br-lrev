from nlp_ptbr.base import *
from nlp_ptbr.preprocessing import normalize_label, normalize_tweets, normalize_funcs
from nlp_ptbr.preprocessing import RE_EMOTICON, EMOJI_PATTERN, TWITTER_USER_MENTION, URL_PATTERN, EMAIL_PATTERN, TWITTER_HASHTAG_PATTERN

import os, glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from datasets import DatasetDict, Dataset, ClassLabel
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold, LeaveOneOut


class SentimentDatasets():
    paths = {
        'binary': {'path': os.path.join('sentiment-analysis', 'binary-classification')},
        'multiclass': {'path': os.path.join('sentiment-analysis', 'multiclass-classification')},
        'mix': {'path': os.path.join('sentiment-analysis', 'multi-binary-classification')}
    }
    
    def __init__(self, datasets_path=None, normalize = False, normalize_funcs=None, remove_duplicates=False, remove_long_sentences=False, long_sentences=64, classes='binary', label_col = 'class'):
        if datasets_path is None:
            self.datasets_path = os.path.join(Experiment.UFF_SENTIMENT_DATASETS, SentimentDatasets.paths[classes]['path'])
        else:
            self.datasets_path = datasets_path
        self.datasets_files = glob.glob(os.path.join(self.datasets_path, '*.csv'), recursive=True)
        self.sorted_datasets_files = [f[0] for f in sorted([(f, len(pd.read_csv(f, encoding='utf-8', on_bad_lines='skip', lineterminator='\n'))) for f in self.datasets_files], key=lambda x: x[1], reverse=False)]
        
        n_classes = [len(pd.read_csv(f, encoding='utf-8', on_bad_lines='skip', lineterminator='\n')[label_col].value_counts(dropna=False)) for f in self.datasets_files]
        if len(set(n_classes)) == 1:
            self.n_classes = n_classes[0]
        else:
            # Indicates the set of datasets has different numbers of lables
            self.n_classes = -1
        
        self.sorted_datasets_names = [get_base_name(f) for f in self.sorted_datasets_files]
        self.normalize = normalize
        self.normalize_funcs = normalize_funcs
        self.remove_duplicates=remove_duplicates
        self.remove_long_sentences=remove_long_sentences
        self.long_sentences=long_sentences
        self.suffix='-final'
        self.classes=classes

    @staticmethod
    def fold_data_generator(X, y, cv, only_these_folds=None):
        for i, (train_index, test_index) in enumerate(cv.split(X=X, y=y)):
            #print(f'Informado subconjunto de folds para uso: {only_these_folds}')
            if only_these_folds is not None:
                if i not in only_these_folds:
                    continue
            train_text = X.iloc[train_index].to_list()
            train_label = y.iloc[train_index].to_list()
            val_text = X.iloc[test_index].to_list()
            val_label = y.iloc[test_index].to_list()
            
            class_labels = list(set(train_label))
            num_class = len(class_labels)
            
            encoder = LabelEncoder()
            encoder.fit(np.array(class_labels))
            train_y = encoder.transform(train_label)
            val_y = encoder.transform(val_label)
            
            yield i, (train_text, train_y, val_text, val_y, encoder, train_index, test_index)
    
    @staticmethod
    def concat_datasets(idx, datasets, tweet_col = 'tweet', label_col = 'class', normalize=False, remove_duplicates=False, remove_long_sentences=False, long_sentences=64, normalize_funcs=None, classes='binary'):
        return pd.concat([SentimentDatasets.read_and_normalize(datasets[i], tweet_col = tweet_col, label_col = label_col, normalize=normalize, remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, normalize_funcs=normalize_funcs, classes=classes) for i in idx]).reset_index(drop=True)
    
    @staticmethod
    def read_and_normalize(csv, tweet_col = 'tweet', label_col = 'class', normalize=False, remove_duplicates=False, remove_long_sentences=False, long_sentences=64, normalize_funcs=None, classes='binary'):
        df = pd.read_csv(csv, encoding='utf-8', on_bad_lines='skip', lineterminator='\n')
        
        classes='binary' if len(df[label_col].value_counts(dropna=False)) == 2 else 'multiclass'
        
        df = normalize_label(df, label_col = label_col, classes=classes)
        target_text_col = tweet_col
        
        if normalize:
            target_text_col = f'{tweet_col}_normalized'
            df[target_text_col] = normalize_tweets(df, tweet_col = tweet_col, funcs=normalize_funcs)
        
        if remove_duplicates:
            df = df.drop_duplicates(subset=[target_text_col, label_col], ignore_index=True)
        if remove_long_sentences:
            df = df[(df[target_text_col].str.split().apply(len) <= long_sentences)].reset_index(drop=True)
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def get_df_stats(df, name, quantiles = [.25, .5, .75], text_col='tweet', label_col = 'class'):
        counts = df[label_col].value_counts(dropna=False)
        counts.name = 'absolute'
        counts.rename(lambda x: str(x), inplace=True)
        percents = df[label_col].value_counts(dropna=False, normalize=True)
        percents.name = 'percent'
        percents.rename(lambda x: str(x) + ' (%)', inplace=True)
        quartile = df[text_col].str.split().apply(len).quantile(quantiles)
        quartile.name = 'quartile'
        quartile.rename(dict(zip(quantiles, [f'Q{i+1}' for i in range(len(quantiles))])), inplace=True)
        
        stats = pd.concat([counts, percents, quartile]).reset_index().T
        stats = stats.rename(columns=stats.iloc[0]).drop(stats.index[0])
        stats['min'] = df[text_col].str.split().apply(len).min()
        stats['mean'] = df[text_col].str.split().apply(len).mean()
        stats['max'] = df[text_col].str.split().apply(len).max()
        stats.insert(0, 'rows', len(df))
        stats['duplicated'] = df.duplicated([text_col, 'class']).sum()
        stats['duplicated (%)'] = stats['duplicated']/stats['rows']
        stats['< 10'] = (df[text_col].str.split().apply(len) < 10).sum()
        stats['< 10 (%)'] = stats['< 10']/stats['rows']
        stats['> 64'] = (df[text_col].str.split().apply(len) > 64).sum()
        stats['> 64 (%)'] = stats['> 64']/stats['rows']
        stats['emoticon'] = (df[text_col].str.count(RE_EMOTICON) != 0).sum()
        stats['emoticon (%)'] = stats['emoticon']/stats['rows']
        stats['emoji'] = (df[text_col].str.count(EMOJI_PATTERN) != 0).sum()
        stats['emoji (%)'] = stats['emoji']/stats['rows']
        stats['user_mention'] = (df[text_col].str.count(TWITTER_USER_MENTION) != 0).sum()
        stats['user_mention (%)'] = stats['user_mention']/stats['rows']
        stats['url'] = (df[text_col].str.count(URL_PATTERN) != 0).sum()
        stats['url (%)'] = stats['url']/stats['rows']
        stats['email'] = (df[text_col].str.count(EMAIL_PATTERN) != 0).sum()
        stats['email (%)'] = stats['email']/stats['rows']
        stats['hashtag'] = (df[text_col].str.count(TWITTER_HASHTAG_PATTERN) != 0).sum()
        stats['hashtag (%)'] = stats['hashtag']/stats['rows']
        stats.index = [name]
        
        #stats.loc[:, ['positivo (%)', 'negativo (%)']] = stats.loc[:, ['positivo (%)', 'negativo (%)']].astype('float') * 100
        stats.loc[:, stats.columns.str.contains(pat = '(%)', regex=False)] = stats.loc[:, stats.columns.str.contains(pat = '(%)', regex=False)].astype('float') * 100
        
        stats = stats.round(2)
        stats.loc[:, counts.index.values.tolist() + ['Q1', 'Q2', 'Q3']] = stats.loc[:, counts.index.values.tolist() + ['Q1', 'Q2', 'Q3']].astype('int64')      

        texts_dict = {
            'text_min_length': list(set(df[text_col][df[text_col].str.split().apply(len) == df[text_col].str.split().apply(len).min()].to_list())),
            'text_max_length': list(set(df[text_col][df[text_col].str.split().apply(len) == df[text_col].str.split().apply(len).max()].to_list()))
        }

        stats = pd.concat([stats, pd.DataFrame.from_records([texts_dict], index=[name])], axis=1)
        
        return stats
    
    
    @staticmethod
    def get_df_token_stats(df, name, tokenizer, trainer, pretrained_word_embedding, num_tokens):
        stats = {}
        tokenizer.train_from_iterator(iterator=df, trainer=trainer)
        embedding_matrix, hits, misses, missed_tokens = pretrained_word_embedding.generate_embedding_matrix(vocab_size=num_tokens, word_index=tokenizer.get_vocab())

        stats['vocab_size'] = [len(tokenizer.get_vocab())]
        stats['hits'] = [hits]
        stats['misses'] = [misses]
        stats['hits (%)'] = [hits/len(tokenizer.get_vocab())]
        stats['misses (%)'] = [misses/len(tokenizer.get_vocab())]
        return pd.DataFrame(index=[name], data=stats)

    def get_datasets_stats(self, index=[], text_col='tweet', label_col = 'class', split=False):
        dfs = {'binary': [], 'multiclass': []}
        for f, df in self.get_datasets_by_index(index=index):
            classes='binary' if len(df[label_col].value_counts(dropna=False)) == 2 else 'multiclass'
            dfs[classes].append(SentimentDatasets.get_df_stats(df, name=f, text_col=text_col))
        
        if split:
            return pd.concat(dfs['binary'], axis=0), pd.concat(dfs['multiclass'], axis=0)
        else:
            return pd.concat(dfs['binary'] + dfs['multiclass'], axis=0)
    
    @staticmethod
    def raw_for_cv_generator(dataset_file_or_dataframe, cv, text_col='tweet', label='class', remove_duplicates=False, remove_long_sentences=False, long_sentences=64, disable_tqdm=False, only_these_folds=None, classes='binary'):
        if isinstance(dataset_file_or_dataframe, pd.DataFrame):
            df = dataset_file_or_dataframe
        else:
            df = SentimentDatasets.read_and_normalize(dataset_file_or_dataframe, remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, classes=classes)

        n_folds = cv.n_splits if only_these_folds is None else len(only_these_folds)
        
        for fold, (train_text, train_y, val_text, val_y, label_encoder, train_index, test_index) in SentimentDatasets.fold_data_generator(X=df[text_col], y=df[label], cv=cv, only_these_folds=only_these_folds):
            yield fold, (train_text, train_y, val_text, val_y, label_encoder, train_index, test_index)
    
    @staticmethod
    def custom_tokenized_for_cv_generator(
        dataset_file_or_dataframe, tokenizer, trainer, sequence_length, cv,
        text_col='tweet', label='class', remove_duplicates=False, remove_long_sentences=False, long_sentences=64, disable_tqdm=False, only_these_folds=None, classes='binary'):
        
        for fold, (train_text, train_y, val_text, val_y, label_encoder, train_index, test_index) in SentimentDatasets.raw_for_cv_generator(
            dataset_file_or_dataframe=dataset_file_or_dataframe, cv=cv, text_col=text_col, label=label,
            remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, classes=classes):
            
            tokenizer_object, word_index, vocab_size = get_tokenizer(
                sequence_length=sequence_length,
                text=train_text,
                hugging_face_tokenizer=tokenizer,
                hugging_face_tokenizer_trainer=trainer)

            train_x_tokenized = tokenizer_object.encode_batch(train_text)
            val_x_tokenized = tokenizer_object.encode_batch(val_text)

            yield fold, train_x_tokenized, train_y, val_x_tokenized, val_y, tokenizer_object, label_encoder, train_index, test_index
    
    @staticmethod
    def custom_tokenized_dataloader_for_cv_generator(
        dataset_file_or_dataframe, tokenizer, trainer, sequence_length, cv,
        text_col='tweet', label='class', remove_duplicates=False, remove_long_sentences=False, long_sentences=64, disable_tqdm=False, only_these_folds=None, batch_size=64, drop_last=False, classes='binary'):
        
        for fold, train_x_tokenized, train_y, val_x_tokenized, val_y, fold_tokenizer, label_encoder, train_index, test_index in SentimentDatasets.custom_tokenized_for_cv_generator(
            dataset_file_or_dataframe=dataset_file_or_dataframe, tokenizer=tokenizer, trainer=trainer, sequence_length=sequence_length,
            cv=cv, text_col=text_col, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, classes=classes):

            train_loader = get_data_loader(train_x_tokenized, train_y, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, drop_last=drop_last, custom_tokenizer=True)
            val_loader = get_data_loader(val_x_tokenized, val_y, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, drop_last=drop_last, custom_tokenizer=True)

            yield fold, train_loader, train_y, val_loader, val_y, fold_tokenizer, label_encoder
    
    @staticmethod
    def get_tokenized_dataset(raw_datasets, tokenizer, text_col='tweet', truncation=True, max_length=None):
        tokenized_datasets = raw_datasets.map(lambda examples: tokenizer(examples[text_col], truncation=truncation, max_length=max_length), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns([text_col])
        tokenized_datasets.set_format("torch")
        tokenized_datasets["train"].column_names
        
        return tokenized_datasets
    
    @staticmethod
    def get_tokenized_dataloader(raw_datasets, tokenizer, text_col='tweet', shuffle=True, batch_size=32, drop_last=False, max_length=None):
        tokenized_datasets = SentimentDatasets.get_tokenized_dataset(raw_datasets=raw_datasets, tokenizer=tokenizer, text_col=text_col, max_length=max_length)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=shuffle, batch_size=batch_size, collate_fn=data_collator, drop_last=drop_last)
        
        if 'validation' in raw_datasets.keys():
            eval_dataloader = DataLoader(
            tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator, drop_last=drop_last)
            return train_dataloader, eval_dataloader
        
        return train_dataloader
    
    # Raw Hugging Face Dataset For CV
    @staticmethod
    def raw_dataset_for_cv_generator(dataset_file_or_dataframe, cv, text_col='tweet', label='class', label_out_name='labels',
                                     remove_duplicates=False, remove_long_sentences=False, long_sentences=64, disable_tqdm=False, only_these_folds=None, classes='binary', drop_label_column = False):
        
        if isinstance(dataset_file_or_dataframe, pd.DataFrame):
            df = dataset_file_or_dataframe
        else:
            df = SentimentDatasets.read_and_normalize(dataset_file_or_dataframe, remove_duplicates=remove_duplicates, remove_long_sentences=remove_long_sentences, long_sentences=long_sentences, classes=classes)
        
        n_folds = cv.n_splits if only_these_folds is None else len(only_these_folds)
        
        pbar_folds = tqdm(enumerate(SentimentDatasets.fold_data_generator(X=df[text_col], y=df[label], cv=cv, only_these_folds=only_these_folds)), total=n_folds, leave=False, desc="Split", disable=disable_tqdm)
        
        for i, (fold, (train_text, train_y, val_text, val_y, label_encoder, train_index, test_index)) in pbar_folds:
            pbar_folds.set_description(f'Fold {i+1} de {n_folds}')
            data_files = {}
            data_files['train'] = Dataset.from_pandas(pd.DataFrame({text_col: train_text, label_out_name: train_y}))
            data_files['validation'] = Dataset.from_pandas(pd.DataFrame({text_col: val_text, label_out_name: val_y}))
            raw_datasets=DatasetDict(data_files)
            
            if drop_label_column:
                raw_datasets=raw_datasets.remove_columns([label_out_name])
            
            yield fold, raw_datasets, label_encoder
        
        pbar_folds.close()
    

    # Tokenized Hugging Face Dataset For CV
    @staticmethod
    def tokenized_dataset_for_cv_generator(tokenizer, dataset_file_or_dataframe, cv, text_col='tweet', label='class', label_out_name='labels', disable_tqdm=False, max_length=None, only_these_folds=None, classes='binary', drop_label_column = False):
        for i, (fold, raw_datasets, label_encoder) in enumerate(SentimentDatasets.raw_dataset_for_cv_generator(dataset_file_or_dataframe=dataset_file_or_dataframe, cv=cv, text_col=text_col, label=label, label_out_name=label_out_name, disable_tqdm=disable_tqdm, only_these_folds=only_these_folds, classes=classes, drop_label_column = drop_label_column)):
            yield SentimentDatasets.get_tokenized_dataset(raw_datasets=raw_datasets, tokenizer=tokenizer, text_col=text_col, max_length=max_length), label_encoder
            
    # Tokenized Dataloaders For CV
    @staticmethod
    def dataloader_for_cv_generator(tokenizer,
                                    dataset_file_or_dataframe, cv, text_col='tweet', label='class', label_out_name='labels', batch_size=32,
                                    max_length=None, disable_tqdm=False, only_these_folds=None, drop_label_column = False, classes='binary'):
        
        for i, (fold, raw_datasets, label_encoder) in enumerate(
            SentimentDatasets.raw_dataset_for_cv_generator(dataset_file_or_dataframe=dataset_file_or_dataframe,
                                                           cv=cv, text_col=text_col, label=label, only_these_folds=only_these_folds,
                                                           label_out_name=label_out_name, disable_tqdm=disable_tqdm, classes=classes, drop_label_column = drop_label_column)):
            
            yield fold, SentimentDatasets.get_tokenized_dataloader(raw_datasets=raw_datasets, tokenizer=tokenizer, text_col=text_col, shuffle=True, batch_size=batch_size, max_length=max_length), label_encoder
    
    def load_fold_from_dataset(self, dataset_base_name, fold, text_col, n_splits=10, shuffle=True, random_state=2017):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        df=self.load_datasets_by_index(index=self.get_datasets_index_by_name(name=[dataset_base_name]))[dataset_base_name]
        data = SentimentDatasets.raw_for_cv_generator(df, cv=cv, text_col=text_col, only_these_folds=[fold], classes=self.classes)
        
        return next(data)
    
    def load_datasets_by_index(self, index=[]):
        return {f: df for f, df in self.get_datasets_by_index(index=index)}
    
    # Return a list of datasets index by names
    def get_datasets_index_by_name(self, name=[]):
        datasets = {}
        if len(name) == 0:
            files = self.sorted_datasets_names
        else:
            accessed_mapping = map(self.sorted_datasets_names.index, name)
            files = list(accessed_mapping)
        return files
    
    # Return a list of datasets by indexes
    def get_datasets_path_by_index(self, index=[]):
        datasets = {}
        if len(index) == 0:
            files = self.sorted_datasets_files
        else:
            accessed_mapping = map(self.sorted_datasets_files.__getitem__, index)
            files = list(accessed_mapping)
        return files
     
    # Return a list of datasets by indexes
    def get_datasets_by_index(self, index=[]):
        files = self.get_datasets_path_by_index(index=index)
        
        for f in files:
            df = SentimentDatasets.read_and_normalize(
                csv=f, 
                remove_duplicates=self.remove_duplicates,
                remove_long_sentences=self.remove_long_sentences,
                long_sentences=self.long_sentences,
                normalize=self.normalize,
                normalize_funcs=self.normalize_funcs,
                classes=self.classes)
            yield get_base_name(f), df
            
    
    # Concatenate all datasets given the path files normalizing text and label
    def get_concatenated_datasets(self, datasets_files=None, label='class'):
        if datasets_files is None:
            n_datasets = len(self.sorted_datasets_files)
            datasets = self.sorted_datasets_files
        else:
            n_datasets = len(datasets_files)
            datasets = datasets_files
        
        train = SentimentDatasets.concat_datasets(range(n_datasets), datasets, normalize=self.normalize, normalize_funcs=self.normalize_funcs, remove_duplicates=self.remove_duplicates, remove_long_sentences=self.remove_long_sentences, classes=self.classes)
        
        class_labels = list(set(train[label].to_list()))
        num_class = len(class_labels)

        encoder = LabelEncoder()
        encoder.fit(np.array(class_labels))
        train.loc[:, label] = encoder.transform(train[label])
        
        #print(encoder.classes_)
        #print(list(encoder.inverse_transform([0, 1, 2])))
        
        return train, encoder
    
    # Generator for leave-one-out validation strategy at the level of dataset
    def dataset_loo_generator(self, datasets_files=None, label='class'):
        if datasets_files is None:
            n_datasets = len(self.sorted_datasets_files)
            datasets = self.sorted_datasets_files
        else:
            n_datasets = len(datasets_files)
            datasets = datasets_files
            
        loo = LeaveOneOut()
        
        for i, (train_index, test_index) in enumerate(loo.split(range(n_datasets))):
            train_datasets = [os.path.basename(datasets[j]).replace('-final.csv', '') for j in train_index]
            test_datasets = [os.path.basename(datasets[k]).replace('-final.csv', '') for k in test_index]
            train_ds = SentimentDatasets.concat_datasets(train_index, datasets, normalize=self.normalize, normalize_funcs=self.normalize_funcs, remove_duplicates=self.remove_duplicates, remove_long_sentences=self.remove_long_sentences, classes=self.classes)
            test_ds = SentimentDatasets.read_and_normalize(datasets[test_index[0]], normalize=self.normalize, normalize_funcs=self.normalize_funcs, remove_duplicates=self.remove_duplicates, remove_long_sentences=self.remove_long_sentences, classes=self.classes)
            
            class_labels = list(set(train_ds[label].to_list()))
            num_class = len(class_labels)

            print(class_labels)
            
            encoder = LabelEncoder()
            encoder.fit(np.array(class_labels))
            train_ds.loc[:, label] = encoder.transform(train_ds[label])
            test_ds.loc[:, label] = encoder.transform(test_ds[label])
            
            yield i, train_ds, test_ds, train_datasets, test_datasets, encoder


    # Raw Hugging Face Dataset For Training Language Model - Only One Training Set is returned
    @staticmethod
    def get_raw_hf_train_dataset(train_df=None, val_df=None, text_col='tweet', label='class', label_out_name='labels', label_names=None, drop_label_column=False, label_encoder=None):            
        # A training dataset in the form of pandas dataframe has already been provided. We only need to make this a Hugging Face Dataset.
        # train_df and val_df before encoding label
        train = train_df.copy()
        class_labels = list(set(train[label].to_list()))
        num_class = len(class_labels)
        
        if label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(np.array(class_labels))
        
        train.loc[:, label] = label_encoder.transform(train[label])
        if val_df is not None:
            val = val_df.copy()
            val.loc[:, label] = label_encoder.transform(val[label])
        
        data_files = {}
        data_files['train'] = Dataset.from_pandas(pd.DataFrame({text_col: train[text_col], label_out_name: train[label]}))
        if val_df is not None:
            data_files['validation'] = Dataset.from_pandas(pd.DataFrame({text_col: val[text_col], label_out_name: val[label]}))
        raw_datasets=DatasetDict(data_files)

        if label_names is None:
            label_names = label_names=[c.upper() for c in label_encoder.classes_]

        new_features = raw_datasets["train"].features.copy()
        new_features[label_out_name] = ClassLabel(names=label_names)    
        raw_datasets = raw_datasets.cast(new_features)
        
        if drop_label_column:
            raw_datasets=raw_datasets.remove_columns([label_out_name])

        return raw_datasets, label_encoder
    
    # Raw Hugging Face Dataset For Training Language Model - Only One Training Set is returned
    def raw_hf_train_dataset(self, datasets_files=None, train_df=None, val_df=None, text_col='tweet', label='class', label_out_name='labels', label_names=None, drop_label_column=False, label_encoder=None):            
        # A training dataset in the form of pandas dataframe has already been provided. We only need to make this a Hugging Face Dataset.
        # train_df and val_df before encoding label
        if train_df is not None:
            train = train_df.copy()
            class_labels = list(set(train[label].to_list()))
            num_class = len(class_labels)
            
            if label_encoder is None:
                label_encoder = LabelEncoder()
                label_encoder.fit(np.array(class_labels))
            
            train.loc[:, label] = label_encoder.transform(train[label])
            if val_df is not None:
                val_df.loc[:, label] = label_encoder.transform(val_df[label])
        else:
            train, le = self.get_concatenated_datasets(datasets_files=datasets_files)
            if label_encoder is None:
                label_encoder = le
        
        data_files = {}
        data_files['train'] = Dataset.from_pandas(pd.DataFrame({text_col: train[text_col], label_out_name: train[label]}))
        if val_df is not None:
            data_files['validation'] = Dataset.from_pandas(pd.DataFrame({text_col: val_df[text_col], label_out_name: val_df[label]}))
        raw_datasets=DatasetDict(data_files)

        if label_names is None:
            label_names = label_names=[c.upper() for c in label_encoder.classes_]

        new_features = raw_datasets["train"].features.copy()
        new_features[label_out_name] = ClassLabel(names=label_names)    
        raw_datasets = raw_datasets.cast(new_features)

        if drop_label_column:
            raw_datasets=raw_datasets.remove_columns([label_out_name])

        return raw_datasets, label_encoder
    
    
    # Raw Hugging Face Dataset For Leave One Out Validation Strategy
    def raw_datasets_hf_generator(self, datasets_files=None, text_col='tweet', label='class', label_out_name='labels', drop_label_column=False):
        if datasets_files is None:
            n_datasets = len(self.sorted_datasets_files)
            datasets = self.sorted_datasets_files
        else:
            n_datasets = len(datasets_files)
            datasets = datasets_files
        
        for i, train, test, train_datasets, test_datasets, label_encoder in self.dataset_loo_generator(datasets):
            data_files = {}
            data_files['train'] = Dataset.from_pandas(pd.DataFrame({text_col: train[text_col], label_out_name: train[label]}))
            data_files['validation'] = Dataset.from_pandas(pd.DataFrame({text_col: test[text_col], label_out_name: test[label]}))
            # name of the label col must be named 'label'
            raw_datasets=DatasetDict(data_files)
            
            new_features = raw_datasets["train"].features.copy()
            new_features[label_out_name] = ClassLabel(names=[c.upper() for c in label_encoder.classes_])
            raw_datasets = raw_datasets.cast(new_features)
            
            if drop_label_column:
                raw_datasets=raw_datasets.remove_columns([label_out_name])
            
            yield i, raw_datasets, train_datasets, test_datasets, label_encoder
            
    # Dataframes for all-data     
    def get_df_for_alldata(self, target_dataset, target_fold, text_col='tweet_normalized', label='class'):
        list_of_index = list(range(len(self.sorted_datasets_names)))
        list_of_index.remove(self.get_datasets_index_by_name([target_dataset])[0])

        # load all datasets concatenated except the target dataset
        train, label_encoder = self.get_concatenated_datasets(self.get_datasets_path_by_index(list_of_index))
        # get the target fold of the target dataset
        fold, (train_text, train_y, val_text, val_y, label_encoder, train_index, test_index) = self.load_fold_from_dataset(dataset_base_name=target_dataset, fold=target_fold, text_col=text_col)
        # concatenate the "not targets" datasets with trainset of the target fold of target dataset
        train = pd.concat([train.loc[:, [text_col, label]], pd.DataFrame({text_col: train_text, label: train_y})], axis=0, ignore_index=True)
        val = pd.DataFrame({text_col: val_text, label: val_y})

        return train, val, label_encoder