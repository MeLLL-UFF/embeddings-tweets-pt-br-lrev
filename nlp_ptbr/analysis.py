import pandas as pd
import numpy as np

def highlight_results(
    results_df, split=['test'], experiment=['feature extraction'], origin=['pre-trained'], field=['tweet_normalized'], classifier=['lr'],
    models = ['bertimbau-base', 'bertweet', 'bertweetbr', 'fasttext', 'mbert', 'xlmr-base'],
    metrics = ['acc', 'f1', 'precision', 'recall'], save=False):
    
    metrics_less_is_better = ['loss', 'fn', 'fp', 'time']
    
    file_name = f'{split}_{experiment}_{origin}_{field}_{classifier}.xlsx'
    
    def highlight_max(s, props=''):
        return np.where(s == np.nanmax(s.values), props, '')
    
    filtered = results_df[
        (results_df.model.isin(models) & results_df.field.isin(field) & results_df.split.isin(split) & results_df.experiment.isin(experiment) & results_df.origin.isin(origin) & results_df.classifier.isin(classifier))].reset_index(drop=True)
    
    less_is_better=filtered.groupby(by=['dataset'])[[m for m in metrics if m in metrics_less_is_better]].rank(ascending=True, pct=False, method='min').astype(int)
    more_is_better=filtered.groupby(by=['dataset'])[[m for m in metrics if m not in metrics_less_is_better]].rank(ascending=False, pct=False, method='min').astype(int)
    
    filtered_wtith_ranks = filtered.merge(right=less_is_better, how='inner', left_index=True, right_index=True, suffixes=[None, '_rank']).merge(right=more_is_better, how='inner', left_index=True, right_index=True, suffixes=[None, '_rank'])
    
    experiment_feature_extraction = filtered_wtith_ranks.pivot(index='dataset', columns='model')[metrics]
    
    rank_experiment_feature_extraction = (filtered_wtith_ranks.pivot(index='dataset', columns='model')[[f'{m}_rank' for m in metrics]] == 1).sum().rename("wins").unstack()
    rank_experiment_feature_extraction = rank_experiment_feature_extraction.rename(index={i:i.replace('_rank', '') for i in rank_experiment_feature_extraction.index.get_level_values(0).unique()})
    
    idx = pd.IndexSlice
    slices = {metric: idx[:, idx[metric]] for metric in metrics}
    
    if list(slices.keys())[0] in metrics_less_is_better:
        formatted_df = experiment_feature_extraction.style.highlight_min(props='color:red;font-weight: bold', axis=1, subset=list(slices.values())[0])
    else:
        formatted_df = experiment_feature_extraction.style.highlight_max(props='color:black;font-weight: bold', axis=1, subset=list(slices.values())[0])
    
    for i, item in enumerate(list(slices.values())[1:]):
        if list(slices.keys())[i+1] in metrics_less_is_better:
            formatted_df = formatted_df.highlight_min(props='color:red;font-weight: bold', axis=1, subset=item)
        else:
            formatted_df = formatted_df.highlight_max(props='color:black;font-weight: bold', axis=1, subset=item)
    
    if save:
        formatted_df.to_excel(os.path.join('results', file_name), engine='openpyxl', float_format='%.4f')
    
    return formatted_df, rank_experiment_feature_extraction

def rank_summary(results_df, split='test', experiment='feature extraction', origin='pre-trained', field='tweet_normalized', classifier='lr', models = ['bertimbau-base', 'bertweet', 'bertweetbr', 'fasttext', 'mbert', 'xlmr-base'], metrics = ['acc_rank', 'f1_rank', 'precision_rank', 'recall_rank'], rank=[1]):
    by = ['dataset', 'split', 'experiment', 'origin', 'field', 'classifier']
    rank_metrics = ['acc', 'f1', 'precision', 'recall']
    
    temp = results_df.merge(
        right=results_df[results_df.model.isin(models)].groupby(by=by)[rank_metrics].rank(ascending=False, pct=False, method='min').astype(int),
        how='inner', left_index=True, right_index=True, suffixes=[None, '_rank'])
    
    res = temp[temp.model.isin(models) & (temp.field == field) & (temp.split == split) & (temp.experiment == experiment) & (temp.origin == origin) & (temp.classifier == classifier)].groupby(['model'])[metrics].agg(['value_counts']).unstack(fill_value=0)
    res.columns = res.columns.droplevel(1)
    return res.iloc[:, res.columns.get_level_values(1).isin(rank)].T