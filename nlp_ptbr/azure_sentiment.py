import os
import numpy as np
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

from datasets import load_metric
from nlp_ptbr.base import Experiment
from sklearn.preprocessing import LabelEncoder

endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]

text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def get_df_batches(dataframe, batch_size=10):
    for _, batch in dataframe.groupby(pd.RangeIndex(len(dataframe)) // batch_size):
        yield batch


def to_azure_format(batch, text_col = "tweet"):
    return batch.loc[:, [text_col]].reset_index().rename(columns={"index": "id", text_col: "text"}).to_dict('records')


def get_sentiment(text_analytics_client, documents):
    result = text_analytics_client.analyze_sentiment(documents, show_opinion_mining=False, language="pt-br")
    results_dict = [{'id': doc.id, 'text': documents[idx]['text'], 'sentiment': doc.sentiment, 'positive': doc.confidence_scores['positive'], 'neutral': doc.confidence_scores['neutral'], 'negative': doc.confidence_scores['negative'], 'error': doc.is_error} for idx, doc in enumerate(result) if not doc.is_error]
    return results_dict


def get_sentiment_dataset(df, text_col = "tweet"):
    results_list = []

    for batch in get_df_batches(df, batch_size=10):
        documents = to_azure_format(batch, text_col=text_col)
        results_list.append(pd.DataFrame.from_records(get_sentiment(text_analytics_client, documents)))
    
    return pd.concat(results_list).astype({'id':int})


def max_column(row, columns=['positive', 'neutral', 'negative']):
    """
    Retorna o nome da coluna que tem o maior valor entre as colunas especificadas.
    :param row: Linha do DataFrame do Pandas.
    :param columns: Lista de nomes de colunas a serem verificadas.
    :return: Nome da coluna com o maior valor.
    """
    subset = row[columns]
    if isinstance(subset, pd.DataFrame):
        numeric_subset = subset.select_dtypes(include='number')
    else:
        numeric_subset = subset.astype(float)
    if numeric_subset.empty:
        raise ValueError("Nenhuma coluna numérica encontrada")
    max_col = numeric_subset.idxmax()
    return max_col

def translate_sentiment(sentiment):
    if sentiment == 'positive':
        return 'positivo'
    elif sentiment == 'negative':
        return 'negativo'
    elif sentiment == 'neutral':
        return 'neutro'

def decode_mixed(x, classes=2):
    if x.sentiment == 'mixed':
        if classes == 2:
            return translate_sentiment(max_column(x, columns=['positive', 'negative']))
        else:
            return translate_sentiment(max_column(x, columns=['positive', 'neutral', 'negative']))
    else:
        if classes == 2:
            return translate_sentiment(max_column(x, columns=['positive', 'negative']))
        else:
            return translate_sentiment(x.sentiment)


def compute_metrics(df, target='class', predictions_col='azure'):
    encoder = LabelEncoder()
    encoder.fit(np.array(list(set(df[target]))))
    references = encoder.transform(df[target])
    predictions = encoder.transform(df[predictions_col])
    
    metrics = load_metric('nlp_ptbr/metrics.py')
    metrics.add_batch(predictions=predictions, references=references)
    return metrics.compute(labels=np.unique(references), predictions_proba=None, labels_names=None)


def get_azure_sentiments(datasets, index=[], text_col = "tweet", results_dir='azure'):

    results_dict = {}

    results_path = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_RESULTS, results_dir)
    os.makedirs(results_path, exist_ok=True)

    for df in datasets.get_datasets_by_index(index=index):
        print(df[0])
        results_dict[df[0]] = df[1].merge(get_sentiment_dataset(df[1], text_col = text_col).set_index('id').iloc[:, 1:], left_index=True, right_index=True, how='inner')
        results_dict[df[0]]['azure'] = results_dict[df[0]].apply(lambda row: decode_mixed(row, len(results_dict[df[0]]['class'].value_counts(dropna=False))), axis=1)

    binary_metrics_dict = []
    multiclass_metrics_dict = []

    for k, df in results_dict.items():
        m = compute_metrics(results_dict[k])
        m['model'] = 'azure'
        m['field'] = text_col
        m['dataset'] = k
        if len(results_dict[k]['class'].value_counts(dropna=False)) == 2:
            binary_metrics_dict.append(m)
        else:
            multiclass_metrics_dict.append(m)

    binary = pd.DataFrame.from_records(binary_metrics_dict)

    new_columns = ['model', 'field', 'dataset'] + [col for col in binary.columns if col not in ['model', 'field', 'dataset']]
    binary = binary[new_columns]

    binary.to_excel(os.path.join(results_path, f'azure_binary_{text_col}.xlsx'), float_format='%.4f', index=False)

    multiclass = pd.DataFrame.from_records(multiclass_metrics_dict)

    new_columns = ['model', 'field', 'dataset'] + [col for col in multiclass.columns if col not in ['model', 'field', 'dataset']]
    multiclass = multiclass[new_columns]

    multiclass.to_excel(os.path.join(results_path, f'azure_multiclass_{text_col}.xlsx'), float_format='%.4f', index=False)

    for k, df in results_dict.items():
        df.to_excel(os.path.join(results_path, f'{k}_{text_col}.xlsx'), float_format='%.4f', index=False)

    return results_dict, binary, multiclass


if __name__ == '__main__':
    documents = [
        {"id": "1", "text": """notícias boas, que bom #quarentena #quartaligadajusticasdv. mais de 70 vacinas contra o coronavírus estão sendo desenvolvidas diz nature #coronavirus https://t.co/kdgzxogm2b"""},
        {"id": "2", "text": """#fiqueatento! ⚖️ plenário julga, nesta quarta-feira (15), por videoconferência, ações relacionadas às medidas de combate ao #coronavírus. https://t.co/t6veb1ehcc"""},
        {"id": "3", "text": """conselho de medicina interdita médica que sugeriu “soro anti-coronavírus” - folha impacto https://t.co/mdcmfatmg6"""},
        {"id": "4", "text": """deputada, a única opção rápida é realmente a renúncia do presidente? juridicamente nada pode ser feito? a conta pra esse sociopata só vai chegar com a avalanche de cadáveres? o brasil não merece esse lixo de pr https://t.co/7dc2lipufc"""}
        ]
    
    get_sentiment(documents)