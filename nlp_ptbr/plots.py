from plotly.subplots import make_subplots
import plotly.graph_objects as go
from nlp_ptbr.preprocessing import get_most_frequent_words

def get_class_histogram (df, text_col='tweet', label='class', nbinsx=100, cols=True, top=10, title='Number of Words Distribution'):
    if cols:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Positive Tweets", "Negative Tweets"))
        fig.append_trace(go.Histogram(name='Positive', x=df.loc[df['class']=='positivo'][text_col].str.split(' ').apply(len), nbinsx=nbinsx), 1, 1)
        fig.append_trace(go.Histogram(name='Negative', x=df.loc[df['class']=='negativo'][text_col].str.split(' ').apply(len), nbinsx=nbinsx), 1, 2)
        height=top*40
    else:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Positive Tweets", "Negative Tweets"))
        fig.append_trace(go.Histogram(name='Positive', x=df.loc[df['class']=='positivo'][text_col].str.split(' ').apply(len), nbinsx=nbinsx), 1, 1)
        fig.append_trace(go.Histogram(name='Negative', x=df.loc[df['class']=='negativo'][text_col].str.split(' ').apply(len), nbinsx=nbinsx), 2, 1)
        height=top*20
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    
    fig.update_layout(
            title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            showlegend=False, autosize=True, height=top*50,
            margin={"l": 50, "r": 50, "b": 50, "t": 140})
    return fig

def get_word_freq_plot(df, col='tweet_normalized', top=20, stopwords=[], title='Word Frequencies', cols=True):
    negative = get_most_frequent_words(df[df['class']=='negativo'][col], top=top, stop_words=stopwords)
    positive = get_most_frequent_words(df[df['class']=='positivo'][col], top=top, stop_words=stopwords)
    
    return get_freq_plot(positive, negative, top=top, title=title)

def get_freq_plot(positive, negative, top=20, title='Word Frequencies', cols=True):
    if cols:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Positive Tweets", "Negative Tweets"))
        fig.append_trace(go.Bar(x=positive['Count'], y=positive['Word'],orientation='h', text=positive['Count'], textposition='auto'), 1, 1)
        fig.append_trace(go.Bar(x=negative['Count'], y=negative['Word'],orientation='h', text=negative['Count'], textposition='auto'), 1, 2)
        height=top*40
    else:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Positive Tweets", "Negative Tweets"))
        fig.append_trace(go.Bar(x=positive['Count'], y=positive['Word'],orientation='h', text=positive['Count'], textposition='auto'), 1, 1)
        fig.append_trace(go.Bar(x=negative['Count'], y=negative['Word'],orientation='h', text=negative['Count'], textposition='auto'), 2, 1)
        height=top*20
    
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    
    fig.update_layout(
            title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            showlegend=False, autosize=True, height=top*50,
            margin={"l": 50, "r": 50, "b": 50, "t": 140})
    return fig

def get_results_plot(df, count_col='tweet', label='class', name=''):
    df_result = df.groupby(label)[count_col].count()
    fig = go.Figure(data=[go.Pie(labels=df_result.index, values=df_result.values, textposition='outside',
                   name=name, textinfo='label+percent',
                   insidetextorientation='radial', hole=.5)])
    return fig