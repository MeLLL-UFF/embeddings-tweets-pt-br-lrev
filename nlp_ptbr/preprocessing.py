import re
import nltk
nltk.download('stopwords')
from itertools import chain
from collections import Counter
import pandas as pd
import numpy as np
from nlp_ptbr.TweetNormalizer import normalizeTweet
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits, WhitespaceSplit, Split
from sklearn.feature_extraction.text import CountVectorizer

SPECIAL_TOKENS_DEFAULT = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

SPECIAL_TOKENS_KEYS = ["HTTPURL", "USER", "EMOJI", "EMAIL", "EMOTICON"]
START_DELIMITER = '['
END_DELIMITER = ']'
SPECIAL_TOKENS_VALUES = [f'{START_DELIMITER}{t}{END_DELIMITER}' for t in SPECIAL_TOKENS_KEYS]
TWITTER_SPECIAL_TOKENS_LOWER = re.compile(r'|'.join([re.escape(s.lower()) for s in SPECIAL_TOKENS_VALUES]))
SPECIAL_TOKENS_DICT = dict(zip(SPECIAL_TOKENS_KEYS, SPECIAL_TOKENS_VALUES))

LONG3_CHAR_PATTERN = r'(([a-zA-Z])\2{3,})'
LONG3_PONCTUATION_KEEP3_PATTERN = r'(([!.?])\2{3,})'
LONG3_PONCTUATION_KEEP1_PATTERN = r'(([@#])\2{2,})'
LONG_SPACE_PATTERN = r'[\s]{2,}'
LONG_SPECIAL_PATTERN = r'(([@#])\2{2,})'
TWITTER_HASHTAG_PATTERN = r'#[a-zA-Z0-9_]+'
TWITTER_USER_MENTION = r'@[a-zA-Z0-9_]+'
URL_PATTERN = r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
EMAIL_PATTERN = r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'
SPLIT_SLASH_PATTERN = r'[^/]+(?://[^/]*)*'
PONCTUATION_PATTERN = r'[․”¡!¿?.:,;\"\“\'\…\-\—]+'
ACCENTS_PATTERN = r'[´`^~’‘]+'
SYMBOLS_PATTERN = r'[๏↝ᆺυ⇡←₩˙˟₍⁾↓↑⁽·¸„@#®\t\n\<>*↷¤¯→¨€¥£$}{¬™”‘↪•\u0089\u0091\u2060\u200b\u2066\u2069]+'
MATH_PATTERN = r'[›÷º×=«»%°]+'
SPLIT_PATTERN = r'[↔&+,\/_\|\\-]+'
BETWEEN_SQUARE_BRACKETS = r'\[(.*?)\]'
BETWEEN_PARENTHESES = r'\((.*?)\)'
BETWEEN_CURLY_BRACKETS = r'\{(.*?)\}'
BRACKETS_PATTERN = r'[\(\)\[\]\{\}]+'
SUPERSCRIPT_NUMBERS_PATTERN = r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+'
FINAL_UNICODE_PATTERN = r'[^\x00-\x7f]'

HTML = ['&lt;', '&gt', '&le;', '&ge;', '&amp;']

# Emoticons and emojis
RE_HEART = r'(?:<+/?3+)+'
EMOTICONS_START = [r'>:', r':', r'=', r';']
EMOTICONS_MID = [r' ', r'´', r'-', r',', r'^', u'\'', u'\"']
EMOTICONS_END = [r'D', r'd', r'p', r'P', r'v', r')', r'o', r'O', r'(', r'3', r'/', r'|', u'\\']
EMOTICONS_EXTRA = [r'o≡o', r'¯\_(ツ)_/¯', r'^^', r'-_-', r'x_x', r'^_^', r'o.o', r'o_o', r'(:', r'):', r');', r'(;', r':´(', r': )', r': ´ (', r': ´ )', r': ¨ )', r':¨(', r'/: - (']

RE_EMOTICON = r'|'.join([re.escape(s) for s in EMOTICONS_EXTRA])
for s in EMOTICONS_START:
    for m in EMOTICONS_MID:
        for e in EMOTICONS_END:
            RE_EMOTICON += '|{0}{1}?{2}+'.format(re.escape(s), re.escape(m), re.escape(e))

EMOJI_FLAGS_PATTERN = r'[\U0001F1E0-\U0001F1FF]+'
EMOJI_SYMBOLS_PATTERN = r'[\U0001F300-\U0001F5FF]+'
EMOJI_EMOTICONS_PATTERN = r'[\U0001F600-\U0001F64F]+'
EMOJI_TRANSPORT_PATTERN = r'[\U0001F680-\U0001F6FF]+'
EMOJI_CHEMICAL_PATTERN = r'[\U0001F700-\U0001F77F]+'
EMOJI_GEOMETRIC_PATTERN = r'[\U0001F780-\U0001F7FF]+'
EMOJI_ARROWS_PATTERN = r'[\U0001F800-\U0001F8FF]+'
EMOJI_SUP_SYMBOLS_PATTERN = r'[\U0001F900-\U0001F9FF]+'
EMOJI_EXT_SYMBOLS_PATTERN = r'[\U0001FA70-\U0001FAFF]+'
EMOJI_CHESS_PATTERN = r'[\U0001FA00-\U0001FA6F]'
EMOJI_DINGBATS_PATTERN = r'[\U00002702-\U000027B0]'
MISCELLANEOUS_TECHNICAL = r'[\U00002300-\U000023FF]'
MISCELLANEOUS_SYMBOLS_PICTOGRAPHS = r'[\U0001F300-\U0001F5FF]'
EMOJI_OTHER_PATTERN = r'[\U000024C2-\U0001F251]'

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "\U00002300-\U000023FF" # Miscellaneous Technical
    "\U000024C2-\U0001F251" # Miscellaneous Symbols and Pictographs
    "]+")


STOPWORDS = nltk.corpus.stopwords.words('portuguese')

STOPWORDS_AND_OTHERS = STOPWORDS + ['', 'pra', 'q', 'tá', 'tô']
STOPWORDS_AND_OTHERS.append('...')
STOPWORDS_AND_OTHERS.append('/')
STOPWORDS_AND_OTHERS.extend(PONCTUATION_PATTERN)



def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    result = " ".join(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', hashtag_body))
    return result

def upper(text):
    return text.group()[0:].upper()

def count_pattern(df, pattern, col = 'tweet'):
    return df[col].apply(lambda x: len(re.findall(pattern, x)))

give_back_funcs = [lambda x: re.sub(k, v, x) for k, v in SPECIAL_TOKENS_DICT.items()]

normalize_funcs = [
    str.strip,
    lambda x: re.sub(URL_PATTERN, SPECIAL_TOKENS_DICT['HTTPURL'], x),
    lambda x: re.sub(EMAIL_PATTERN, SPECIAL_TOKENS_DICT['EMAIL'], x),
    lambda x: re.sub(LONG3_CHAR_PATTERN, r"\2"*3, x),
    lambda x: re.sub(LONG3_PONCTUATION_KEEP3_PATTERN, r"\2"*3, x),
    lambda x: re.sub(LONG3_PONCTUATION_KEEP1_PATTERN, r"\2", x),
    lambda x: re.sub(LONG_SPECIAL_PATTERN, r"\2", x),
    lambda x: re.sub(LONG_SPACE_PATTERN, " ", x),
    lambda x: re.sub(TWITTER_HASHTAG_PATTERN, hashtag, x),
    lambda x: re.sub(TWITTER_USER_MENTION, SPECIAL_TOKENS_DICT['USER'], x),
    lambda x: re.sub(RE_EMOTICON, SPECIAL_TOKENS_DICT['EMOTICON'], x),
    lambda x: re.sub(EMOJI_PATTERN, SPECIAL_TOKENS_DICT['EMOJI'], x),
    lambda x: re.sub(PONCTUATION_PATTERN, '', x),
    lambda x: re.sub(ACCENTS_PATTERN, '', x),
    lambda x: re.sub(SYMBOLS_PATTERN, '', x),
    lambda x: re.sub(MATH_PATTERN, '', x),
    lambda x: re.sub(SUPERSCRIPT_NUMBERS_PATTERN, '', x),
    lambda x: re.sub(SPLIT_PATTERN, ' ', x),
    str.lower,
    lambda x: re.sub(TWITTER_SPECIAL_TOKENS_LOWER, upper, x),
    lambda x: re.sub(BRACKETS_PATTERN, '', x),
    lambda x: re.sub('HTTPURL', SPECIAL_TOKENS_DICT['HTTPURL'], x),
    lambda x: re.sub('USER', SPECIAL_TOKENS_DICT['USER'], x),
    lambda x: re.sub('EMAIL', SPECIAL_TOKENS_DICT['EMAIL'], x),
    lambda x: re.sub('EMOTICON', SPECIAL_TOKENS_DICT['EMOTICON'], x),
    lambda x: re.sub('EMOJI', SPECIAL_TOKENS_DICT['EMOJI'], x),
    lambda x: re.sub(LONG_SPACE_PATTERN, " ", x)
]

NORMALIZE_BERTWEET_STRIP_SPACES = [str.lower, normalizeTweet, str.strip, lambda x: re.sub(LONG_SPACE_PATTERN, ' ', x)]

def normalize_tweets(df, tweet_col = 'tweet', funcs = []):
    normalize_tweets = df[tweet_col]
    for f in funcs:
        normalize_tweets = normalize_tweets.apply(f)
    
    return normalize_tweets


def normalize_label(df, label_col = 'class', classes='binary'):
    temp = df.copy()
    if classes == 'multiclass':
        temp.loc[temp[label_col].isin(['negativo', 'negative', -1]), label_col] = 'Negativo'
        temp.loc[temp[label_col].isin(['neutro', 'neutral', 0]), label_col] = 'Neutro'
        temp.loc[temp[label_col].isin(['positivo', 'positive', 1]), label_col] = 'Positivo'
    elif classes == 'binary':
        temp.loc[temp[label_col] == -1, label_col] = 'Negativo'
        temp.loc[temp[label_col] == 0, label_col] = 'Negativo'
        temp.loc[temp[label_col] == 1, label_col] = 'Positivo'
    
    temp[label_col] = temp[label_col].astype(str).str.lower()
    
    return temp


def get_hugging_face_tokenizer(special_tokens=None, aditional_special_tokens=None, unk_token="[UNK]", vocab_size=30000, min_frequency=0):
    if special_tokens is None:
        special_tokens=SPECIAL_TOKENS_DEFAULT
    if aditional_special_tokens is None:
        aditional_special_tokens=SPECIAL_TOKENS_VALUES
    
    tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
    trainer = WordLevelTrainer(vocab_size=vocab_size, min_frequency=min_frequency, show_progress=True, special_tokens=special_tokens)
    tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Split(token, behavior='isolated') for token in aditional_special_tokens] + [WhitespaceSplit(), Digits(individual_digits=False)])
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    
    return tokenizer, trainer


def get_counts(df, text_col='tweet'):
    counts_dict = {'words': df[text_col].str.split(' ').apply(len),
                  'avg_words_length': df[text_col].str.split(' ').apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)),
                  'emoticon': df[text_col].str.count(RE_EMOTICON),
                  'emoji': df[text_col].str.count(EMOJI_PATTERN),
                  'user_mention': df[text_col].str.count(TWITTER_USER_MENTION),
                  'url': df[text_col].str.count(URL_PATTERN),
                  'email': df[text_col].str.count(EMAIL_PATTERN),
                  'hashtag': df[text_col].str.count(TWITTER_HASHTAG_PATTERN)}
    return pd.DataFrame(counts_dict)


def get_most_frequent_words(series, top=30, stop_words=None):
    words = chain(*series.str.split(' '))
    if stop_words:
        words = [word for word in words if word.lower() not in stop_words]
    counts = Counter(words)
    temp = pd.DataFrame(counts.most_common(top))
    temp.columns = ['Word','Count']
    
    return temp.sort_values(by='Count', ascending=True)

def get_top_ngram(corpus, n=None, top=10):
    vectorizer = CountVectorizer(lowercase=False, analyzer='word', ngram_range=(n, n), stop_words=None).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top]

def get_top_ngram_df(col, n=2, top=10):
    x,y = map(list, zip(*get_top_ngram(col.values.tolist(), n=n)[:top]))
    return pd.DataFrame({'Word': x, 'Count': y}).sort_values(by='Count', ascending=True)