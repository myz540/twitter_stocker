import tweepy
import datetime
import pytz
import nltk
from nltk import word_tokenize
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas_datareader as pdr
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class CorpusHandler(object):

    def __init__(self, api, company_df):
        self.api = api
        self.company_df = company_df

        # the corpus objects are dictionaries, with the stock company ticker as the key, and the 100 tweets as the
        # value. The 100 tweets will be stored as a list of processed tweets, with each token being separated by a white
        # space
        self.gainer_corpus = None
        self.loser_corpus = None

    @staticmethod
    def get_corpus(status):
        if isinstance(status, tweepy.models.Status):
            return status.text
        else:
            raise TypeError("Input not of type tweepy.api.Status")

    @staticmethod
    def convert_corpus_to_list(corpus):
        return word_tokenize(corpus)

    @staticmethod
    def convert_list_to_corpus(text_list):
        return ' '.join(text_list)

    def collect_tweets(self, query, limit=1000, dt=datetime.datetime.now(), tz='US/Eastern'):

        assert (isinstance(query, str))

        local_tz = pytz.timezone(tz)
        local_dt = local_tz.localize(dt)

        valid_results = []
        for s in tweepy.Cursor(self.api.search, q=query, rpp=10, count=100, lang='en').items(limit):
            if local_tz.localize(s.created_at) < local_dt:
                valid_results.append(s)

        if len(valid_results) < 100:
            print("WARN: Less than 100 results, you should be using expanded_search()")

        return valid_results[:100]

    def expanded_search(self, ticker, df):
        alt_terms = df[df.loc[:, 'Symbol'] == ticker].loc[:, ['Symbol', 'Name', 'Sector']].values.tolist()[0]

        valid_results = []
        for term in alt_terms:
            _sub_results = self.collect_tweets(term)
            valid_results.extend(_sub_results)

            if len(valid_results) > 100:
                return valid_results[:100]

    @staticmethod
    def make_wordcloud(word_list):
        joined_text = CorpusHandler.convert_list_to_corpus(word_list)

        wordcloud = WordCloud().generate(joined_text)
        plt.imshow(wordcloud, interpolation='bilinear')

    @staticmethod
    def get_word_counts(words):
        all_words = []
        for word in words:
            all_words.extend([w for w in word.strip().split()])

        print(len(all_words))
        word_counter = Counter(all_words)

        df = pd.DataFrame(
            {'word': [w for w in word_counter.keys()],
             'frequency': [count for count in word_counter.values()]}
        )

        print(df.head())

        return df

    @staticmethod
    def analyze_sentiment(word_list):
        sentiment_analyzer = SentimentIntensityAnalyzer()

        sentiment_analyzer.polarity_scores(CorpusHandler.convert_list_to_corpus(word_list))


class StockHandler(object):

    @staticmethod
    def _get_diff(df):
        assert(df.shape[0] == 1)
        return float(df['close'] - df['open']) / float(df['open'])

    @staticmethod
    def get_all_diffs(tickers, limit=50, dt=datetime.datetime.now() - datetime.timedelta(1)):
        if limit and limit < len(tickers):
            _tickers = tickers[:limit]
        else:
            _tickers = tickers

        _diffs = dict()

        for ticker in _tickers:

            try:
                _df = pdr.DataReader(ticker, 'iex', dt, dt)
                diff_value = StockHandler._get_diff(_df)
                _diffs[ticker] = diff_value
            except Exception as e:
                # if for whatever reason you cannot do the above, just skip it
                pass

        return _diffs

    @staticmethod
    def find_gainers_and_losers(diff_dict):
        _df = pd.DataFrame([diff_dict.keys(), diff_dict.values()]).T
        _df.dropna(axis=0, inplace=True)
        _df.columns = ['ticker', 'diff']
        _df.sort_values('diff', inplace=True, ascending=False)
        _df.set_index('ticker', inplace=True, drop=True)

        winners = _df.iloc[:3, :]
        losers = _df.iloc[-3:, :]

        return winners.to_dict()['diff'], losers.to_dict()['diff']


class PreProcessor:

    def __init__(self):
        self._text = None

    def process_text_tweet(self, text):
        self._text = text

        self._text = self._remove_https_tag(self._text)
        self._text = self._tokenize(self._text)
        self._text = self._lemmatize(self._text)
        self._text = self._lower(self._text)
        self._text = self._remove_symbols(self._text)
        self._text = self._remove_stopwords(self._text)
        return self._text

    def process_text_comment(self, text):
        self._text = text

        self._text = self._tokenize(self._text)
        self._text = self._lemmatize(self._text)
        self._text = self._lower(self._text)
        self._text = self._remove_symbols(self._text)
        self._text = self._remove_stopwords(self._text)
        return self._text

    @staticmethod
    def _remove_https_tag(raw):
        return re.sub('https://[\w\.\/]+', '', raw).strip()

    @staticmethod
    def _tokenize(raw):
        return nltk.word_tokenize(raw)

    @staticmethod
    def _stem(raw_tokens):
        porter = nltk.PorterStemmer()
        return [porter.stem(t) for t in raw_tokens]

    @staticmethod
    def _lemmatize(raw_tokens):
        wnl = nltk.WordNetLemmatizer()
        return [wnl.lemmatize(t) for t in raw_tokens]

    @staticmethod
    def _lower(raw_tokens):
        return [t.lower() for t in raw_tokens]

    @staticmethod
    def _remove_nonwords(raw_tokens):
        return [t for t in raw_tokens if t in nltk.corpus.words.words('en')]

    @staticmethod
    def _remove_symbols(raw_tokens):
        # bad_char = ['`','~','!','@','#','$','%','^','&','*','(',')','-','_','+','=','/','\\']
        _t = [t for t in raw_tokens if not re.match("\d+\.?\d+", t)]
        _t = [t for t in _t if t.isalpha()]
        return _t

    @staticmethod
    def _remove_punctuation_chars(raw_tokens):
        _t = [t.translate(str.maketrans('', '', string.punctuation)) for t in raw_tokens]
        return _t

    @staticmethod
    def _remove_stopwords(raw_tokens):
        return [t for t in raw_tokens if t not in nltk.corpus.stopwords.words('english')]
