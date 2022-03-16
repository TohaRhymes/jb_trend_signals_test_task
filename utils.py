import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from operator import itemgetter

from bs4 import BeautifulSoup


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud

from matplotlib.pyplot import figure


def unflatten(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


def tag_to_list(tags: str) -> list:
    try:
        return [
            word.replace(">", "").replace("<", "").lower() for word in tags.split("><")
        ]
    except AttributeError:
        return []


def title_to_list(tags: str) -> list:
    try:
        return [word.lower() for word in tags.split(" ")]
    except AttributeError:
        return []


def body_to_list(text: str) -> list:
    try:
        htmlParse = BeautifulSoup(text, "html.parser")
        paragraphs = [
            para.get_text().replace("\n", " ") for para in htmlParse.find_all("p")
        ]
        enums = [
            para.get_text().replace("\n", " ") for para in htmlParse.find_all("li")
        ]
        return [word.lower() for word in " ".join(paragraphs + enums).split(" ")]
    except (AttributeError, TypeError):
        return []


def extract_by_function(series: pd.Series, fun):
    tag_list = list(series.apply(fun))
    return tag_list, unflatten(list(tag_list))


def extract_phrases(series: pd.Series, fun, k: int = 2):
    BAD_WORDS = ['=', '{', '},', '}', '/', '//', '...', '..', '.',',', '});', '<script', '<input', '/>', '-', '<', ':','[', ']','(', ')', '--', '{', '}','/*', '*/', '<a', '"', '<div', '</div>', ' ', '  ', '   ', '=>', '<button', '0;', 'var', '</td', '', 'background-color:', 'false;', '0', '1', '30', '31', 'math.pi', '--info', '--debug', 'O', '2']
    stop_words = set(stopwords.words("english"))
    porter = PorterStemmer()
    word_list = list(series.apply(fun))
    phrases = []
    for sub in word_list:
        word_sublist = [porter.stem(w) for w in sub if not w in stop_words]
        word_sublist = [w for w in word_sublist if not w in BAD_WORDS]
        for i in range(len(word_sublist) - k + 1):
            phrases.append(" ".join(word_sublist[i : i + k]))
    return phrases


def draw_popular(tags, n=20):
    vocabulary = set(tags)
    print(f"Different tags: {len(vocabulary)}")
    frequency_dist = nltk.FreqDist(tags)
    top = dict(
        sorted(dict(frequency_dist).items(), key=itemgetter(1), reverse=True)[:n]
    )
    plt.figure(figsize=(15, 10))
    plt.bar(np.arange(len(top)), list(top.values()), color=sns.color_palette("pastel"))
    plt.xticks(np.arange(len(top)), list(top.keys()), rotation=90, fontsize=16)
    plt.title(f"Top-{n} tags:", fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    return frequency_dist


def draw_cloud(frequency_dist):
    wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    figure(figsize=(12, 7), dpi=1500)
    plt.show()