from functools import lru_cache
from typing import Set, List

import numpy as np
import pyLDAvis
import tomotopy as tp
from nltk import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

wnl = WordNetLemmatizer()


@lru_cache(100000)
def word_lemmatize(w: str) -> str:
    return wnl.lemmatize(w, pos="n")


def normalize(txt: str, stopws: Set[str]) -> List[List[str]]:
    # cleverer ways take longer
    sentences = sent_tokenize(txt)
    tokenized_sentences = [[w for w in word_tokenize(s) if str.isalnum(w)] for s in sentences]
    filtered_sentences = [[w for w in t if w.lower() not in stopws] for t in tokenized_sentences]
    lemmatized_sentences = [[word_lemmatize(w).lower() for w in s if len(w) > 2 and str.isalpha(w)]
                            for s in filtered_sentences]
    postfiltered_sentences = [[w for w in t if w not in stopws] for t in lemmatized_sentences]
    return postfiltered_sentences


def make_pyldavis_html(mdl: tp.LDAModel, out_html: str):
    """ Генерация визуализации тематической модели  с помощью pyLDAvis """
    topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq
    vis = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency,
        sort_topics=False,
        mds="tsne"
    )
    pyLDAvis.save_html(vis, out_html)
