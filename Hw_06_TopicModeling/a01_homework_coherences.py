import logging
from typing import List, Callable

from simple_inverted_index import TinyInvertedIndex
from simple_window_counter import WindowCounter
from math import log2

Epsilon = 1e-15


def umass(w1: str, w2: str, index: TinyInvertedIndex) -> float:
    """ Если счётчики по нулям, возвращаем float("-inf"); в знаменателе обычно меньшая df """
    w1_df, w2_df = index.df(w1), index.df(w2)
    if w1_df == 0 or w2_df == 0:
        logging.error(f"Quite suspicious counts: {w1}:{w1_df} and {w2}:{w2_df}; returning -inf.")
        return float("-inf")
    
    return log2((index.and_count(w1, w2) + 1) / min(w1_df, w2_df))


def npmi(w1: str, w2: str, wc: WindowCounter) -> float:
    """ Если счётчики по нулям, возвращаем -1.0 """

    if w1 not in wc.word2idx or w2 not in wc.word2idx:
        return -1.0

    w1_count = wc.unigram_presence[wc.word2idx[w1]]
    w2_count = wc.unigram_presence[wc.word2idx[w2]]
    w12_count = wc.get_pair_count(w1, w2)

    if w1_count == 0 or w2_count == 0 or wc.total_windows == 0 or w12_count == 0:
        return -1.0

    return - log2(w12_count / (w1_count * w2_count) * wc.total_windows) / log2(w12_count / wc.total_windows)


def topic_coherence_averaging(top_terms: List[str], scorer: Callable[[str, str], float]) -> float:
    N = len(top_terms)
    total_pairs = N * (N - 1) // 2
    mean_npmi = 0

    for i, w1 in enumerate(top_terms):
        for j in range(i + 1, len(top_terms)):
            w2 = top_terms[j]
            mean_npmi += scorer(w1, w2)

    return mean_npmi / total_pairs


def topic_npmi(top_terms: List[str], wc: WindowCounter) -> float:
    return topic_coherence_averaging(top_terms, lambda w1, w2: npmi(w1, w2, wc))


def topic_umass(top_terms: List[str], index: TinyInvertedIndex) -> float:
    return topic_coherence_averaging(top_terms, lambda w1, w2: umass(w1, w2, index))


if __name__ == "__main__":

    import time

    start = time.perf_counter()

    topic1 = ["cow", "farm", "milk", "pond", "duck", "whistle"]
    topic2 = ["crow", "club", "moose", "computer", "dose", "eager"]
    topic3 = ["love", "affair", "affection", "tenderness", "sympathy", "charm"]

    index = TinyInvertedIndex.load_json("wiki.ii.boolean.json")
    index_loaded = time.perf_counter()
    print(f"DF index loaded {index_loaded - start:.6f} seconds")

    cooc = WindowCounter.load_json("wiki.win-5.json")
    print("window cooccurrence loaded")
    cooc_loaded = time.perf_counter()
    print(f"Window cooccurrence loaded {cooc_loaded - index_loaded:.6f} seconds")

    for topic in [topic1, topic2, topic3]:
        print(topic)
        print("umass:", topic_umass(topic, index))
        print("npmi: ", topic_npmi(topic, cooc))
        print()

    end = time.perf_counter()
    print(f"Scores computed {end - cooc_loaded:.6f} seconds")

    """
    На моей машине:
        DF index loaded 7.422165 seconds
        Window cooccurrence loaded 46.806701 seconds
        Scores computed 0.045393 seconds
        
    По NPMI и по UMass -- последняя тема лучше всех.
    Главное, что вторая -- точно плохая по обеим меркам.
    """
