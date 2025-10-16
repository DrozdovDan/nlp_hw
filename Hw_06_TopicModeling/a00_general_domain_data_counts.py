""" Подготовка вспомогательных структур данных по файлу с предобработанными вики-статьями  """
# можно просто запустить и пойти пить кофе

import json
from collections import Counter

from nltk.corpus import stopwords
from tqdm import tqdm

from simple_inverted_index import TinyInvertedIndex
from simple_window_counter import WindowCounter


def iter_docs(path="wiki_100k_normalized-old.jsonl",
              operation="reading wiki, collecting tokens",
              limit=100000):
    with open(path, "r", encoding="utf-8") as rf:
        for line_idx, line in tqdm(enumerate(rf), desc=operation):
            if line_idx > limit:
                break
            data = json.loads(line.strip())
            yield [[w for w in s.split() if str.isalnum(w) and w not in stops]
                   for s in data["text"].strip().split("\n")]


if __name__ == "__main__":

    K, stops = 5, set(stopwords.words("english"))
    term_counts = Counter()

    # обычно за глаза хватает 100к, но 25к -- маловато
    # однако очень не хочется возиться с огромными объектами в памяти
    cutaway = 25000

    all_docs = [d for d in iter_docs(limit=cutaway)]
    print("Docs prepared")

    for doc in tqdm(all_docs, "tf counts collection"):
        all_terms = [w for s in doc for w in s]
        term_counts.update(all_terms)
    print("Base tf computed")

    vocabulary = sorted(list(term_counts.keys()))

    with open("wiki.tf.json", "w", encoding="utf-8") as wf:
        wf.write(json.dumps(dict(term_counts)))
    del term_counts
    print("TFs saved, rebuilding str -> int ids...")

    word2idx = {w: i for i, w in enumerate(vocabulary)}
    all_term_ids = [[[word2idx[w] for w in s] for s in d] for d in all_docs]
    del all_docs
    print("Docs converted to term-ids")

    inv_index = TinyInvertedIndex(word2idx=word2idx)

    for doc_id, doc in tqdm(enumerate(all_term_ids), "doc co-occurrence stats"):
        for s in doc:
            inv_index.add_doc_int(doc_id, s)

    inv_index.finalize()
    print("Inverted index for DFs built")

    print(inv_index.and_count("mother", "king"))
    print(inv_index.and_count("win", "draw"))

    print("Saving inverted index...")
    inv_index.save_json("wiki.ii.boolean.json", include_vocab=True)
    del inv_index
    print("Inverted index for DFs saved.")

    wc = WindowCounter(word2idx)
    wc.build_window_coocc_csr(all_term_ids, V=len(word2idx), K=K, batch_docs=1000)
    print("Window-co-occurrence computed, saving...")
    wc.save_json(f"wiki.win-{K}.json")
    print("Window-co-occurrence saved.")
