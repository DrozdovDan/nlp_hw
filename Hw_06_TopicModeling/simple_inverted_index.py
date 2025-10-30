import json
from array import array
from collections import defaultdict
from typing import Dict, Iterable, List, Optional


class TinyInvertedIndex:

    def __init__(self, word2idx: Dict[str, int]):
        """Неявно предполагаем, что словарь достаточно полный и что айди документов -- int-ы"""

        self.word2idx = word2idx

        # размер словаря
        self.V = max(word2idx.values(), default=-1) + 1
        # собственно инв. индекс в процессе построения: терм -> список айди документов
        self._building: Dict[int, List[int]] = defaultdict(list)
        # после finalize: список array('I') или None
        self._postings: Optional[List[Optional[array]]] = None
        # самый большой индекс документа на данный момент
        self._N = 0

    def add_doc_int(self, doc_id: int, terms: Iterable[int]):
        """ Добавляем документ `doc_id` в списки термов по айдишникам термов `terms` """

        seen_tids = set()

        # закидываем айди документа в списки для каждого токена
        for tid in terms:
            if tid is None or tid in seen_tids:
                continue
            seen_tids.add(tid)
            self._building[tid].append(doc_id)

        # обновляем наибольший айдишник, если нужно
        if doc_id > self._N:
            self._N = doc_id

    def add_doc_str(self, doc_id: int, terms: Iterable[str]) -> None:
        """Тут не мешок слов! А факт вхождения терма в документ"""
        toks = (str(t).lower() for t in terms)
        tids = [self.word2idx[tok] for tok in toks]
        self.add_doc_int(doc_id, tids)

    def add_doc(self, doc_id: int, terms: Iterable[str]) -> None:
        self.add_doc_str(doc_id, terms)

    def finalize(self):
        """
        Приводим индекс в порядок:
        - сортируем списки (постинги)
        - убираем на всякий случай дубликаты
        - приводим списки к компактному виду array('I')
        - словарь (термы -> списки) из мапы переводим в список (None если пусто)
        """
        postings: List[Optional[array]] = [None] * self.V

        for tid, docs in self._building.items():
            if not docs:
                continue
            docs.sort()

            w = 1
            for r in range(1, len(docs)):
                if docs[r] != docs[w - 1]:
                    docs[w] = docs[r]
                    w += 1
            docs = docs[:w]

            # упаковываем в array('I')
            postings[tid] = array("I", docs)

        self._postings = postings
        self._building.clear()

    @property
    def num_docs(self) -> int:
        return self._N

    def df_tid(self, tid: int) -> int:
        """В скольких документах встречается терм"""
        return len(self._postings[tid])

    def df(self, term: str) -> int:
        """В скольких документах встречается терм"""
        tid = self.word2idx.get(term.lower(), -1)
        if tid < 0 or self._postings is None:
            return 0
        return self.df_tid(tid)

    def and_count(self, w1: str, w2: str) -> int:
        """|docs(w1) ∩ docs(w2)| слиянием списков"""
        if self._postings is None:
            return 0
        t1 = self.word2idx.get(w1.lower(), -1)
        t2 = self.word2idx.get(w2.lower(), -1)

        if t1 < 0 or t2 < 0:
            return 0

        p1 = self._postings[t1]
        p2 = self._postings[t2]

        if p1 is None or p2 is None:
            return 0

        # начинать стоит с малого
        if len(p1) > len(p2):
            p1, p2 = p2, p1

        return len(set(p1) & set(p2))

    def and_docs(self, *terms: str) -> List[int]:
        """Все документы, где встречаются термы"""
        if self._postings is None:
            return []

        tids = [self.word2idx.get(t.lower(), -1) for t in terms]

        if any(t < 0 for t in tids):
            return []

        # собираем все списки и сортируем по количеству вхождений
        plist = [self._postings[t] for t in tids]
        if any(p is None for p in plist):
            return []

        paired = sorted(((len(p), p) for p in plist), key=lambda x: x[0])
        cur = paired[0][1]
        out = array("I")

        for _, nxt in paired[1:]:
            out = array("I")
            i = j = 0
            while i < len(cur) and j < len(nxt):
                a, b = cur[i], nxt[j]
                if a == b:
                    out.append(a)
                    i += 1
                    j += 1
                elif a < b:
                    i += 1
                else:
                    j += 1
            if not out:
                return []
            cur = out
        return list(cur)

    def save_json(self, path: str, include_vocab: bool = False):
        """Сохраняем сразу в JSON-файл. Формат интуитивно понятен, полагаю."""
        if self._postings is None:
            self.finalize()

        dump = {
            "num_docs": self._N,
            "vocab_size": self.V,
            "postings": {
                str(tid): list(arr)
                for tid, arr in enumerate(self._postings)
                if arr is not None
            },
        }
        if include_vocab:
            dump["word2idx"] = self.word2idx

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False)

    @classmethod
    def load_json(
        cls, path: str, word2idx: Optional[Dict[str, int]] = None
    ) -> "TinyInvertedIndex":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        file_vocab = payload.get("word2idx")
        if file_vocab is None and word2idx is None:
            raise ValueError(
                "JSON has no 'word2idx'; please pass word2idx to load_json()."
            )

        w2i = file_vocab if file_vocab is not None else word2idx
        obj = cls(word2idx=w2i)

        obj._N = int(payload.get("num_docs", 0))
        obj.V = int(payload.get("vocab_size", max(w2i.values(), default=-1) + 1))

        # rebuild postings list of length V, using compact array('I') where present
        postings_map = payload.get("postings", {})
        postings: List[Optional[array]] = [None] * obj.V
        for tid_str, docs in postings_map.items():
            tid = int(tid_str)
            if 0 <= tid < obj.V:
                postings[tid] = array("I", docs)

        obj._postings = postings
        obj._building.clear()
        return obj


if __name__ == "__main__":

    vocab = {
        "cat": 0,
        "dog": 1,
        "and": 2,
        "fish": 3,
        "milk": 4,
        "sat": 5,
        "on": 6,
        "the": 7,
        "mat": 8,
        "became": 9,
        "friends": 10,
        "likes": 11,
        "birds": 12,
        "sing": 13,
        "barks": 14,
        "owl": 15,
        "mint": 16,
        "gingerbread": 17,
        "crime": 18,
        "big": 19,
    }
    idx = TinyInvertedIndex(word2idx=vocab)
    docs = {
        1: "cat sat on the mat".split(),
        2: "dog and cat became enemies".split(),
        3: "birds sing and dog barks".split(),
        4: "big cat likes crime and mint gingerbread".split(),
    }
    for did, text in docs.items():
        idx.add_doc(did, text)

    # ВАЖНО НЕ ЗАБЫВАТЬ ВЫЗЫВАТЬ .finalize()!
    idx.finalize()
    print("df(cat) =", idx.df("cat"))  # 3
    print("AND count(cat,dog) =", idx.and_count("cat", "dog"))  # 1
    print("AND docs(cat,dog,and) =", idx.and_docs("cat", "dog", "and"))  # [2]
    idx.save_json("tiny_inverted_index.json", include_vocab=True)
    print("saved to tiny_inverted_index.json")
