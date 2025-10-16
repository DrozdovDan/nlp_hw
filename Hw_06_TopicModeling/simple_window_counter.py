""" Счётчик совместных соупоминаний, и чем он быстрее, тем лучше """

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm


class WindowCounter:

    def __init__(self, word2idx: Dict[str, int]):
        self.pair_counts: Optional[csr_matrix] = None
        self.word2idx: Dict[str, int] = word2idx
        self.unigram_presence: Optional[np.ndarray] = None
        self.total_windows: int = 0

    @staticmethod
    def _flush_chunk(
            rows_buf: List[np.ndarray],
            cols_buf: List[np.ndarray],
            vocab_size: int,
            accumulator: Optional[csr_matrix],
            exclude_self: bool = True,
    ) -> Optional[csr_matrix]:
        """
        Собираем накопленные пары (r,c) из буферов в один COO-чанк (data=1),
        превращает в CSR, симметризует (A <- A + A^T), обнуляет диагональ (по желанию),
        и добавляет к аккумулятору acc (CSR). Возвращает обновлённый acc.
        """
        if not rows_buf:
            return accumulator

        r, c = np.concatenate(rows_buf), np.concatenate(cols_buf)
        data = np.ones_like(r, dtype=np.int32)

        chunk = coo_matrix((data, (r, c)), shape=(vocab_size, vocab_size)).tocsr()
        chunk = chunk + chunk.T

        if exclude_self:
            chunk.setdiag(0)
            chunk.eliminate_zeros()

        if accumulator is None:
            return chunk
        return accumulator + chunk

    def build_window_coocc_csr(
            self,
            docs: Iterable[List[List[int]]],
            V: int,
            K: int = 5,
            batch_docs: int = 2000,
            exclude_self: bool = True,
            progress: bool = True,
    ) -> Tuple[csr_matrix, int]:
        """
        Главная функция построения co-occurrence по окнам (для NPMI).

        Вход:
          - docs: итератор по документам, где каждый документ — список предложений,
                  каждое предложение — список ID токенов.
          - V: размер словаря (максимальный id + 1).
          - K: размер окна. Если длина предложения ≤ K — берётся одно окно = всё предложение;
               иначе скользящее окно фиксированной длины K (stride = 1).
          - batch_docs: каждые N документов сбрасываем накопленные пары в матрицу (экономия памяти).
          - exclude_self: исключать (x, x) пары.
          - progress: показывать ли прогресс-бар.

        Подсчёт:
          - В каждом окне уникализируем токены (presence внутри окна).
          - Для каждого окна учитываем все уникальные пары слов (каждая пара максимум 1 раз на окно).
          - Накапливаем:
              * self.pair_counts — CSR[V,V] с количеством окон, где пара (i,j) встречалась совместно.
              * self.unigram_presence — массив длины V с количеством окон, где слово i встречалось.
              * self.total_windows — общее число обработанных окон.

        Выход:
          - win_coocc_csr: CSR-матрица размера VxV с целочисленными presence-счётчиками пар по окнам.
          - total_windows: общее число окон (для нормировки в NPMI).
        """
        rows_buf: List[np.ndarray] = []
        cols_buf: List[np.ndarray] = []
        coo_acc_csr: Optional[csr_matrix] = None
        self.unigram_presence = np.zeros(V, dtype=np.int64)
        total_windows = 0

        doc_counter = 0
        iterator = (
            tqdm(docs, desc="building window coocc", unit="doc") if progress else docs
        )

        for doc in iterator:
            for sent in doc:
                if not sent:
                    continue
                idx = np.asarray(sent, dtype=np.int64)
                n = idx.size
                if n == 0:
                    continue

                if n <= K:
                    win = np.unique(idx)
                    if win.size:
                        total_windows += 1
                        self.unigram_presence[win] += 1
                        if win.size >= 2:
                            iu, ju = np.triu_indices(win.size, k=1)
                            rows_buf.append(win[iu])
                            cols_buf.append(win[ju])
                else:
                    # фиксированной длины окно [start : start+K)
                    for start in range(0, n - K + 1):
                        w = np.unique(idx[start: start + K])
                        if w.size == 0:
                            continue
                        total_windows += 1
                        self.unigram_presence[w] += 1
                        if w.size >= 2:
                            iu, ju = np.triu_indices(w.size, k=1)
                            rows_buf.append(w[iu])
                            cols_buf.append(w[ju])

            doc_counter += 1

            if doc_counter % batch_docs == 0:
                coo_acc_csr = self._flush_chunk(
                    rows_buf,
                    cols_buf,
                    vocab_size=V,
                    accumulator=coo_acc_csr,
                    exclude_self=exclude_self,
                )
                rows_buf.clear()
                cols_buf.clear()

        # Сбросить хвост
        coo_acc_csr = self._flush_chunk(
            rows_buf,
            cols_buf,
            vocab_size=V,
            accumulator=coo_acc_csr,
            exclude_self=exclude_self,
        )
        rows_buf.clear()
        cols_buf.clear()

        if coo_acc_csr is None:
            coo_acc_csr = csr_matrix((V, V), dtype=np.int32)

        self.pair_counts = coo_acc_csr
        self.total_windows = int(total_windows)
        return self.pair_counts, self.total_windows

    def get_pair_count(self, w1: str | int, w2: str | int) -> int:
        """
        Возвращает количество соупоминаний пары слов (w1, w2)
        из построенной матрицы self.pair_counts.

        Аргументы:
          - w1, w2: либо строки (требуется self.word2idx), либо числовые индексы.

        Возвращает:
          - Целое число (кол-во наблюдений).
        """
        if self.pair_counts is None:
            raise ValueError("Matrix not built yet. Run build_window_coocc_csr first.")
        # TODO: ЗАДАНИЕ
        raise NotImplementedError

    def save_json(self, path: str) -> None:
        """
        Сохраняет матрицу со-упоминаний (CSR) и словарь word2idx в JSON.
        Формат: компактное хранение CSR (data/indices/indptr/shape/dtype).
        """
        if self.pair_counts is None:
            raise ValueError("Nothing to save: pair_counts is None.")

        csr = self.pair_counts.tocsr()  # ensure CSR
        payload = {
            "format": "window_presence_v1",
            "csr": {
                "data": csr.data.tolist(),
                "indices": csr.indices.tolist(),
                "indptr": csr.indptr.tolist(),
                "shape": list(csr.shape),
                "dtype": str(csr.dtype),
            },
            "unigram_presence": None if self.unigram_presence is None else self.unigram_presence.tolist(),
            "total_windows": int(self.total_windows),
            "word2idx": self.word2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "WindowCounter":
        """
        Загружает объект из JSON (матрица CSR + word2idx).
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("format") not in ("window_presence_v1", "window_coocc_v1"):
            raise ValueError("Unsupported format in JSON.")

        w2i = payload["word2idx"]
        obj = cls(word2idx=w2i)

        csr_info = payload["csr"]
        data = np.array(csr_info["data"], dtype=np.dtype(csr_info["dtype"]))
        indices = np.array(csr_info["indices"], dtype=np.int32)
        indptr = np.array(csr_info["indptr"], dtype=np.int32)
        shape = tuple(csr_info["shape"])

        obj.pair_counts = csr_matrix((data, indices, indptr), shape=shape)
        obj.unigram_presence = None
        if "unigram_presence" in payload and payload["unigram_presence"] is not None:
            obj.unigram_presence = np.array(payload["unigram_presence"], dtype=np.int64)
        obj.total_windows = int(payload.get("total_windows", 0))
        return obj


def top_pairs(
        sp_csr: csr_matrix, idx2word: Optional[Sequence[str]] = None, k: int = 20
) -> List[Tuple[int, Tuple[str, str]]]:
    coo = sp_csr.tocoo()

    if coo.nnz == 0:
        return []

    order = np.argsort(-coo.data)[:k]
    out = []

    for t in order:
        out.append(
            (int(coo.data[t]), (idx2word[int(coo.row[t])], idx2word[int(coo.col[t])]))
        )

    return out


if __name__ == "__main__":
    word2idx = {
        "cat": 0,
        "sat": 1,
        "on": 2,
        "the": 3,
        "mat": 4,
        "dog": 5,
        "and": 6,
        "became": 7,
        "friends": 8,
        "good": 9,
    }
    idx2word = [None] * len(word2idx)
    for w, i in word2idx.items():
        idx2word[i] = w
    vocabulary_size = len(word2idx)

    demo_docs = [
        [[word2idx[w] for w in "good cat sat on the mat".split()]],
        [[word2idx[w] for w in "dog and cat became good friends".split()]],
    ]

    wc = WindowCounter(word2idx=word2idx)

    win_csr, total_windows = wc.build_window_coocc_csr(
        docs=demo_docs,
        V=vocabulary_size,
        K=3,
        batch_docs=20,
        exclude_self=True,
        progress=False,
    )

    print("Shape:", win_csr.shape, "nnz:", win_csr.nnz)
    print("total_windows:", total_windows)
    print("cat–dog:", wc.get_pair_count("cat", "dog"))

    wc.save_json("win_presence.json")
    wc2 = WindowCounter.load_json("win_presence.json")
    print("Roundtrip cat–dog:", wc2.get_pair_count("cat", "dog"))

    for cnt, (wi, wj) in top_pairs(win_csr, idx2word, k=100):
        print(f"{wi:>8s} — {wj:<8s}: {cnt}")
