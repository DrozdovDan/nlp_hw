# RUN: pytest -q a01a_test_coherences.py

import math

from simple_inverted_index import TinyInvertedIndex
from simple_window_counter import WindowCounter
from a01_homework_coherences import umass, npmi


def _build_synthetic_indices():
    """
    Corpus (3 docs):
      D1: a b c d
      D2: a b a
      D3: b c

    For UMASS (doc-level):
      df(a)=2 (D1,D2), df(b)=3 (D1,D2,D3), df(c)=2 (D1,D3), df(d)=1 (D1)
      and_count(a,b)=2 (D1,D2)
      and_count(a,c)=1 (D1)
      and_count(c,d)=1 (D1)

    For NPMI (window presence, window size K=2, stride=1):
      D1 windows: [a b], [b c], [c d]
      D2 windows: [a b], [b a]  -> both are {a,b}
      D3 windows: [b c]
      Total windows W = 6

      Unigram window counts:
        a: 3  ([a b] in D1, both windows in D2)
        b: 5  (D1: [a b],[b c]; D2: two {a,b}; D3: [b c])
        c: 3  (D1: [b c],[c d]; D3: [b c])
        d: 1  (D1: [c d])

      Pair window counts:
        (a,b): 3  (D1:1, D2:2)
        (b,c): 2  (D1:1, D3:1)
        (c,d): 1  (D1:1)
    """
    vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
    idx2 = {0: "a", 1: "b", 2: "c", 3: "d"}

    ii = TinyInvertedIndex(word2idx=vocab)
    docs_str = {
        1: "a b c d".split(),
        2: "a b a".split(),
        3: "b c".split(),
    }

    for did, toks in docs_str.items():
        ii.add_doc(did, toks)

    ii.finalize()

    docs_int = [
        [[vocab[w] for w in "a b c d".split()]],
        [[vocab[w] for w in "a b a".split()]],
        [[vocab[w] for w in "b c".split()]],
    ]
    wc = WindowCounter(word2idx=vocab)
    pair_csr, total_windows = wc.build_window_coocc_csr(
        docs=docs_int, V=len(vocab), K=2, batch_docs=10, exclude_self=True, progress=False
    )
    assert wc is not None and pair_csr is wc.pair_counts
    assert total_windows == wc.total_windows

    return ii, wc, vocab, idx2


def test_umass_values():
    ii, _, _, _ = _build_synthetic_indices()

    # (a,b): log2((2+1)/min(2,3)) = log2(1.5) ≈ 0.5849625
    val_ab = umass("a", "b", ii)
    assert math.isclose(val_ab, math.log2(1.5), rel_tol=1e-9, abs_tol=1e-12)

    # (a,c): log2((1+1)/min(2,2)) = log2(1) = 0
    val_ac = umass("a", "c", ii)
    assert math.isclose(val_ac, 0.0, abs_tol=1e-12)

    # (c,d): log2((1+1)/min(2,1)) = log2(2) = 1
    val_cd = umass("c", "d", ii)
    assert math.isclose(val_cd, 1.0, abs_tol=1e-12)


def test_npmi_counts_and_values():
    _, wc, vocab, _ = _build_synthetic_indices()

    # Check window accounting
    assert wc.total_windows == 6

    a, b, c, d = [vocab[x] for x in ("a", "b", "c", "d")]
    # unigram window counts
    assert int(wc.unigram_presence[a]) == 3
    assert int(wc.unigram_presence[b]) == 5
    assert int(wc.unigram_presence[c]) == 3
    assert int(wc.unigram_presence[d]) == 1

    # pair window counts (symmetry)
    assert int(wc.pair_counts[a, b]) == 3
    assert int(wc.pair_counts[b, a]) == 3
    assert int(wc.pair_counts[b, c]) == 2
    assert int(wc.pair_counts[c, b]) == 2
    assert int(wc.pair_counts[c, d]) == 1
    assert int(wc.pair_counts[d, c]) == 1

    # Expected NPMI values (base-invariant):
    # (a,b): pij=3/6=0.5, pi=3/6, pj=5/6 -> NPMI ≈ 0.262
    # (b,c): pij=2/6=1/3, pi=5/6, pj=3/6 -> NPMI ≈ -0.203
    # (c,d): pij=1/6,   pi=3/6, pj=1/6   -> NPMI ≈ 0.3869

    nab = npmi("a", "b", wc)
    nbc = npmi("b", "c", wc)
    ncd = npmi("c", "d", wc)

    assert math.isclose(nab, 0.262364, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(nbc, -0.203000, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(ncd, 0.386852, rel_tol=1e-3, abs_tol=1e-3)


def test_npmi_edge_defined():
    _, wc, _, _ = _build_synthetic_indices()
    # terms that never co-occur in a size-2 window: (a,c) -> returns finite value
    # ensure it doesn't raise and is <= 0
    val = npmi("a", "c", wc)
    assert isinstance(val, float)
    assert val <= 0.0

if __name__ == "__main__":
    test_umass_values()
    test_npmi_counts_and_values()
    test_npmi_edge_defined()