import argparse
import csv
import datetime
import json
import logging
import math
import os
from typing import List

import numpy as np
import optuna
import tomotopy as tp
from nltk.corpus import stopwords
from optuna.samplers import TPESampler
from tqdm import tqdm

from a00a_utils import normalize, make_pyldavis_html
from a01_homework_coherences import topic_umass, topic_npmi
from simple_inverted_index import TinyInvertedIndex
from simple_window_counter import WindowCounter

run_start = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"debug_{run_start}.log"),
        logging.StreamHandler()
    ]
)

Stopwords = set(stopwords.words("english"))

# трудновато обучаться на очень большом
# количестве документов для подбора параметров
Limit = 10000


def read_texts(csv_path: str, text_col: str = "text", limit: int = Limit) -> List[List[str]]:
    docs, i = [], 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in tqdm(r, "reading target texts"):
            if i >= limit:
                break
            tokens = [w for s in normalize(row[text_col], Stopwords) for w in s]
            if tokens:
                docs.append(tokens)
            i += 1
    logging.debug("Texts read.")
    return docs


def perplexity(model: tp.LDAModel) -> float:
    # так быстрее, чем model.perplexity
    llpw = model.ll_per_word
    return math.exp(-llpw) if llpw is not None else float("inf")


def diversity(model: tp.LDAModel, topn: int = 5) -> float:
    topics_top = []
    for tid in range(model.k):
        topics_top.extend([w for w, _ in model.get_topic_words(tid, topn)])
    return len(set(topics_top)) / len(topics_top)


def topic_terms(model: tp.LDAModel, topn: int) -> List[List[str]]:
    return [[w for w, _ in model.get_topic_words(k, top_n=topn)] for k in range(model.k)]


def objective(trial: optuna.Trial, docs_tok: List[List[str]], seed: int) -> float:
    """ Цель, к которой мы стремимся, подгоняя гиперпараметры """

    k = trial.suggest_int("k", 5, 10)
    iters = trial.suggest_int("iterations", 500, 750)
    alpha = trial.suggest_float("alpha", 0.01, 0.75, log=True)
    eta = trial.suggest_float("eta", 0.001, 0.005, log=True)
    min_cf = trial.suggest_int("min_cf", 40, 50)
    rm_top = trial.suggest_int("rm_top", 0, 10)

    model = tp.LDAModel(k=k, alpha=alpha, eta=eta, seed=seed,
                        min_cf=min_cf, rm_top=rm_top)

    for doc in docs_tok:
        model.add_doc(doc)

    # ну а вдруг
    if len(model.docs) == 0:
        return float("inf")

    model.train(iters, workers=1)
    ppl = perplexity(model)
    div = diversity(model)
    return ppl, div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv") # файл для обучения
    ap.add_argument("--ii", default="wiki.ii.boolean.json")
    ap.add_argument("--wc", default="wiki.win-5.json")
    ap.add_argument("--out", default="runs_lda")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=423)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # читаем и токенизируем
    docs_tok = read_texts(args.train)
    logging.debug(f"A total of {len(docs_tok)} read and prepared.")

    # поиск по перплексии
    study = optuna.create_study(directions=["minimize", "maximize"],
                                study_name="lda_tomotopy",
                                sampler=TPESampler(n_startup_trials= max(20, args.trials // 5),
                                                   multivariate=True, group=True,
                                                   n_ei_candidates=64,
                                                   constant_liar=True,
                                                   seed=args.seed))

    logging.info(f"Created study {study}")
    study.optimize(lambda tr: objective(tr, docs_tok, args.seed),
                   n_trials=args.trials,
                   n_jobs=-1)

    logging.info("Done optimizing the hyperparams. Loading counts from Wiki... (may take a few minutes)")
    best_trials = sorted(study.best_trials, key=lambda t: (t.values[0], -t.values[1]))[:3]

    # грузим подготовленные структуры, занимает время
    inv = TinyInvertedIndex.load_json(args.ii)
    wc = WindowCounter.load_json(args.wc)
    logging.info("Counts based on Wiki loaded.")

    results = []

    for rank, tr in enumerate(best_trials, start=1):
        p = tr.params

        # корпус под выбранные фильтры
        model = tp.LDAModel(k=p["k"],
                            alpha=p["alpha"], eta=p["eta"],
                            min_cf=p["min_cf"], rm_top=p["rm_top"],
                            seed=args.seed)

        logging.debug("Filling the corpus again...")

        for doc in docs_tok:
            model.add_doc(doc)

        logging.info(f"Retraining model #{rank} (from Pareto frontier pool)...")
        model.train(p["iterations"], workers=1)
        ppl = perplexity(model)
        div = diversity(model)

        # сохраняем модель
        model_path = os.path.join(args.out, f"{run_start}_model_rank{rank}_k{p['k']}_ppl{ppl:.2f}_div{div:.2f}.bin")
        model.save(model_path, full=True)
        logging.info(f"Saved to {model_path}. Generating pyLDAvis report...")

        # PyLDAVis (ручная сборка)
        vis_path = os.path.join(args.out, f"{run_start}_pyldavis_rank{rank}_k{p['k']}.html")
        make_pyldavis_html(model, vis_path)

        # темы
        topics = topic_terms(model, args.topn)
        topics_str = "\n> ".join([" ".join(topic) for topic in topics])
        logging.info("\n> " + topics_str)

        # когерентности через наши функции
        c_um = float(np.mean([topic_umass(t, inv) for t in topics])) if topics else 0.0
        c_npm = float(np.mean([topic_npmi(t, wc) for t in topics])) if topics else 0.0

        meta = {
            "rank": rank,
            "params": p,
            "perplexity": ppl,
            "diversity": div,
            "model_path": model_path,
            "pyldavis_html": vis_path,
            "coherence": {
                "c_umass": c_um,
                "c_npmi": c_npm
            }
        }

        results.append(meta)
        logging.debug("Saving summary of this model.")

        with open(os.path.join(args.out, f"{run_start}_rank{rank}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, f"{run_start}_summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for r in results:
        str_res = f"[rank {r['rank']}] ppl={r['perplexity']:.2f} div={r['diversity']:.2f}  k={r['params']['k']}" + \
                  f"  c_umass={r['coherence']['c_umass']:.4f}  c_npmi={r['coherence']['c_npmi']:.4f}"
        logging.info(str_res)


if __name__ == "__main__":
    main()
