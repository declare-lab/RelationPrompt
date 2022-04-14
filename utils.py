import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from fire import Fire
from pydantic import BaseModel
from pydantic.main import Extra
from tqdm import tqdm

Span = Tuple[int, int]
BasicValue = Union[str, int, bool, float]


def train_test_split(*args, **kwargs) -> list:
    raise NotImplementedError


def find_sublist_index(items: list, query: list):
    length = len(query)
    for i in range(len(items) - length + 1):
        if items[i : i + length] == query:
            return i
    return -1


def test_find_sublist_query():
    items = [1, 6, 3, 5, 7]
    print(dict(items=items))
    for query in [[6], [7], [6, 3], [3, 5, 7], [7, 5]]:
        print(dict(query=query, i=find_sublist_index(items, query)))


def find_sublist_indices(items: list, query: list) -> List[int]:
    i = find_sublist_index(items, query)
    if i == -1:
        return []
    return list(range(i, i + len(query)))


def test_find_sublist_indices():
    items = [1, 6, 3, 5, 7]
    assert find_sublist_indices(items, [6, 3, 5]) == [1, 2, 3]
    print(dict(test_find_sublist_indices=True))


class WikiProperty(BaseModel):
    """
    # https://query.wikidata.org
    # All properties with descriptions and aliases and types

    SELECT ?p ?pType ?pLabel ?pDescription ?pAltLabel WHERE {
        ?p wikibase:propertyType ?pType .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    ORDER BY ASC(xsd:integer(STRAFTER(STR(?p), 'P')))
    """

    p: str
    pType: str
    pLabel: str
    pDescription: str
    pAltLabel: str

    @property
    def id(self) -> str:
        return self.p.split("/")[-1]

    @property
    def aliases(self) -> List[str]:
        names = [n.strip() for n in self.pAltLabel.split(",")]
        return sorted(set(names))


def load_wiki_relation_map(path: str) -> Dict[str, WikiProperty]:
    df = pd.read_csv(path)
    props = [WikiProperty(**r) for r in df.to_dict(orient="records")]
    return {p.id: p for p in props}


def load_label_to_properties(
    path: str, use_alias: bool = True
) -> Dict[str, WikiProperty]:
    relation_map = load_wiki_relation_map(path)
    mapping = {}
    for p in relation_map.values():
        if not p.pLabel in mapping.keys():
            mapping[p.pLabel] = p
    if use_alias:
        for p in relation_map.values():
            for a in p.aliases:
                if a not in mapping.keys():
                    mapping[a] = p
    return mapping


def test_load_wiki():
    relation_map = load_wiki_relation_map("data/wiki_properties.csv")
    for k, v in list(relation_map.items())[:3]:
        print(dict(k=k, v=v, aliases=v.aliases))


class DynamicModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class StrictModel(BaseModel):
    class Config:
        extra = Extra.forbid
        frozen = True
        validate_assignment = True


def compute_macro_PRF(
    predicted_idx: np.ndarray, gold_idx: np.ndarray, i=-1, empty_label=None
) -> Tuple[float, float, float]:
    # https://github.com/dinobby/ZS-BERT/blob/master/model/evaluation.py
    """
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    """
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = predicted_idx[:i] == r
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0.0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


def test_compute_prf():
    a = np.array([0, 0, 0, 0, 0])
    b = np.array([0, 0, 1, 1, 0])
    print(compute_macro_PRF(a, b))


def glob_rmtree(folder: str, pattern: str, verbose=True):
    for path in Path(folder).glob(pattern):
        shutil.rmtree(path)
        if verbose:
            print(dict(rmtree=path))


def test_glob_rmtree():
    folder = "tmp/test_glob_rmtree"
    Path(folder).mkdir(exist_ok=False, parents=True)
    glob_rmtree("tmp", "test_glob*")


def hash_text(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()


def check_overlap(a: Span, b: Span) -> bool:
    # Assumes end in (start, end) is exclusive like python slicing
    return (
        a[0] <= b[0] < a[1]
        or a[0] <= b[1] - 1 < a[1]
        or b[0] <= a[0] < b[1]
        or b[0] <= a[1] - 1 < b[1]
    )


class RelationSentence(BaseModel):
    tokens: List[str]
    head: List[int]
    tail: List[int]
    label: str
    head_id: str = ""
    tail_id: str = ""
    label_id: str = ""
    error: str = ""
    raw: str = ""
    score: float = 0.0
    zerorc_included: bool = True

    def as_tuple(self) -> Tuple[str, str, str]:
        head = " ".join([self.tokens[i] for i in self.head])
        tail = " ".join([self.tokens[i] for i in self.tail])
        return head, self.label, tail

    def as_line(self) -> str:
        return self.json() + "\n"

    def is_valid(self) -> bool:
        for x in [self.tokens, self.head, self.tail, self.label]:
            if len(x) == 0:
                return False
        for x in [self.head, self.tail]:
            if -1 in x:
                return False
        return True

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @classmethod
    def from_spans(cls, text: str, head: str, tail: str, label: str, strict=True):
        tokens = text.split()
        sent = cls(
            tokens=tokens,
            head=find_span(head, tokens),
            tail=find_span(tail, tokens),
            label=label,
        )
        if strict:
            assert sent.is_valid(), (head, label, tail, text)
        return sent

    def as_marked_text(self) -> str:
        tokens = list(self.tokens)
        for i, template in [
            (self.head[0], "[H {}"),
            (self.head[-1], "{} ]"),
            (self.tail[0], "[T {}"),
            (self.tail[-1], "{} ]"),
        ]:
            tokens[i] = template.format(tokens[i])
        return " ".join(tokens)


def align_span_to_tokens(span: str, tokens: List[str]) -> Tuple[int, int]:
    # Eg align("John R. Allen, Jr.", ['John', 'R.', 'Allen', ',', 'Jr.'])
    char_word_map = {}
    num_chars = 0
    for i, w in enumerate(tokens):
        for _ in w:
            char_word_map[num_chars] = i
            num_chars += 1
    char_word_map[num_chars] = len(tokens)

    query = span.replace(" ", "")
    text = "".join(tokens)
    assert query in text
    i = text.find(query)
    start = char_word_map[i]
    end = char_word_map[i + len(query) - 1]
    assert 0 <= start <= end
    return start, end + 1


def test_align_span(
    span: str = "John R. Allen, Jr.",
    tokens=("The", "John", "R.", "Allen", ",", "Jr.", "is", "here"),
):
    start, end = align_span_to_tokens(span, tokens)
    print(dict(start=start, end=end, span=tokens[start:end]))


def find_span(span: str, tokens: List[str]) -> List[int]:
    if span == "":
        return []
    start = find_sublist_index(tokens, span.split())
    if start >= 0:
        return [start + i for i in range(len(span.split()))]
    else:
        start, end = align_span_to_tokens(span, tokens)
        return list(range(start, end))


def test_find_span(
    span: str = "Hohenzollern",
    text: str = "Princess of Hohenzollern-Sigmaringen ( born 26 March 1949",
):
    tokens = text.split()
    indices = find_span(span, tokens)
    print(dict(test_find_span=[tokens[i] for i in indices]))


class QualifierSentence(RelationSentence):
    qualifier: str = ""
    qualifier_id: str
    value: List[int]
    value_type: str

    def as_tuple(self) -> Tuple[str, str, str, str, str]:
        head = " ".join([self.tokens[i] for i in self.head])
        tail = " ".join([self.tokens[i] for i in self.tail])
        value = " ".join([self.tokens[i] for i in self.value])
        return head, self.label, tail, self.qualifier, value


class RelationData(BaseModel):
    sents: List[RelationSentence]

    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            lines = f.readlines()
            sents = [
                RelationSentence(**json.loads(x))
                for x in tqdm(lines, desc="RelationData.load")
            ]
        return cls(sents=sents)

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            f.write("".join([s.as_line() for s in self.sents]))

    @property
    def unique_labels(self) -> List[str]:
        return sorted(set([s.label for s in self.sents]))

    def train_test_split(
        self, test_size: Union[int, float], random_seed: int, by_label: bool = False
    ):
        if by_label:
            labels_train, labels_test = train_test_split(
                self.unique_labels, test_size=test_size, random_state=random_seed
            )
            train = [s for s in self.sents if s.label in labels_train]
            test = [s for s in self.sents if s.label in labels_test]
        else:
            groups = self.to_sentence_groups()
            keys_train, keys_test = train_test_split(
                sorted(groups.keys()), test_size=test_size, random_state=random_seed
            )
            train = [s for k in keys_train for s in groups[k]]
            test = [s for k in keys_test for s in groups[k]]

        # Enforce no sentence overlap
        texts_test = set([s.text for s in test])
        train = [s for s in train if s.text not in texts_test]

        data_train = RelationData(sents=train)
        data_test = RelationData(sents=test)
        if by_label:
            assert len(data_test.unique_labels) == test_size
            assert not set(data_train.unique_labels).intersection(
                data_test.unique_labels
            )

        info = dict(
            sents_train=len(data_train.sents),
            sents_test=len(data_test.sents),
            labels_train=len(data_train.unique_labels),
            labels_test=len(data_test.unique_labels),
        )
        print(json.dumps(info, indent=2))
        return data_train, data_test

    def to_sentence_groups(self) -> Dict[str, List[RelationSentence]]:
        groups = {}
        for s in self.sents:
            groups.setdefault(s.text, []).append(s)
        return groups

    def to_label_groups(self) -> Dict[str, List[RelationSentence]]:
        groups = {}
        for s in self.sents:
            groups.setdefault(s.label, []).append(s)
        return groups

    def filter_group_sizes(self, min_size: int = 0, max_size: int = 999):
        groups = self.to_sentence_groups()
        sents = [
            s
            for k, lst in groups.items()
            for s in lst
            if min_size <= len(lst) <= max_size
        ]
        return RelationData(sents=sents)

    def filter_errors(self):
        def check_valid_span(span: List[int]) -> bool:
            start = sorted(span)[0]
            end = sorted(span)[-1] + 1
            return span == list(range(start, end))

        sents = []
        for s in self.sents:
            if s.is_valid():
                if check_valid_span(s.head) and check_valid_span(s.tail):
                    sents.append(s)

        print(dict(filter_errors_success=len(sents) / len(self.sents)))
        return RelationData(sents=sents)

    def analyze(self, header: Optional[str] = None):
        labels = self.unique_labels
        groups = self.to_sentence_groups()
        spans = []
        words = []
        for s in self.sents:
            head, label, tail = s.as_tuple()
            spans.append(head)
            spans.append(tail)
            words.extend(s.tokens)
        info = dict(
            header=header,
            sents=len(self.sents),
            labels=str([len(labels), labels]),
            unique_texts=len(groups.keys()),
            unique_spans=len(set(spans)),
            unique_words=len(set(words)),
            group_sizes=str(Counter([len(lst) for lst in groups.values()])),
        )
        print(json.dumps(info, indent=2))
        return info


def wiki_uri_to_id(uri: str) -> str:
    i = uri.split("/")[-1]
    if i[0] in "QP" and i[1:].isdigit():
        return i
    else:
        return ""


def split_common_prefix(texts: List[str]) -> Tuple[str, List[str]]:
    end = 0
    i_max = min(map(len, texts))
    for i in range(i_max):
        if len(set([t[i] for t in texts])) > 1:
            break
        end += 1

    prefix = texts[0][:end]
    texts = [t[end:] for t in texts]
    return prefix, texts


def delete_checkpoints(
    folder: str = ".", pattern="**/checkpoint*", delete: bool = True
):
    for p in Path(folder).glob(pattern):
        if (p.parent / "config.json").exists():
            print(p)
            if delete:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.is_file():
                    os.remove(p)
                else:
                    raise ValueError("Unknown Type")


class Timer(BaseModel):
    name: str
    start: float = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = round(time.time() - self.start, 3)
        print(dict(name=self.name, duration=duration))


def test_timer(interval: int = 2):
    with Timer(name="test_timer"):
        time.sleep(interval)


def sorted_glob(folder: str, pattern: str) -> List[Path]:
    # Best practice to be deterministic and avoid weird behavior
    return sorted(Path(folder).glob(pattern))


def test_sorted_glob():
    for path in sorted_glob("outputs/data/zsl/wiki", "*/test.jsonl"):
        print(path)


def mark_wiki_entity(edge):
    e1 = edge["left"]
    e2 = edge["right"]
    return e1, e2


def mark_fewrel_entity(edge):
    e1 = edge["h"][2][0]
    e2 = edge["t"][2][0]
    return e1, e2


class WikiDataset:
    def __init__(self, mode, data, pid2vec, property2idx):
        assert mode in ["train", "dev", "test"]
        self.mode = mode
        self.data = data
        self.pid2vec = pid2vec
        self.property2idx = property2idx
        self.len = len(self.data)

    def load_edges(
        self, i: int, label_ids: Optional[Set[str]] = None
    ) -> List[RelationSentence]:
        g = self.data[i]
        tokens = g["tokens"]
        sents = []
        for j in range(len(g["edgeSet"])):
            property_id = g["edgeSet"][j]["kbID"]
            edge = g["edgeSet"][j]
            head, tail = mark_wiki_entity(edge)
            if label_ids and property_id not in label_ids:
                continue
            s = RelationSentence(
                tokens=tokens, head=head, tail=tail, label="", label_id=property_id
            )
            sents.append(s)
        return sents

    def __getitem__(self, item: int) -> RelationSentence:
        # The ZS-BERT setting is throw away all except first edge
        return self.load_edges(item)[0]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    """
    python new_utils.py analyze_relation_data --path data/relations/trex/100000.jsonl
    """
    test_find_sublist_query()
    test_load_wiki()
    test_compute_prf()
    test_glob_rmtree()
    test_find_sublist_indices()
    Fire()
