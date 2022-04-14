from pathlib import Path
from typing import Dict, List, Tuple

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer

from transformer_base import run_summarization
from utils import RelationData, RelationSentence


class Encoder(BaseModel):
    def encode_x(self, x: str) -> str:
        raise NotImplementedError

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        raise NotImplementedError

    def decode(self, x: str, y: str) -> RelationSentence:
        raise NotImplementedError

    def decode_x(self, x: str) -> str:
        raise NotImplementedError

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def encode_to_line(self, sent: RelationSentence) -> str:
        raise NotImplementedError

    def decode_from_line(self, line: str) -> RelationSentence:
        raise NotImplementedError

    def parse_line(self, line: str) -> Tuple[str, str]:
        raise NotImplementedError


class GenerateEncoder(Encoder):
    def encode_x(self, r: str) -> str:
        return f"Relation : {r} ."

    def decode_x(self, text: str) -> str:
        return text.split("Relation : ")[-1][:-2]

    def encode_triplet(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Context : {sent.text} Head Entity : {s} , Tail Entity : {o} ."

    def decode_triplet(self, text: str, label: str) -> RelationSentence:
        front, back = text.split(" Head Entity : ")
        _, context = front.split("Context : ")
        head, back = back.split(" , Tail Entity : ")
        tail = back[:-2]
        return RelationSentence.from_spans(context, head, tail, label)

    def encode_y(self, sent: RelationSentence) -> str:
        return self.encode_x(sent.label) + " " + self.encode_triplet(sent)

    def decode_y(self, text: str, label: str) -> RelationSentence:
        del label
        front, back = text.split(" . Context : ")
        label = self.decode_x(front + " .")
        return self.decode_triplet("Context : " + back, label)

    def decode(self, x: str, y: str) -> RelationSentence:
        r = self.decode_x(x)
        sent = self.decode_y(y, r)
        return sent

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.label)
        y = self.encode_y(sent)
        return x, y

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return y + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        return "", line.strip()


class ExtractEncoder(Encoder):
    def encode_x(self, text: str) -> str:
        return f"Context : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("Context : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Head Entity : {s} , Tail Entity : {o} , Relation : {r} ."

    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(" , Relation : ")
        label = label[:-2]
        front, tail = front.split(" , Tail Entity : ")
        _, head = front.split("Head Entity : ")
        return RelationSentence.from_spans(context, head, tail, label)

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"Head Entity : {head} , Tail Entity : {tail} , Relation :"

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return run_summarization.encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return run_summarization.decode_from_line(line)


def test_encoders(
    paths: List[str] = [
        "outputs/data/zsl/wiki/unseen_5_seed_0/train.jsonl",
        "outputs/data/zsl/fewrel/unseen_5_seed_0/train.jsonl",
    ],
    print_limit: int = 4,
    encoder_names: List[str] = ["generate", "extract"],
    limit: int = 1000,
):
    encoders = {k: select_encoder(k) for k in encoder_names}

    for p in paths:
        data = RelationData.load(Path(p))
        _, data = data.train_test_split(min(limit, len(data.sents)), random_seed=0)

        for name, e in tqdm(list(encoders.items())):
            num_fail = 0
            print(dict(name=name, p=p))
            for s in data.sents:
                encoded = e.encode_to_line(s)
                x, y = e.parse_line(encoded)
                decoded: RelationSentence = e.safe_decode(x, y)

                if decoded.as_tuple() != s.as_tuple():
                    if num_fail < print_limit:
                        print(dict(gold=s.as_tuple(), text=s.text))
                        print(dict(pred=decoded.as_tuple(), text=decoded.text))
                        print(dict(x=x, y=y, e=decoded.error))
                        print()
                    num_fail += 1

            print(dict(success_rate=1 - (num_fail / len(data.sents))))
            print("#" * 80)


def select_encoder(name: str) -> Encoder:
    mapping: Dict[str, Encoder] = dict(
        extract=ExtractEncoder(),
        generate=GenerateEncoder(),
    )
    encoder = mapping[name]
    return encoder


def test_entity_prompts(
    path: str = "outputs/data/zsl/wiki/unseen_10_seed_0/test.jsonl", limit: int = 100
):
    def tokenize(text: str, tok) -> List[str]:
        return tok.convert_ids_to_tokens(tok(text, add_special_tokens=False).input_ids)

    data = RelationData.load(Path(path))
    e = ExtractEncoder()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    print(tokenizer)
    for i, s in enumerate(tqdm(data.sents[:limit])):
        head, label, tail = s.as_tuple()
        x, y = e.encode(s)
        prompt = e.encode_entity_prompt(head, tail)
        tokens_y = tokenize(y, tokenizer)
        tokens_prompt = tokenize(prompt, tokenizer)
        assert tokens_y[: len(tokens_prompt)] == tokens_prompt
        if i < 3:
            print(tokens_y)


if __name__ == "__main__":
    Fire()
