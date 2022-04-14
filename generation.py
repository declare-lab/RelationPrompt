from typing import Dict, List, Optional, Tuple

import torch
from fire import Fire
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from encoding import ExtractEncoder
from utils import DynamicModel, RelationSentence, find_sublist_index


class TextGenerator(DynamicModel):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        texts: List[str],
        do_sample=True,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        multi_prompt_ids: Optional[List[List[int]]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos, bos = tok.eos_token_id, tok.bos_token_id

        if prompt is not None:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if prompt_ids is not None:
            prompt_ids = [eos, bos] + prompt_ids
            decoder_input_ids = torch.tensor([prompt_ids])
        if multi_prompt_ids is not None:
            assert len(texts) == len(multi_prompt_ids)
            multi_prompt_ids = [[eos, bos] + lst for lst in multi_prompt_ids]
            decoder_input_ids = torch.tensor(multi_prompt_ids)
        if decoder_input_ids is not None:
            kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))

        outputs = self.model.generate(
            **self.tokenize(texts),
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return,
            return_dict_in_generate=True,
            output_scores=save_scores,
            max_length=self.max_length,
            **kwargs,
        )

        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        return self.decode(outputs.sequences)

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.bos_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts


class LabelConstraint:
    def __init__(
        self,
        labels: List[str],
        tokenizer: PreTrainedTokenizerFast,
        prefix: str = " Relation :",
    ):
        self.prefix: List[int] = tokenizer(prefix, add_special_tokens=False).input_ids
        self.label_map: Dict[int, str] = {
            tokenizer(" " + x, add_special_tokens=False).input_ids[0]: x for x in labels
        }
        self.tokenizer = tokenizer

    def run(self, triplet: RelationSentence, scores: Tensor) -> RelationSentence:
        triplet = triplet.copy(deep=True)
        assert scores.ndim == 2
        token_ids = scores.argmax(dim=-1).int().tolist()
        i = find_sublist_index(token_ids, self.prefix)
        if i == -1:
            return triplet

        position = i + len(self.prefix)
        best = ""
        best_score = -1e9
        for j, label in self.label_map.items():
            score = scores[position, j].item()
            if score > best_score:
                best = label
                best_score = score

        if triplet.label in self.label_map.values():
            assert best == triplet.label

        assert len(best) > 0
        triplet.label = best
        triplet.score = best_score
        return triplet


class TripletSearchDecoder(DynamicModel):
    gen: TextGenerator
    constraint: LabelConstraint
    encoder: ExtractEncoder
    top_k: int = 4

    def generate(self, text: str, **kwargs) -> Tuple[str, Tensor]:
        outputs = self.gen.run(
            [text],
            do_sample=False,
            num_return=1,
            num_beams=1,
            save_scores=True,
            **kwargs,
        )

        assert len(outputs) == 1
        assert self.gen.scores is not None
        scores = torch.log_softmax(self.gen.scores[0], dim=-1)
        assert scores.ndim == 2
        return outputs[0], scores

    def find_prefix_end(self, token_ids: List[str], prefix: str) -> int:
        prefix_ids = self.gen.tokenizer(prefix, add_special_tokens=False).input_ids
        i = find_sublist_index(token_ids, prefix_ids)
        position = i + len(prefix_ids)
        return position

    def branch(
        self, text: str, prefix: str, prompt: Optional[str] = None, **kwargs
    ) -> List[Tuple[str, float]]:
        _, scores = self.generate(text, prompt=prompt, **kwargs)
        token_ids = scores.argmax(dim=-1).int().tolist()
        i = self.find_prefix_end(token_ids, prefix)

        pairs = []
        for j in torch.argsort(scores[i])[-self.top_k :]:
            p = (prompt or "") + self.gen.decode([token_ids[:i] + [j]])[0]
            pairs.append((p, scores[i, j].item()))

        return pairs

    def run(self, text: str) -> List[RelationSentence]:
        x = self.encoder.encode_x(text)
        outputs = []

        for prompt_a, score_a in self.branch(x, prefix="Head Entity :"):
            for prompt_b, score_b in self.branch(
                x, prefix=" Tail Entity :", prompt=prompt_a
            ):
                output, scores = self.generate(x, prompt=prompt_b)
                token_ids = token_ids = scores.argmax(dim=-1).int().tolist()
                i = self.find_prefix_end(token_ids, prefix=" Relation :")
                score_c = max(scores[i].tolist())
                s = self.encoder.safe_decode(x=x, y=output)
                s = self.constraint.run(s, scores)
                # score_c = s.score  # From LabelConstraint
                s.score = (score_a + score_b + score_c) / 3
                outputs.append(s)

        return outputs


if __name__ == "__main__":
    Fire()
