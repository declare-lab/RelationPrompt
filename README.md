## RelationPrompt: Leveraging Prompts to Generate Synthetic Data for Zero-Shot Relation Triplet Extraction

[![PWC](https://img.shields.io/badge/PapersWithCode-Benchmark-%232cafb1)](https://paperswithcode.com/paper/relationprompt-leveraging-prompts-to-generate)
[![Colab](https://img.shields.io/badge/Colab-Code%20Demo-%23fe9f00)](https://colab.research.google.com/drive/18lrKD30kxEUolQ61o5nzUJM0rvWgpbFK?usp=sharing)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook%20Demo-important)](https://github.com/declare-lab/RelationPrompt/blob/main/demo.ipynb)

This repository implements our ACL Findings 2022 research paper [RelationPrompt: Leveraging Prompts to Generate Synthetic Data for Zero-Shot Relation Triplet Extraction](https://doi.org/10.48550/arXiv.2203.09101). 
The goal of Zero-Shot Relation Triplet Extraction (ZeroRTE) is to extract relation triplets of the format `(head entity, tail entity, relation)`, despite not having annotated data for the test relation labels.

![diagram](https://github.com/declare-lab/RelationPrompt/releases/download/v1.0.0/diagram.png)

### Installation

- Python 3.7
- If your GPU uses CUDA 11, first install the specific PyTorch: `pip install torch==1.10.0 --extra-index-url https://download.pytorch.org/whl/cu113`
- Install requirements: `pip install -r requirements.txt` or `conda env create --file environment.yml`
- Download and extract the [datasets here](https://github.com/declare-lab/RelationPrompt/releases/download/v1.0.0/zero_rte_data.zip) to `outputs/data/splits/zero_rte`
- [FewRel Pretrained Model](https://github.com/declare-lab/RelationPrompt/releases/download/v1.0.0/model_fewrel_unseen_10_seed_0.tar) (unseen=10, seed=0)
- [Wiki-ZSL Pretrained Model](https://github.com/declare-lab/RelationPrompt/releases/download/v1.0.0/model_wiki_unseen_10_seed_0.tar) (unseen=10, seed=0)


### Data Exploration | [![Colab](https://img.shields.io/badge/Colab-Code%20Demo-%23fe9f00)](https://colab.research.google.com/drive/18lrKD30kxEUolQ61o5nzUJM0rvWgpbFK#scrollTo=vw3NlKDddMIP&line=2&uniqifier=1)

```
from wrapper import Dataset

data = Dataset.load(path)
for s in data.sents:
    print(s.tokens)
    for t in s.triplets:
        print(t.head, t.tail, t.label)
```

### Generate with Pretrained Model | [![Colab](https://img.shields.io/badge/Colab-Code%20Demo-%23fe9f00)](https://colab.research.google.com/drive/18lrKD30kxEUolQ61o5nzUJM0rvWgpbFK#scrollTo=tUFis82oGUAS&line=1&uniqifier=1)

```
from wrapper import Generator

model = Generator(load_dir="gpt2", save_dir="outputs/wrapper/fewrel/unseen_10_seed_0/generator")
model.generate(labels=["location", "religion"], path_out="synthetic.jsonl")
```

### Extract with Pretrained Model | [![Colab](https://img.shields.io/badge/Colab-Code%20Demo-%23fe9f00)](https://colab.research.google.com/drive/18lrKD30kxEUolQ61o5nzUJM0rvWgpbFK#scrollTo=eGxP3vVmID9W&line=1&uniqifier=1)

```
from wrapper import Extractor

model = Extractor(load_dir="facebook/bart-base", save_dir="outputs/wrapper/fewrel/unseen_10_seed_0/extractor_final")
model.predict(path_in=path_test, path_out="pred.jsonl")
```

### Model Training | [![Colab](https://img.shields.io/badge/Colab-Code%20Demo-%23fe9f00)](https://colab.research.google.com/drive/18lrKD30kxEUolQ61o5nzUJM0rvWgpbFK#scrollTo=qi5PAW5ocjfj&line=1&uniqifier=1)

Train the Generator and Extractor models:
```
from pathlib import Path
from wrapper import Generator, Extractor

generator = Generator(
    load_dir="gpt2",
    save_dir=str(Path(save_dir) / "generator"),
)
extractor = Extractor(
    load_dir="facebook/bart-base",
    save_dir=str(Path(save_dir) / "extractor"),
)
generator.fit(path_train, path_dev)
extractor.fit(path_train, path_dev)
```

Generate synthetic data with relation triplets for test labels:
```
generator.generate(labels_test, path_out=path_synthetic)
```

Train the final Extractor model using the synthetic data and predict on test sentences:
```
extractor_final = Extractor(
    load_dir=str(Path(save_dir) / "extractor" / "model"),
    save_dir=str(Path(save_dir) / "extractor_final"),
)
extractor_final.fit(path_synthetic, path_dev)
extractor_final.predict(path_in=path_test, path_out=path_pred)
```

### Experiment Scripts

Run training in [wrapper.py](https://github.com/declare-lab/RelationPrompt/blob/783f33c301813368a5a6e3bdbbe50c47df7647bf/wrapper.py#L370) (You can change "fewrel" to "wiki" or unseen to 5/10/15 or seed to 0/1/2/3/4):
```
python wrapper.py main \
--path_train outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/train.jsonl \                                       
--path_dev outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/dev.jsonl \                                           
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \                                         
--save_dir outputs/wrapper/fewrel/unseen_10_seed_0   
```

Run evaluation (Single-triplet setting)
```
python wrapper.py run_eval \                                                                                               
--path_model outputs/wrapper/fewrel/unseen_10_seed_0/extractor_final \                                                  
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \
--mode single
```

Run evaluation (Multi-triplet setting)
```
python wrapper.py run_eval \                                                                                               
--path_model outputs/wrapper/fewrel/unseen_10_seed_0/extractor_final \                                                  
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \
--mode multi
```

### Research Citation
If the code is useful for your research project, we appreciate if you cite the following [paper](https://arxiv.org/abs/2203.09101):
```
@inproceedings{chia-etal-2022-relationprompt,
    title = "RelationPrompt: Leveraging Prompts to Generate Synthetic Data for Zero-Shot Relation Triplet Extraction",
    author = "Chia, Yew Ken and Bing, Lidong and Poria, Soujanya and Si, Luo",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    year = "2022",
    url = "https://arxiv.org/abs/2203.09101",
    doi = "https://doi.org/10.48550/arXiv.2203.09101",
}
```
