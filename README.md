#### Table of contents
1. [Introduction](#introduction)
2. [Using BARTpho with `transformers`](#transformers)
3. [Using BARTpho with `fairseq`](#fairseq)
4. [Notes](#notes)

# <a name="introduction"></a> BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese


> We present BARTpho with two versions, BARTpho-syllable and BARTpho-word, which are the first public large-scale monolingual sequence-to-sequence models pre-trained for Vietnamese. BARTpho uses the "large" architecture and the pre-training scheme of the sequence-to-sequence denoising autoencoder BART, thus it is especially suitable for generative NLP tasks. We conduct experiments to compare our BARTpho with its competitor mBART on a downstream task of Vietnamese text summarization and show that: in both automatic and human evaluations, BARTpho outperforms the strong baseline mBART and improves the state-of-the-art. We further evaluate and compare BARTpho and mBART on the Vietnamese capitalization and punctuation restoration tasks and also find that BARTpho is more effective than mBART on these two tasks.

The general architecture and experimental results of BARTpho can be found in our [paper](https://arxiv.org/abs/2109.09701):

	@inproceedings{bartpho,
	    title     = {{BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese}},
	    author    = {Nguyen Luong Tran and Duong Minh Le and Dat Quoc Nguyen},
	    booktitle = {Proceedings of the 23rd Annual Conference of the International Speech Communication Association},
	    year      = {2022}
	}

**Please CITE** our paper when BARTpho is used to help produce published results or incorporated into other software.


## <a name="transformers"></a> Using BARTpho in [`transformers`](https://github.com/huggingface/transformers)

### Installation
- Install `transformers` with pip: `pip install transformers`, or [install `transformers` from source](https://huggingface.co/docs/transformers/installation#installing-from-source).  <br />
Note that we merged a slow tokenizer for BARTpho into the main `transformers` branch. The process of merging a fast tokenizer for BARTpho is in the discussion, as detailed in [this pull request](https://github.com/huggingface/transformers/pull/17254). If users would like to utilize the fast tokenizer, the users might install `transformers` as follows:

```
git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
cd transformers
pip install -e .
```

- Install `sentencepiece` and `tokenizers` with pip: `pip install sentencepiece tokenizers`

### Pre-trained models

Model | #params | Arch. | Max length | Input text
---|---|---|---|---
[`vinai/bartpho-syllable-base`](https://huggingface.co/vinai/bartpho-syllable-base) | 132M | base | 1024 | Syllable level
[`vinai/bartpho-syllable`](https://huggingface.co/vinai/bartpho-syllable) | 396M | large | 1024 | Syllable level
[`vinai/bartpho-word-base`](https://huggingface.co/vinai/bartpho-word-base) | 150M | base | 1024 | Word level
[`vinai/bartpho-word`](https://huggingface.co/vinai/bartpho-word) | 420M | large | 1024 | Word level

### Example usage

```python3
import torch
from transformers import AutoModel, AutoTokenizer

#BARTpho-syllable
syllable_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
bartpho_syllable = AutoModel.from_pretrained("vinai/bartpho-syllable")
TXT = 'Chúng tôi là những nghiên cứu viên.'  
input_ids = syllable_tokenizer(TXT, return_tensors='pt')['input_ids']
features = bartpho_syllable(input_ids)

#BARTpho-word
word_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
bartpho_word = AutoModel.from_pretrained("vinai/bartpho-word")
TXT = 'Chúng_tôi là những nghiên_cứu_viên .'  
input_ids = word_tokenizer(TXT, return_tensors='pt')['input_ids']
features = bartpho_word(input_ids)
```

## <a name="fairseq"></a> Using BARTpho in [`fairseq`](https://github.com/pytorch/fairseq)

### Installation

There is an issue w.r.t. the `encode` function in the BART hub_interface, as discussed in this pull request [https://github.com/pytorch/fairseq/pull/3905](https://github.com/pytorch/fairseq/pull/3905). While waiting for this pull request's approval, please install `fairseq` as follows:

	git clone https://github.com/datquocnguyen/fairseq.git
	cd fairseq
	pip install --editable ./

### Pre-trained models

Model | #params | Download | Input text
---|---|---|---
BARTpho-syllable | 396M | [fairseq-bartpho-syllable.zip](https://drive.google.com/file/d/1iw44DztS03JyVP9IcJx0Jh2q_3Y63oio/view?usp=sharing) | Syllable level
BARTpho-word | 420M | [fairseq-bartpho-word.zip](https://drive.google.com/file/d/1j23nCYQlqwwFQPpcwiogfZ9VHDHIO0UD/view?usp=sharing) | Word level

- `unzip fairseq-bartpho-syllable.zip`
- `unzip fairseq-bartpho-word.zip`

### Example usage

```python
from fairseq.models.bart import BARTModel  

#Load BARTpho-syllable model:  
model_folder_path = '/PATH-TO-FOLDER/fairseq-bartpho-syllable/'  
spm_model_path = '/PATH-TO-FOLDER/fairseq-bartpho-syllable/sentence.bpe.model'  
bartpho_syllable = BARTModel.from_pretrained(model_folder_path, checkpoint_file='model.pt', bpe='sentencepiece', sentencepiece_model=spm_model_path).eval()
#Input syllable-level/raw text:  
sentence = 'Chúng tôi là những nghiên cứu viên.'  
#Apply SentencePiece to the input text
tokenIDs = bartpho_syllable.encode(sentence, add_if_not_exist=False)
#Extract features from BARTpho-syllable
last_layer_features = bartpho_syllable.extract_features(tokenIDs)

##Load BARTpho-word model:  
model_folder_path = '/PATH-TO-FOLDER/fairseq-bartpho-word/'  
bpe_codes_path = '/PATH-TO-FOLDER/fairseq-bartpho-word/bpe.codes'  
bartpho_word = BARTModel.from_pretrained(model_folder_path, checkpoint_file='model.pt', bpe='fastbpe', bpe_codes=bpe_codes_path).eval()
#Input word-level text:  
sentence = 'Chúng_tôi là những nghiên_cứu_viên .'  
#Apply BPE to the input text
tokenIDs = bartpho_word.encode(sentence, add_if_not_exist=False)
#Extract features from BARTpho-word
last_layer_features = bartpho_word.extract_features(tokenIDs)
```

## <a name="notes"></a> Notes

-  Before fine-tuning BARTpho on a downstream task, users should perform Vietnamese tone normalization on the downstream task's data as this pre-process was also applied to the pre-training corpus. A Python script for Vietnamese tone normalization is available at [HERE](https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md).
- For `BARTpho-word`, users should use [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) to segment input raw texts as it was used to perform both Vietnamese tone normalization and word segmentation on the pre-training corpus. 


## License
    
    MIT License

    Copyright (c) 2021 VinAI

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
