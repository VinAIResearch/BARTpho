# BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese


Two BARTpho versions `BARTpho-syllable` and `BARTpho-word` are the first public large-scale monolingual sequence-to-sequence models pre-trained for Vietnamese. BARTpho uses the "large" architecture and pre-training scheme of the sequence-to-sequence denoising model [BART](https://github.com/pytorch/fairseq/tree/main/examples/bart), thus especially suitable for generative NLP tasks. Experiments on a downstream task of Vietnamese text summarization show that in both automatic and human evaluations, BARTpho outperforms the strong baseline [mBART](https://github.com/pytorch/fairseq/tree/main/examples/mbart) and improves the state-of-the-art.

The general architecture and experimental results of BARTpho can be found in our [paper](https://arxiv.org/abs/2109.09701):

	@article{bartpho,
	title     = {{BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese}},
	author    = {Nguyen Luong Tran and Duong Minh Le and Dat Quoc Nguyen},
	journal   = {arXiv preprint},
	volume    = {arXiv:2109.09701},
	year      = {2021}
	}

**Please CITE** our paper when BARTpho is used to help produce published results or incorporated into other software.

## <a name="fairseq"></a> Using BARTpho in [`fairseq`](https://github.com/pytorch/fairseq)

### Pre-trained models

Model | #params | Download | Input text
---|---|---|---
`BARTpho-syllable` | 396M | [fairseq-bartpho-syllable.zip](https://drive.google.com/file/d/1iw44DztS03JyVP9IcJx0Jh2q_3Y63oio/view?usp=sharing) | Syllable level
`BARTpho-word` | 420M | [fairseq-bartpho-word.zip](https://drive.google.com/file/d/1j23nCYQlqwwFQPpcwiogfZ9VHDHIO0UD/view?usp=sharing) | Word level

- `unzip fairseq-bartpho-syllable.zip`
- `unzip fairseq-bartpho-word.zip`

### Example usage

```python
from fairseq.models.bart import BARTModel  
 
#Encode an input text: OOV tokens are converted into <unk>
def encode(fairseq_hub, sentence):  
    tokens = fairseq_hub.bpe.encode(sentence)  
    if len(tokens.split(" ")) > fairseq_hub.max_positions - 2:  
        tokens = " ".join(tokens.split(" ")[: fairseq_hub.max_positions - 2])  
    bpe_sentence = "<s> " + tokens + " </s>"
    tokens = fairseq_hub.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)  
    return tokens.long()

#Load BARTpho-syllable model:  
model_folder_path = '/PATH-TO-FOLDER/fairseq-bartpho-syllable/'  
spm_model_path = '/PATH-TO-FOLDER/fairseq-bartpho-syllable/sentence.bpe.model'  
bartpho_syllable = BARTModel.from_pretrained(model_folder_path, checkpoint_file='model.pt', bpe='sentencepiece', sentencepiece_model=spm_model_path).eval()
#Input syllable-level/raw text:  
sentence = 'Chúng tôi là những nghiên cứu viên.'  
#Apply SentencePiece to the input text
tokenIDs = encode(bartpho_syllable, sentence)
#Extract features from BARTpho-syllable
last_layer_features = bartpho_syllable.extract_features(tokenIDs)

##Load BARTpho-word model:  
model_folder_path = '/PATH-TO-FOLDER/fairseq-bartpho-word/'  
bpe_codes_path = '/PATH-TO-FOLDER/fairseq-bartpho-word/bpe.codes'  
bartpho_word = BARTModel.from_pretrained(model_folder_path, checkpoint_file='model.pt', bpe='fastbpe', bpe_codes=bpe_codes_path).eval()
#Input word-level text:  
sentence = 'Chúng_tôi là những nghiên_cứu_viên .'  
#Apply BPE to the input text
tokenIDs = encode(bartpho_word, sentence)
#Extract features from BARTpho-word
last_layer_features = bartpho_word.extract_features(tokenIDs)
```

## <a name="fairseq"></a> Using BARTpho in [`transformers`](https://github.com/huggingface/transformers)

To be updated soon!


## License
    
    MIT License

    Copyright (c) 2021 VinAI Research

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