# Neural Machine Translation with Bahdanau Attention

A sequence-to-sequence neural machine translation (NMT) system built with TensorFlow/Keras. It uses a GRU encoder–decoder with **Bahdanau (additive) attention** and is trained on the Stanford NLP translation datasets (English → Vietnamese, with English ↔ German data also included).

## How it works

```
source sentence ──► Encoder (Embedding + GRU) ──► hidden states ─┐
                                                                  ▼
                                                    Bahdanau Attention
                                                                  │ context vector
                                                                  ▼
<start> ──► Decoder (Embedding + GRU + Dense) ──► next token ──► … ──► <end>
```

- **Encoder** (`src/encoder.py`): embeds the source tokens and runs them through a GRU, producing a hidden state for every input position.
- **Bahdanau attention** (`src/bahdanau_attention.py`): at every decoding step, scores each encoder state against the current decoder state, `score = V · tanh(W1·h_enc + W2·h_dec)`, and builds a weighted context vector from the encoder states. This lets the model focus on the relevant source words instead of compressing the whole sentence into one fixed vector.
- **Decoder** (`src/decoder.py`): joins the context vector with the embedded previous token, passes it through a GRU, and predicts the next word over the target vocabulary.
- **Training** uses teacher forcing, a masked sparse categorical cross-entropy loss, and the Adam optimizer.
- **Evaluation** decodes greedily and reports sentence-level BLEU (NLTK, smoothing method 1).

## Project structure

```
.
├── data/                          # Datasets (stored with Git LFS)
│   ├── train.envi.txt / train.vi.txt     # IWSLT'15 English–Vietnamese training set
│   ├── tst2012.en.txt / tst2012.vi.txt   # English–Vietnamese test set
│   ├── train.en.txt / train.de.txt       # WMT'14 English–German (full)
│   ├── train.10k.*.txt / test.100.*.txt  # Small English–German subsets
│   └── translate.txt                     # Scratch file for single-sentence translation
└── src/
    ├── neural_machine_translator.py  # Main entry point: train / test / translate
    ├── encoder.py                    # GRU encoder
    ├── decoder.py                    # GRU decoder with attention
    ├── bahdanau_attention.py         # Additive attention layer
    ├── preprocess.py                 # Normalisation, tokenisation, train/val split
    └── NMT_v1.py, NMT_v1.2.py,       # Earlier experiments (BiGRU / BiLSTM baselines
        nmt_v2.py, nmt_v2.1.py        #   and Functional-API attention models)
```

## Getting started

### Prerequisites

- Python 3.8+
- [Git LFS](https://git-lfs.com/) (the files in `data/` are LFS pointers until pulled)

### Installation

```bash
git clone https://github.com/shivakumar-hassan-lokesh/neural-machine-translation-attention.git
cd neural-machine-translation-attention

# Download the datasets
git lfs install
git lfs pull

# Install dependencies
pip install tensorflow numpy nltk scikit-learn
```

> **Note:** the code uses `keras.optimizers.legacy.Adam` and `keras.preprocessing`, so it needs a TensorFlow/Keras 2.x release (for example TensorFlow 2.11–2.15).

## Usage

Run every command from the repository root so the relative `./data/` paths resolve.

**Train**

```bash
python src/neural_machine_translator.py --fun train --ckpt ./checkpoints
```

**Evaluate on the test set (BLEU)**

```bash
python src/neural_machine_translator.py --fun test --ckpt ./checkpoints
```

**Translate a single sentence**

```bash
python src/neural_machine_translator.py --fun translate --translate "how are you ?" --ckpt ./checkpoints
```

### Command-line options

| Flag             | Default        | Description                                    |
|------------------|----------------|------------------------------------------------|
| `--fun`          | `train`        | Mode: `train`, `test`, or `translate`          |
| `--data_path`    | `./data/`      | Directory containing the dataset files         |
| `--num_examples` | `30000`        | Number of sentence pairs to load               |
| `--translate`    | `hello world!` | Sentence to translate in `translate` mode      |
| `--ckpt`         | `/ckpt`        | Checkpoint directory to restore from           |
| `--id_gpu`       | `-1`           | GPU index to use (`-1` runs on the CPU)        |

### Default hyperparameters

| Parameter       | Value |
|-----------------|-------|
| Embedding size  | 256   |
| GRU units       | 1024  |
| Batch size      | 64    |
| Epochs          | 10    |
| Validation split| 20%   |

A checkpoint is written to `./checkpoints/` every 2 epochs.

## Known limitations

- Training always saves to `./checkpoints/`, while `--ckpt` only controls where checkpoints are restored from. Pass `--ckpt ./checkpoints` so the two match.
- `test` mode opens `data/tst2012.en` and `data/tst2012.vi` (without the `.txt` extension). Rename or symlink the test files, or update the paths in `neural_machine_translator.py`.
- Preprocessing strips everything except ASCII letters and basic punctuation, so Vietnamese diacritics are removed and words outside the training vocabulary cannot be translated.
- The main pipeline is wired for English → Vietnamese. The German data is used only by the experimental scripts.

## References

- Bahdanau, Cho & Bengio, [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) (ICLR 2015)
- TensorFlow tutorial: [Neural machine translation with attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention), which the encoder, decoder, and attention code is adapted from
- [Stanford NLP NMT datasets](https://nlp.stanford.edu/projects/nmt/)
