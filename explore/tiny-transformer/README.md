# Tiny Transformer

Can a small Transformer learn to copy a sequence it has never seen? Give it four digits and a separator, such as `3 1 4 2 |`, and ask it to generate `3 1 4 2`. The experiment generates all 4,096 possible inputs from digits 0–7 and holds out 512 for evaluation.

## Run locally

From the repository root, use Python 3.13 and uv:

```sh
uv venv --python 3.13
uv pip install torch numpy
.venv/bin/python explore/tiny-transformer/train.py
```

The script uses one CPU thread and generates its data locally. Try `--steps 0` for an untrained baseline or `--steps 100 --seed 7` for a shorter run with a different initialization and data split.

## Read the model

1. **Embeddings** map each token and each position to a learned vector. Positions let the model distinguish the first digit from the second even if they have the same value.
2. **Causal self-attention** projects each vector into queries, keys, and values. Each of four heads scores the available keys for each query and mixes their values. The triangular mask blocks future positions.
3. **Two decoder blocks** each apply attention and a feed-forward network, with layer normalization and residual connections. The final projection scores the eight possible output digits.
4. **Training** shows the model the correct sequence as context and computes cross-entropy only on the four copy predictions. **Evaluation** starts with the prompt alone and feeds each generated digit back into the model. Exact-copy accuracy counts a prompt as correct only if all four predictions match.

The run prints held-out accuracy, sample predictions, and the attention weights for the separator position, where the model predicts the first copied digit. Look for heads that put most of their weight on the first input position. Each row shows one head's weights for one example. The model combines information across heads, layers, and residual connections.

## First run

With Python 3.13, PyTorch 2.15.0, and the default seed of 42, the 26,280-parameter model copied all 512 held-out prompts correctly by step 100. Before training it copied zero of them exactly. The full 600-step run took about 2.4 seconds on the local CPU, excluding startup and dependency installation.

For the prompt `1 6 4 1 |`, the first layer's fourth head placed 98.9% of the separator's attention on the first position. The same digit also appeared in the fourth position, so the attention row lets you see which occurrence this head focused on.
