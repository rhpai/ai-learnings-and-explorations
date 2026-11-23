"""Train a decoder-only Transformer to repeat a sequence of four digits."""

import argparse
import time

import torch
from torch import nn
from torch.nn import functional as F

DIGITS = 8
COPY_LENGTH = 4
SEPARATOR = DIGITS
CONTEXT = 2 * COPY_LENGTH
WIDTH = 32
HEADS = 4


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.output = nn.Linear(WIDTH, WIDTH)
        self.register_buffer(
            "future", torch.ones(CONTEXT, CONTEXT, dtype=torch.bool).triu(1)
        )

    def forward(self, x):
        batch, length, _ = x.shape
        head_width = WIDTH // HEADS
        qkv = self.qkv(x).reshape(batch, length, 3, HEADS, head_width)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind()
        scores = (queries @ keys.transpose(-2, -1)) / head_width**0.5
        weights = scores.masked_fill(self.future[:length, :length], -torch.inf)
        weights = weights.softmax(dim=-1)
        attended = (weights @ values).transpose(1, 2).reshape(batch, length, WIDTH)
        return self.output(attended), weights


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.attention = CausalAttention()
        self.mlp_norm = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(
            nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH)
        )

    def forward(self, x):
        attended, weights = self.attention(self.attention_norm(x))
        x = x + attended
        return x + self.mlp(self.mlp_norm(x)), weights


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(DIGITS + 1, WIDTH)
        self.positions = nn.Embedding(CONTEXT, WIDTH)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.readout = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, DIGITS))

    def forward(self, tokens):
        x = self.tokens(tokens) + self.positions(torch.arange(tokens.size(1)))
        attention = []
        for block in self.blocks:
            x, weights = block(x)
            attention.append(weights)
        return self.readout(x), attention


def prompt(digits):
    separator = torch.full((digits.size(0), 1), SEPARATOR)
    return torch.cat((digits, separator), dim=1)


@torch.no_grad()
def generate(model, tokens):
    """Predict one digit at a time, feeding each prediction back into the model."""
    for _ in range(COPY_LENGTH):
        logits, _ = model(tokens)
        next_digit = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_digit), dim=1)
    return tokens[:, -COPY_LENGTH:]


@torch.no_grad()
def report(model, digits, step):
    predictions = generate(model, prompt(digits))
    matches = predictions.eq(digits)
    token_accuracy = matches.float().mean().item()
    exact_accuracy = matches.all(dim=1).float().mean().item()
    print(f"step {step:4d} | digit accuracy {token_accuracy:6.1%} "
          f"| exact copy {exact_accuracy:6.1%}")


@torch.no_grad()
def show_attention(model, digits):
    tokens = prompt(digits[:1])
    _, layers = model(tokens)
    labels = [str(digit) for digit in digits[0].tolist()] + ["|"]
    print("Attention from the separator, which predicts the first copied digit")
    print("Keys             " + " ".join(f"{label:>6}" for label in labels))
    for layer, weights in enumerate(layers, start=1):
        for head, row in enumerate(weights[0, :, -1], start=1):
            print(f"layer {layer} head {head}   " + " ".join(f"{w:6.1%}" for w in row))


def interactive_prompt(model):
    print("\nEnter four digits from 0 to 7, such as 3 1 4 2. Type q to quit.")
    while True:
        try:
            raw = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if raw.lower() == "q":
            return
        tokens = "".join(raw.split())
        if len(tokens) != COPY_LENGTH or any(token not in "01234567" for token in tokens):
            print("Please enter exactly four digits from 0 to 7.")
            continue
        digits = torch.tensor([[int(token) for token in tokens]])
        prediction = generate(model, prompt(digits))[0].tolist()
        print("Predicted: " + " ".join(map(str, prediction)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interactive", action="store_true", help="Try prompts after training")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = TinyTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"{parameter_count:,} trainable parameters on CPU")

    # Split all 8**4 possible prompts into disjoint train and evaluation sets.
    order = torch.randperm(DIGITS**COPY_LENGTH)
    powers = DIGITS ** torch.arange(COPY_LENGTH - 1, -1, -1)
    sequences = (order[:, None] // powers) % DIGITS
    test, train = sequences[:512], sequences[512:]

    start = time.perf_counter()
    report(model, test, step=0)
    for step in range(1, args.steps + 1):
        digits = train[torch.randint(len(train), (64,))]
        sequence = torch.cat((prompt(digits), digits), dim=1)
        logits, _ = model(sequence[:, :-1])
        # The output at the separator predicts the first digit in the copy.
        copy_logits = logits[:, COPY_LENGTH:]
        loss = F.cross_entropy(copy_logits.flatten(0, 1), digits.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            print(f"training loss {loss.item():.4f}")
            report(model, test, step)

    print(f"Elapsed time {time.perf_counter() - start:.1f}s")
    copied = generate(model, prompt(test[:3]))
    for digits, prediction in zip(test[:3].tolist(), copied.tolist()):
        print(f"{digits} | predicted {prediction}")
    show_attention(model, test)
    if args.interactive:
        interactive_prompt(model)


if __name__ == "__main__":
    main()
