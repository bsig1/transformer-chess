import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=256):
        super().__init__()
        positional_table = torch.zeros(max_len, num_hiddens)
        positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        angular_speeds = torch.exp(
            torch.arange(0, num_hiddens, 2).float() *
            (-math.log(10000.0) / num_hiddens)
        )
        positional_table[:, 0::2] = torch.sin(positions * angular_speeds)
        positional_table[:, 1::2] = torch.cos(positions * angular_speeds)
        # (1, max_len, num_hiddens)
        self.register_buffer("positional_table", positional_table.unsqueeze(0))

    def forward(self, token_embeddings):
        return token_embeddings + self.positional_table[:, : token_embeddings.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            num_hiddens, num_heads, dropout=dropout, batch_first=True)
        self.addnorm1 = nn.LayerNorm(num_hiddens)
        self.ffn = nn.Sequential(
            nn.Linear(num_hiddens, 4 * num_hiddens),
            nn.ReLU(),
            nn.Linear(4 * num_hiddens, num_hiddens),
        )
        self.addnorm2 = nn.LayerNorm(num_hiddens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask):
        attention_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states, attn_mask=attention_mask
        )
        hidden_states = self.addnorm1(
            hidden_states + self.dropout(attention_output))
        hidden_states = self.addnorm2(
            hidden_states + self.dropout(self.ffn(hidden_states)))
        return hidden_states


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens=64, num_heads=4, num_layers=2, max_len=64):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, max_len=max_len)
        self.blks = nn.ModuleList(
            [EncoderBlock(num_hiddens, num_heads) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(num_hiddens)
        self.output_projection = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, targets=None):
        batch_size, num_steps = X.shape
        if num_steps > self.max_len:
            raise ValueError(f"sequence length {
                             num_steps} > max_len {self.max_len}")
        # Causal mask: token i can only attend to <= i
        causal_mask = torch.triu(torch.ones(
            num_steps, num_steps, device=X.device, dtype=torch.bool), diagonal=1)
        hidden_states = self.pos_encoding(self.embedding(X))
        for blk in self.blks:
            hidden_states = blk(hidden_states, causal_mask)
        logits = self.output_projection(self.final_ln(hidden_states))
        loss = None
        if targets is not None:
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1))
        return logits, loss


def build_vocab(tokens):
    token_to_idx = {"<unk>": 0}
    for token in tokens:
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token


def encode(tokens, token_to_idx):
    return [token_to_idx.get(token, 0) for token in tokens]


def get_batch(token_id_sequences, batch_size, block_size, device):
    valid_sequences = [
        sequence for sequence in token_id_sequences if len(sequence) >= block_size + 1
    ]
    if not valid_sequences:
        raise ValueError(
            "Need at least one data sequence with block_size + 1 tokens; provide longer --data or reduce --block-size"
        )

    sequence_choices = torch.randint(0, len(valid_sequences), (batch_size,))
    X_list, Y_list = [], []
    for sequence_index in sequence_choices.tolist():
        sequence = valid_sequences[sequence_index]
        start_pos = torch.randint(0, len(sequence) - block_size, (1,)).item()
        X_list.append(sequence[start_pos: start_pos + block_size])
        Y_list.append(sequence[start_pos + 1: start_pos + block_size + 1])
    X = torch.stack(X_list).to(device)
    Y = torch.stack(Y_list).to(device)
    return X, Y


@torch.no_grad()
def predict_next_move(model, prompt, token_to_idx, idx_to_token, device):
    model.eval()
    prompt_ids = torch.tensor(
        [encode(prompt.split(), token_to_idx)], dtype=torch.long, device=device)
    if prompt_ids.size(1) > model.max_len:
        prompt_ids = prompt_ids[:, -model.max_len:]
    logits, _ = model(prompt_ids)
    next_token_id = int(logits[0, -1].argmax())
    return idx_to_token[next_token_id]


def main():
    data_inputs = []

    with open('data.txt') as f:
        for line in f.readlines():
            data_inputs.append(line.strip())

    start_model_path = None
    output_model_path = "model.pt"
    prompt = "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O"
    steps = 3000
    batch_size = 32
    block_size = 8
    lr = 3e-3

    token_sequences = [data_str.split()
                       for data_str in data_inputs if data_str.split()]

    tokens = [token for sequence in token_sequences for token in sequence]
    token_to_idx, idx_to_token = build_vocab(tokens)
    token_id_sequences = [
        torch.tensor(encode(sequence, token_to_idx), dtype=torch.long)
        for sequence in token_sequences
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyTransformerLM(vocab_size=len(
        token_to_idx), max_len=block_size).to(device)
    if start_model_path:
        checkpoint = torch.load(start_model_path, map_location=device)
        model_state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()

    pbar = tqdm(range(steps), desc=f"Training...")
    for step in pbar:
        batch_inputs, batch_targets = get_batch(
            token_id_sequences, batch_size, block_size, device)
        _, loss = model(batch_inputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            loss = f"{loss.item():.4f}"
            pbar.set_description(f"Training... | Loss = {loss}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token_to_idx": token_to_idx,
            "idx_to_token": idx_to_token,
            "block_size": block_size,
        },
        output_model_path,
    )
    print(f"saved model checkpoint to: {output_model_path}")
    print("next:", predict_next_move(
        model, prompt, token_to_idx, idx_to_token, device))


if __name__ == "__main__":
    main()
