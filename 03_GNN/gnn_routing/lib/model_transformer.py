import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerDecoder(nn.Module):
    def __init__(self, dataset, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1, max_len=100, profile=None):
        super().__init__()
        self.vocab_size = dataset.tokenizer.vocab_size()
        self.criterion_tokens = {
            'distance': dataset.tokenizer.special_tokens['<DIST>'],
            'duration': dataset.tokenizer.special_tokens['<DUR>'],
            'car_distance': dataset.tokenizer.special_tokens['<CAR_DIST>'],
            'car_duration': dataset.tokenizer.special_tokens['<CAR_DUR>'],
            'bike_distance': dataset.tokenizer.special_tokens['<BIKE_DIST>'],
            'bike_duration': dataset.tokenizer.special_tokens['<BIKE_DUR>'],
        }
        self.bos_idx = dataset.tokenizer.special_tokens['<BOS>']
        self.eos_idx = dataset.tokenizer.special_tokens['<EOS>']
        self.padding_idx = dataset.tokenizer.special_tokens['<PAD>']
        self.use_context = True

        self.criterion_embedding = nn.Embedding(len(self.criterion_tokens), embed_dim)
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embed_dim, self.vocab_size)

        if self.use_context:
            self.context_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, tgt_seq, memory, tgt_mask=None, start_node=None, goal_node=None, criterion='distance', profile=None):
        batch_size, seq_len = tgt_seq.size()

        tgt_emb = self.token_embedding(tgt_seq)
        pos_enc = self.get_sinusoidal_positional_encoding(seq_len, tgt_emb.size(-1), tgt_seq.device)
        tgt_emb = tgt_emb + pos_enc


        if self.use_context:
            start_emb = memory[torch.arange(batch_size), start_node]  # shape: [batch_size, embed_dim]
            goal_emb = memory[torch.arange(batch_size), goal_node]  # shape: [batch_size, embed_dim]

            context = torch.cat([start_emb, goal_emb], dim=-1)           # [batch, 2*embed_dim]
            context = self.context_proj(context)                         # [batch, embed_dim]
            tgt_emb = tgt_emb + context.unsqueeze(1)

        if profile is not None:
            profile_emb = self.profile_proj(profile).unsqueeze(1)  # [batch, 1, embed_dim]
            tgt_emb = tgt_emb + profile_emb

        output = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        return self.output_proj(output)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool().to(next(self.parameters()).device)

    def get_sinusoidal_positional_encoding(self, seq_len, embed_dim, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: [1, seq_len, embed_dim]
