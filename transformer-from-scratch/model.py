import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, out_dim)

    def forward(self, x):
        return self.embedding_layer(x) * math.sqrt(self.out_dim)


class PositionalEncoding(nn.Module):

    def __init__(self, out_dim: int, seq_len: int, dropout: float):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, out_dim)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_dim, 2).float() * (-math.log(10000.0) / out_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, out_dim: int, h: int, dropout: float):
        super().__init__()
        self.out_dim = out_dim
        self.h = h
        assert out_dim % h == 0, f'out_dim: {out_dim} is not divisible by h: {h}'

        self.h_dim = out_dim // h
        self.w_q = nn.Linear(out_dim, out_dim, bias=False)
        self.w_k = nn.Linear(out_dim, out_dim, bias=False)
        self.w_v = nn.Linear(out_dim, out_dim, bias=False)
        self.w_o = nn.Linear(out_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        out_dim = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(out_dim)

        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.h_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.h_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.h_dim).transpose(1, 2)
        x, self.attention = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.h_dim)

        return self.w_o(x)


class LayerNorm(nn.Module):

    def __init__(self, out_dim, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, out_dim, hid_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.layer(x)


class ResidualConnection(nn.Module):
    def __init__(self, out_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(out_dim)
    
    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.layer_norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, out_dim, attention, feed_forward, dropout):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ ResidualConnection(out_dim, dropout) for _ in range(2) ])
    
    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, out_dim, layers):
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNorm(out_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class DecodeBlock(nn.Module):
    def __init__(self, out_dim, attention, cross_attention, feed_forward, dropout):
        super().__init__()
        self.attention = attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ ResidualConnection(out_dim, dropout) for _ in range(3) ])
    
    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, decoder_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, encoder_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, out_dim, layers) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNorm(out_dim)

    def forward(self, x, encoder_output, encode_mask, decode_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, encode_mask, decode_mask)
        return self.layer_norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, out_dim, vocab_size):
        super().__init__()
        self.projection_layer = nn.Linear(out_dim, vocab_size)

    def forward(self, x):
        return self.projection_layer(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, input_emb, output_emb, input_pos, output_pos, projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_emb = input_emb
        self.output_emb = output_emb
        self.input_pos = input_pos
        self.output_pos = output_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.input_emb(src)
        src = self.input_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, output, output_mask):
        output = self.output_emb(output)
        output = self.output_pos(output)
        return self.decoder(output, encoder_output, src_mask, output_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(vocab_size, seq_len=1024, out_dim=512, N=4, h=10, dropout=0.1, hid_dim=1024):
    input_emb = InputEmbedding(vocab_size=vocab_size, out_dim=out_dim)
    output_emb = InputEmbedding(vocab_size=vocab_size, out_dim=out_dim)

    input_pos = PositionalEncoding(out_dim=out_dim, seq_len=seq_len, dropout=dropout)
    output_pos = PositionalEncoding(out_dim=out_dim, seq_len=seq_len, dropout=dropout)

    encoder_blocks = []
    for i in range(N):
        attention = MultiHeadAttentionBlock(out_dim, h, dropout)
        feed_forward = FeedForward(out_dim, hid_dim, dropout)
        encoder_block = EncoderBlock(out_dim, attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for i in range(10):
        attention = MultiHeadAttentionBlock(out_dim, h, dropout)
        cross_attention = MultiHeadAttentionBlock(out_dim, h, dropout)
        feed_forward = FeedForward(out_dim, hid_dim, dropout)
        decoder_block = DecodeBlock(out_dim, attention, cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(out_dim, nn.ModuleList(encoder_blocks))
    decoder = Decoder(out_dim, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(out_dim=out_dim, vocab_size=vocab_size)

    transformer = Transformer(encoder=encoder, decoder=decoder, input_emb=input_emb, output_emb=output_emb, input_pos=input_pos, output_pos=output_pos, projection_layer=projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return transformer
    


