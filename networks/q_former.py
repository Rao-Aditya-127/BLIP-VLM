# example_qformer_init.py
from typing import Literal
import torch
import torch.nn as nn
import copy
import os
import json
from transformers import DistilBertConfig, DistilBertModel


class QueryFFN(nn.Module):
    """Per-layer query-specific FFN, mirroring the original Q-Former's separate
    intermediate_query + output_query pathway (Original_code/Qformer.py BertLayer lines ~399-483).

    In the original, query tokens and text tokens pass through *different* FFN weights
    at every transformer layer. Here we apply this dedicated FFN to the query slice
    after the shared DistilBERT layer has run, giving queries layer-specific parameters."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.fc2(self.act(self.fc1(x))))


def create_attention_mask(
    B,  # batch size
    I,  # image tokens size
    text_presence_mask,  # pad token mask
    mode: Literal["uni_modal", "multi_modal", "multi_modal_causal"],
):

    T = text_presence_mask.size(1)  # text token size
    device = text_presence_mask.device

    mask = torch.zeros(
        B, T + I, T + I, device=device, dtype=torch.bool
    )  # [B, I+T, I+T]

    img_self = torch.ones(B, I, I, device=device, dtype=torch.bool)  # [B, I, I]
    text_self = torch.ones(B, T, T, device=device, dtype=torch.bool)  # [B, T, T]

    if mode == "multi_modal_causal":
        text_self = torch.tril(text_self)

    if mode == "uni_modal":
        multimodal_cross_fn = torch.zeros
    else:
        multimodal_cross_fn = torch.ones

    img_cross = multimodal_cross_fn(B, T, I, device=device, dtype=torch.bool)
    text_cross = multimodal_cross_fn(B, I, T, device=device, dtype=torch.bool)

    mask[:, :T, :T] = text_self
    mask[:, -I:, -I:] = img_self
    mask[:, :T, -I:] = img_cross
    mask[:, -I:, :T] = text_cross

    presence_mask = torch.cat(
        [text_presence_mask, torch.ones(B, I, dtype=torch.bool, device=device)], dim=1
    )
    presence_mask = presence_mask.unsqueeze(2) & presence_mask.unsqueeze(1)
    mask = mask & presence_mask

    # Convert boolean mask to additive mask (DistilBERT uses scores + mask internally).
    # Use a large finite value instead of -inf: softmax([-inf,...,-inf]) = NaN (0/0),
    # while softmax([-1e4,...,-1e4]) = uniform small weights, which is numerically safe.
    # Padded positions still get ~0 attention weight in practice.
    additive = torch.zeros(B, T + I, T + I, device=device, dtype=torch.float)
    additive = additive.masked_fill(~mask, -1e4)
    return additive.unsqueeze(1)  # [B, 1, T+I, T+I]


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        # use MultiheadAttention with batch_first for convenience
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x_queries, kv):
        # x_queries: (B, Q, H), kv: (B, S, H) (visual features)
        attn_out, _ = self.cross_attn(x_queries, kv, kv)  # query, key, value
        x = self.layernorm(x_queries + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class QFormer(nn.Module):
    def __init__(
        self,
        bert_model: DistilBertModel,
        n_queries=32,
        cross_every=2,
        num_heads=12,
    ):
        super().__init__()
        self.cross_every = cross_every
        self.num_heads = num_heads
        self.n_queries = n_queries

        # Save the config of the passed bert model to reconstruct it later
        self.bert_config = bert_model.config

        cfg: DistilBertConfig = bert_model.config
        self.hidden_size = cfg.hidden_size
        self.embeddings = copy.deepcopy(bert_model.embeddings)
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(layer) for layer in bert_model.transformer.layer]
        )
        self.cross_blocks = nn.ModuleDict()
        self.query_ffns = nn.ModuleList()
        for i in range(len(self.encoder_layers)):
            # Fix: original places cross-attention at layers [0, 2, 4, ...] (layer_num % freq == 0)
            if (i % cross_every) == 0:
                self.cross_blocks[str(i)] = CrossAttentionBlock(
                    self.hidden_size, num_heads
                )
            # One query-specific FFN per layer, mirroring BertLayer.intermediate_query
            # + BertLayer.output_query in the original (Qformer.py ~line 399).
            # Weights are copied from the shared layer FFN (same as original's initialization
            # in blip2_qformer.py lines 75-79), so queries start with the same behavior
            # as text tokens and diverge during training.
            qffn = QueryFFN(self.hidden_size)
            layer_ffn = bert_model.transformer.layer[i].ffn
            qffn.fc1.weight.data.copy_(layer_ffn.lin1.weight.data)
            qffn.fc1.bias.data.copy_(layer_ffn.lin1.bias.data)
            qffn.fc2.weight.data.copy_(layer_ffn.lin2.weight.data)
            qffn.fc2.bias.data.copy_(layer_ffn.lin2.bias.data)
            self.query_ffns.append(qffn)

        # Fix: initialize with std≈0.02 matching original BERT initializer_range
        # (original uses zeros().normal_(0, initializer_range) in blip2.py:57-60)
        self.query_embeddings = nn.Parameter(
            torch.zeros(1, n_queries, self.hidden_size).normal_(mean=0.0, std=0.02)
        )

    def save_pretrained(self, save_directory):
        """
        Saves the QFormer model weights and configuration to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config = {
            "n_queries": self.n_queries,
            "cross_every": self.cross_every,
            "num_heads": self.num_heads,
            "bert_config": self.bert_config.to_dict(),
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Loads a QFormer model from a directory containing config.json and pytorch_model.bin.
        """
        # Load config
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)

        bert_config_dict = config.pop("bert_config")
        bert_config = DistilBertConfig.from_dict(bert_config_dict)

        # The weights will be overwritten by load_state_dict later.
        bert_model = DistilBertModel(bert_config)

        model = cls(bert_model=bert_model, **config)
        # Load weights
        state_dict_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return model

    def get_grouped_params(self):
        params = {"default": [], "cross_blocks": [], "query_embeddings": []}
        for name, p in self.named_parameters():
            if "cross_blocks" in name:
                params["cross_blocks"].append(p)
            elif "query_embeddings" in name:
                params["query_embeddings"].append(p)
            else:
                params["default"].append(p)
        return params

    def encode_image(self, visual_feats):
        B = visual_feats.size(0)
        x = self.query_embeddings.expand(B, -1, -1)  # (B, Q, H)
        for i, layer in enumerate(self.encoder_layers):
            result = layer(x)
            x = result[0] if isinstance(result, tuple) else result
            if str(i) in self.cross_blocks:
                x = self.cross_blocks[str(i)](x, visual_feats)
            x = self.query_ffns[i](x)  # query-specific FFN (mirrors intermediate_query)

        return x, x.mean(1)

    def encode_text(self, text_input_ids, text_attention_mask=None):
        """Encode text only through transformer layers (no query tokens, no cross-attention).
        Returns full hidden states [B, T, H]."""
        txt_emb = self.embeddings(input_ids=text_input_ids)  # (B, T, H)
        B, T, _ = txt_emb.shape

        if text_attention_mask is None:
            text_attention_mask = torch.ones(B, T, dtype=torch.bool, device=text_input_ids.device)
        else:
            text_attention_mask = text_attention_mask.to(torch.bool)

        # Text-only self-attention mask: [B, 1, T, T] with padding handled.
        # Use -1e4 instead of -inf to avoid softmax([-inf,...,-inf]) = NaN for fully-padded rows.
        bool_mask = (
            text_attention_mask.unsqueeze(1).unsqueeze(2)
            & text_attention_mask.unsqueeze(1).unsqueeze(3)
        )  # [B, 1, T, T]
        attn_mask = torch.zeros(B, 1, T, T, device=text_input_ids.device, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(~bool_mask, -1e4)

        x = txt_emb
        for i, layer in enumerate(self.encoder_layers):
            result = layer(x, attn_mask)
            x = result[0] if isinstance(result, tuple) else result
            # Skip cross_blocks and query_ffns — text-only encoding

        return x  # [B, T, H]

    def forward(
        self,
        visual_feats,
        text_input_ids,
        text_attention_mask=None,
        attention_mode: Literal[
            "uni_modal", "multi_modal", "multi_modal_causal"
        ] = "uni_modal",
        return_full_sequences=False,
    ):

        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(torch.bool)

        # text embeddings
        txt_emb = self.embeddings(input_ids=text_input_ids)  # (B, T, H)
        B, T, _ = txt_emb.shape
        queries = self.query_embeddings.expand(B, -1, -1)  # (B, Q, H)

        x = torch.cat([txt_emb, queries], dim=1)
        attention_mask = create_attention_mask(
            B=B,
            I=self.n_queries,
            text_presence_mask=text_attention_mask,
            mode=attention_mode,
        )

        for i, layer in enumerate(self.encoder_layers):
            result = layer(x, attention_mask)
            x = result[0] if isinstance(result, tuple) else result
            if str(i) in self.cross_blocks:
                queries = x[:, -self.n_queries:]
                txt_emb = x[:, :-self.n_queries]
                queries = self.cross_blocks[str(i)](queries, visual_feats)
                x = torch.cat([txt_emb, queries], dim=1)
            # Apply query-specific FFN to the query slice only (mirrors BertLayer.output_query)
            queries = x[:, -self.n_queries:]
            queries = self.query_ffns[i](queries)
            x = torch.cat([x[:, :-self.n_queries], queries], dim=1)

        queries = x[:, -self.n_queries:]
        txt_emb = x[:, :-self.n_queries]

        if return_full_sequences:
            return queries, txt_emb  # [B, Q, H], [B, T, H]

        queries_pooled = queries.mean(dim=1)
        txt_emb_pool = txt_emb[:, 0]  # Generally the [CLS] token

        return queries_pooled, txt_emb_pool


if __name__ == "__main__":
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    batch = 2
    seq_len = 5
    num_patches = 49
    d_image = 768
    n_queries = 3

    input_ids = torch.randint(0, 30522, (batch, seq_len), dtype=torch.long)
    image_feats = torch.randn(batch, num_patches, d_image)

    text_attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool
    )
    qformer = QFormer(bert, n_queries=n_queries)

    # Test Forward pass
    qformer.eval()
    with torch.no_grad():
        out_queries, out_text = qformer(
            image_feats,
            input_ids,
            text_attention_mask,
            attention_mode="multi_modal_causal",
        )

    print("Original Output shapes:", out_queries.shape, out_text.shape)

    # Test Save and Load
    save_dir = "tmp_qformer_checkpoints"
    print(f"Saving model to {save_dir}...")
    qformer.save_pretrained(save_dir)

    print(f"Loading model from {save_dir}...")
    loaded_qformer = QFormer.from_pretrained(save_dir)

    # Check sum of all parameters
    sum_orig = sum(p.sum().item() for p in qformer.parameters())
    sum_loaded = sum(p.sum().item() for p in loaded_qformer.parameters())
    print(f"Sum of original weights: {sum_orig}")
    print(f"Sum of loaded weights: {sum_loaded}")
    print(f"Weight sum difference: {abs(sum_orig - sum_loaded)}")

    # Verify weights are loaded correctly (basic check)
    loaded_qformer.eval()
    with torch.no_grad():
        out_queries_loaded, out_text_loaded = loaded_qformer(
            image_feats,
            input_ids,
            text_attention_mask,
            attention_mode="multi_modal_causal",
        )

    print("Loaded Output shapes:", out_queries_loaded.shape, out_text_loaded.shape)

    # Check strict equality of outputs
    diff_queries = (out_queries - out_queries_loaded).abs().max()
    diff_text = (out_text - out_text_loaded).abs().max()

    print(f"Max difference in queries: {diff_queries}")
    print(f"Max difference in text: {diff_text}")

    if diff_queries < 1e-6 and diff_text < 1e-6:
        print("SUCCESS: Model saved and loaded correctly!")
    else:
        print("FAILURE: Model outputs mismatch after loading.")

    # Cleanup
    import shutil

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
