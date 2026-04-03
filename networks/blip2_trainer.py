import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from networks.q_former import QFormer


class Blip2QFormerTrainer(nn.Module):
    """Wrapper around QFormer that adds the projection heads and loss computation
    needed for BLIP-2 Stage 1 pre-training (ITC + ITM losses)."""

    def __init__(self, qformer: QFormer, embed_dim=256):
        super().__init__()
        self.qformer = qformer
        hidden_size = qformer.hidden_size

        # ITC projection heads
        self.vision_proj = nn.Linear(hidden_size, embed_dim)
        self.text_proj = nn.Linear(hidden_size, embed_dim)

        # ITM binary classifier
        self.itm_head = nn.Linear(hidden_size, 2)

        # Learnable temperature for ITC (init 0.07, matching original BLIP-2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def compute_itc(self, image_embeds, text_input_ids, text_attention_mask):
        """Compute Image-Text Contrastive loss.

        Uses separate encoding paths (uni-modal):
        - encode_image: queries attend to image via cross-attention only
        - encode_text: text through transformer without queries or cross-attention

        Returns loss_itc, sim_i2t, sim_t2i (similarity matrices needed for ITM mining).
        """
        # Encode image through queries + cross-attention
        query_output, _ = self.qformer.encode_image(image_embeds)  # [B, Q, H]

        # Encode text independently (no queries, no cross-attention)
        text_output = self.qformer.encode_text(text_input_ids, text_attention_mask)  # [B, T, H]

        # Project and normalize
        image_feats = F.normalize(self.vision_proj(query_output), dim=-1)  # [B, Q, embed_dim]
        text_feat = F.normalize(self.text_proj(text_output[:, 0, :]), dim=-1)  # [B, embed_dim] (CLS)

        B = image_feats.size(0)

        # Per-query similarity, then max across queries (BLIP-2 approach)
        # image_feats: [B, Q, D], text_feat: [B, D]
        sim_q2t = torch.einsum("bqd,nd->bnq", image_feats, text_feat)  # [B, N, Q]
        sim_i2t, _ = sim_q2t.max(dim=-1)  # [B, N]
        sim_i2t = sim_i2t / self.temp

        sim_t2q = torch.einsum("nd,bqd->nbq", text_feat, image_feats)  # [N, B, Q]
        sim_t2i, _ = sim_t2q.max(dim=-1)  # [N, B]
        sim_t2i = sim_t2i / self.temp

        # Targets: diagonal (image i matches text i)
        targets = torch.arange(B, device=image_feats.device)

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        return loss_itc, sim_i2t, sim_t2i

    def compute_itm(self, image_embeds, text_input_ids, text_attention_mask, sim_i2t, sim_t2i):
        """Compute Image-Text Matching loss with hard negative mining.

        Uses the ITC similarity matrices to sample hard negatives, then runs
        Q-Former in multi-modal bidirectional mode to classify match/no-match.
        """
        B = image_embeds.size(0)
        device = image_embeds.device

        # --- Hard negative mining (no gradient) ---
        with torch.no_grad():
            sim_t2i_neg = sim_t2i.clone()
            sim_i2t_neg = sim_i2t.clone()

            # Mask out positive pairs so they can't be selected as negatives
            sim_t2i_neg.fill_diagonal_(-10000)
            sim_i2t_neg.fill_diagonal_(-10000)

            # Higher similarity = more likely to be sampled as hard negative
            weights_t2i = F.softmax(sim_t2i_neg, dim=1)  # [B, B]
            weights_i2t = F.softmax(sim_i2t_neg, dim=1)  # [B, B]

        # For each text, select a hard negative image
        neg_image_indices = torch.multinomial(weights_t2i, 1).squeeze(1)  # [B]
        image_embeds_neg = image_embeds[neg_image_indices]  # [B, P, H_v]

        # For each image, select a hard negative text
        neg_text_indices = torch.multinomial(weights_i2t, 1).squeeze(1)  # [B]
        text_ids_neg = text_input_ids[neg_text_indices]  # [B, T]
        text_atts_neg = text_attention_mask[neg_text_indices]  # [B, T]

        # --- Build 3-group batch ---
        # Group 1: positive (matched_img, matched_txt) -> label 1
        # Group 2: (neg_img, matched_txt) -> label 0
        # Group 3: (matched_img, neg_txt) -> label 0
        text_ids_all = torch.cat([text_input_ids, text_input_ids, text_ids_neg], dim=0)  # [3B, T]
        text_atts_all = torch.cat([text_attention_mask, text_attention_mask, text_atts_neg], dim=0)  # [3B, T]
        image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds], dim=0)  # [3B, P, H_v]

        itm_labels = torch.cat(
            [torch.ones(B, dtype=torch.long, device=device),
             torch.zeros(2 * B, dtype=torch.long, device=device)],
            dim=0,
        )  # [3B]

        # --- Forward through Q-Former in multi-modal bidirectional mode ---
        query_output_itm, _ = self.qformer(
            visual_feats=image_embeds_all,
            text_input_ids=text_ids_all,
            text_attention_mask=text_atts_all,
            attention_mode="multi_modal",
            return_full_sequences=True,
        )  # [3B, Q, H]

        # Classify: itm_head on query outputs, averaged across queries
        vl_output = self.itm_head(query_output_itm)  # [3B, Q, 2]
        logits = vl_output.mean(dim=1)  # [3B, 2]

        loss_itm = F.cross_entropy(logits, itm_labels)
        return loss_itm

    def forward(self, image_embeds, text_input_ids, text_attention_mask):
        """Full forward: compute ITC + ITM losses."""
        # ITC
        loss_itc, sim_i2t, sim_t2i = self.compute_itc(
            image_embeds, text_input_ids, text_attention_mask
        )

        # ITM (uses ITC similarity scores for hard negative mining)
        loss_itm = self.compute_itm(
            image_embeds, text_input_ids, text_attention_mask, sim_i2t, sim_t2i
        )

        loss = loss_itc + loss_itm

        return {"loss": loss, "loss_itc": loss_itc, "loss_itm": loss_itm}

    def get_optimizer_params(self, lr, weight_decay=0.01):
        """Return parameter groups with appropriate learning rates and weight decay.

        Follows original BLIP-2 convention (blip2.py):
        - Pretrained DistilBERT layers: lr * 0.1 (fine-tune slowly)
        - New modules (cross-attention, query tokens, heads): full lr
        - Biases and LayerNorm parameters: no weight decay
        - Temperature scalar: no weight decay
        """
        # Split each group into decay / no-decay (biases and norms shouldn't be decayed)
        def split_decay(params):
            decay, no_decay = [], []
            for p in params:
                if p.dim() == 1:  # bias or norm scale (1-D)
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        qformer_groups = self.qformer.get_grouped_params()

        default_decay, default_no_decay = split_decay(qformer_groups["default"])
        cross_decay, cross_no_decay = split_decay(qformer_groups["cross_blocks"])
        new_params = (
            qformer_groups["query_embeddings"]
            + list(self.vision_proj.parameters())
            + list(self.text_proj.parameters())
            + list(self.itm_head.parameters())
        )
        new_decay, new_no_decay = split_decay(new_params)

        return [
            {"params": default_decay,    "lr": lr * 0.1, "weight_decay": weight_decay},
            {"params": default_no_decay, "lr": lr * 0.1, "weight_decay": 0.0},
            {"params": cross_decay,      "lr": lr,       "weight_decay": weight_decay},
            {"params": cross_no_decay,   "lr": lr,       "weight_decay": 0.0},
            {"params": new_decay,        "lr": lr,       "weight_decay": weight_decay},
            {"params": new_no_decay,     "lr": lr,       "weight_decay": 0.0},
            {"params": [self.temp],      "lr": lr,       "weight_decay": 0.0},
        ]

    def save_pretrained(self, save_directory):
        """Save QFormer checkpoint (downstream-compatible) + auxiliary training heads."""
        os.makedirs(save_directory, exist_ok=True)

        # Save QFormer in its own format (compatible with LM_2_VLM loading)
        self.qformer.save_pretrained(os.path.join(save_directory, "qformer"))

        # Save auxiliary heads (only needed if resuming training)
        torch.save(
            {
                "vision_proj": self.vision_proj.state_dict(),
                "text_proj": self.text_proj.state_dict(),
                "itm_head": self.itm_head.state_dict(),
                "temp": self.temp.data,
            },
            os.path.join(save_directory, "training_heads.pt"),
        )

    def load_training_heads(self, save_directory):
        """Load auxiliary training heads (for resuming training)."""
        heads_path = os.path.join(save_directory, "training_heads.pt")
        if os.path.exists(heads_path):
            checkpoint = torch.load(heads_path, map_location="cpu")
            self.vision_proj.load_state_dict(checkpoint["vision_proj"])
            self.text_proj.load_state_dict(checkpoint["text_proj"])
            self.itm_head.load_state_dict(checkpoint["itm_head"])
            self.temp.data = checkpoint["temp"]
