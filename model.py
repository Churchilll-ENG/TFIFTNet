import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewLSTM(nn.Module):
    def __init__(self, hidden_size=256):
        super(MultiViewLSTM, self).__init__()

        self.lstm_seq = nn.LSTM(input_size=768, hidden_size=hidden_size, batch_first=True)
        self.lstm_feat = nn.LSTM(input_size=224, hidden_size=hidden_size, batch_first=True)
        self.lstm_group = nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True)

        self.proj_seq = nn.Linear(hidden_size, 768)
        self.proj_feat = nn.Linear(hidden_size, 768)
        self.proj_group = nn.Linear(hidden_size, 768)

        self.compress = nn.Linear(4 * 224, 449)  # 将4 * 224压缩到449

    def forward(self, x):
        B = x.size(0)

        x_seq = x.view(B, 4 * 224, 768)
        lstm_seq_out, _ = self.lstm_seq(x_seq)
        seq_out = self.proj_seq(lstm_seq_out)
        seq_out = seq_out.view(B, 4, 224, 768)

        x_feat = x.permute(0, 1, 3, 2).contiguous().view(B, 4 * 768, 224)
        lstm_feat_out, _ = self.lstm_feat(x_feat)
        feat_out = self.proj_feat(lstm_feat_out)
        feat_out = feat_out.view(B, 4, 768, 224).permute(0, 1, 3, 2).contiguous()

        x_group = x.permute(0, 2, 3, 1).contiguous().view(B, 224 * 768, 4)
        lstm_group_out, _ = self.lstm_group(x_group)
        group_out = self.proj_group(lstm_group_out)
        group_out = group_out.view(B, 224, 768, 4).permute(0, 3, 1, 2).contiguous()

        combined = (seq_out + feat_out + group_out) / 3.0  
        combined = combined.view(B, 4 * 224, 768)
        compressed = self.compress(combined.transpose(1, 2)).transpose(1, 2)

        return compressed


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.LSTM_head = MultiViewLSTM
        self.head_dim = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.w_q_f = nn.Linear(d_model, d_model)
        self.w_k_f = nn.Linear(d_model, d_model)
        self.w_v_f = nn.Linear(d_model, d_model)
        self.w_o_f = nn.Linear(d_model, d_model)

        self.w_o_tf = nn.Linear(d_model, d_model)
        self.w_o_ft = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.lstm_seq = nn.LSTM(input_size=768, hidden_size=256, batch_first=True)
        self.lstm_feat = nn.LSTM(input_size=224, hidden_size=256, batch_first=True)
        self.lstm_group = nn.LSTM(input_size=768*224, hidden_size=256, batch_first=True)

        self.proj_seq = nn.Linear(256, 768)
        self.proj_feat = nn.Linear(256, 224)
        self.proj_group = nn.Linear(256, 224*768)

        self.compress = nn.Linear(4 * 224, 449) 

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        half_len = seq_len // 2
        x = query
        first_step = x[:, 0:1, :]
        x_t = x[:, 1:half_len+1, :]  # [batch_size, seq_len/2, d_model]
        x_f = x[:, half_len+1:, :]  # [batch_size, seq_len/2, d_model]

        Q_t = self.w_q(x_t).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K_t = self.w_k(x_t).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V_t = self.w_v(x_t).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores_t = torch.matmul(Q_t, K_t.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            attn_scores_t = attn_scores_t.masked_fill(mask == 0, float('-inf'))

        attn_probs_t = F.softmax(attn_scores_t, dim=-1)
        attn_probs_t = self.dropout(attn_probs_t)

        context_t = torch.matmul(attn_probs_t, V_t)
        context_t = context_t.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_t = self.w_o(context_t)

        Q_f = self.w_q_f(x_f).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K_f = self.w_k_f(x_f).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V_f = self.w_v_f(x_f).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        attn_scores_f = torch.matmul(Q_f, K_f.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attn_scores_f = attn_scores_f.masked_fill(mask == 0, float('-inf'))
        attn_probs_f = F.softmax(attn_scores_f, dim=-1)
        attn_probs_f = self.dropout(attn_probs_f)
        context_f = torch.matmul(attn_probs_f, V_f)
        context_f = context_f.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_f = self.w_o_f(context_f)

        attn_scores_tf = torch.matmul(Q_t, K_f.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attn_scores_tf = attn_scores_tf.masked_fill(mask == 0, float('-inf'))
        attn_probs_tf = F.softmax(attn_scores_tf, dim=-1)
        attn_probs_tf = self.dropout(attn_probs_tf)
        context_tf = torch.matmul(attn_probs_tf, V_f)
        context_tf = context_tf.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_tf = self.w_o_tf(context_tf)

        attn_scores_ft = torch.matmul(Q_f, K_t.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attn_scores_ft = attn_scores_ft.masked_fill(mask == 0, float('-inf'))
        attn_probs_ft = F.softmax(attn_scores_ft, dim=-1)
        attn_probs_ft = self.dropout(attn_probs_ft)
        context_ft = torch.matmul(attn_probs_ft, V_t)
        context_ft = context_ft.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_ft = self.w_o_ft(context_ft)

       concat_features = torch.stack([output_f, output_t, output_tf, output_ft], dim=1)
       B = concat_features.size(0)

       concat_features_seq = concat_features.view(B, 4 * 224, 768)
       lstm_seq_out, _ = self.lstm_seq(concat_features_seq)
       seq_out = self.proj_seq(lstm_seq_out)
       seq_out = seq_out.view(B, 4, 224, 768)

       concat_features_feat = concat_features.permute(0, 1, 3, 2).reshape(B, 4 * 768, 224).contiguous()
       lstm_feat_out, _ = self.lstm_feat(concat_features_feat.contiguous())
       feat_out = self.proj_feat(lstm_feat_out)
       feat_out = feat_out.view(B, 4, 768, 224).permute(0, 1, 3, 2)

       concat_features_group = concat_features.reshape(B, 4, 224 * 768).contiguous()
       lstm_group_out, _ = self.lstm_group(concat_features_group.contiguous())
       group_out = self.proj_group(lstm_group_out)
       group_out = group_out.view(B, 224, 768, 4).permute(0, 3, 1, 2)

       combined = (seq_out + feat_out + group_out) / 3.0 
       combined = combined.view(B, 4 * 224, 768)
       compressed = self.compress(combined.transpose(1, 2)).transpose(1, 2)
       # compressed = torch.cat((output_f + output_f, output_f + output_f, first_step), dim=1)

        return compressed


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        ffn_output = self.ffn(src)
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


