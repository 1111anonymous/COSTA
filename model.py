import torch
from torch import nn

import settings

device = settings.gpuId if torch.cuda.is_available() else 'cpu'


class CheckInEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]
        pop_num = vocab_size["pop"]
        dist_num = vocab_size["dist"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)
        self.pop_embed = nn.Embedding(pop_num + 1, self.embed_size, padding_idx=pop_num)
        self.dist_embed = nn.Embedding(dist_num + 1, self.embed_size, padding_idx=dist_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])
        pop_emb = self.pop_embed(x[5])
        dist_emb = self.dist_embed(x[6])

        if settings.extra_input == 'none' :
            return torch.cat((poi_emb, user_emb, hour_emb, day_emb), 1)
        elif (settings.extra_input == 'pop'):
            return torch.cat((poi_emb, user_emb, hour_emb, day_emb, pop_emb), 1)
        elif (settings.extra_input == 'dist') :
            return torch.cat((poi_emb, user_emb, hour_emb, day_emb, dist_emb), 1)
        elif (settings.extra_input == 'cat') :
            return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb, dist_emb), 1)
        elif settings.extra_input == 'all' :
            # return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb, pop_emb, dist_emb), 1)
            return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb, dist_emb), 1)
            


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for layer in self.layers:
            out = layer(out, out, out)

        return out


# Attention for query and key with different dimension
class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query) 
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)  # [len, 1]
        weight = torch.unsqueeze(weight, 1)
        temp2 = torch.mul(value, weight)
        out = torch.sum(temp2, 0)  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out
    
class Attention_pop(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query) 
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)  # [len, 1]
        temp2 = torch.mul(value, weight)
        out = temp2  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out


class model(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size=60,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        if settings.extra_input == 'none' :
            self.total_embed_size = f_embed_size * 4
        elif (settings.extra_input == 'pop') | (settings.extra_input == 'dist') | (settings.extra_input == 'cat') :
            self.total_embed_size = f_embed_size * 5
        elif settings.extra_input == 'all' :
            self.total_embed_size = f_embed_size * 6

        # Layers
        self.embedding = CheckInEmbedding(
            f_embed_size,
            vocab_size
        )
        self.encoder = TransformerEncoder(
            self.embedding,
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )
        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=num_lstm_layers,
            dropout=0
        )
        self.lstm_pop = nn.LSTM(
            input_size=f_embed_size,
            hidden_size=f_embed_size,
            num_layers=num_lstm_layers,
            dropout=0
        )
        self.final_attention = Attention(
            qdim=f_embed_size,
            kdim=self.total_embed_size
        )
        self.final_attention_pop = Attention_pop(
            qdim=f_embed_size,
            kdim=f_embed_size*2
        )

        self.loss_function = nn.CrossEntropyLoss()

        self.tryone_line2 = nn.Linear(self.total_embed_size, f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))
        self.predict_layer = nn.Linear(self.total_embed_size, f_embed_size)
        self.predict_layer_pop = nn.Linear(f_embed_size*2, f_embed_size)

        self.long_mean_layer = nn.Linear(self.total_embed_size, f_embed_size)
        self.long_sigma_layer = nn.Linear(self.total_embed_size, f_embed_size)
        self.short_mean_layer = nn.Linear(self.total_embed_size, f_embed_size)
        self.short_sigma_layer = nn.Linear(self.total_embed_size, f_embed_size)
        self.distribution_merge_mean = nn.Linear(f_embed_size*2, f_embed_size)
        self.distribution_merge_sigma = nn.Linear(f_embed_size*2, f_embed_size)
        self.distribution_merge = nn.Linear(f_embed_size*2, f_embed_size)

    def feature_mask(self, sequences, mask_prop):
        masked_sequences = []
        for seq in sequences:  # each long term sequences
            feature_seq, day_nums = seq[0], seq[1]
            seq_len = len(feature_seq[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index

            feature_seq[0, masked_index] = self.vocab_size["POI"]  # mask POI
            feature_seq[1, masked_index] = self.vocab_size["cat"]  # mask cat
            feature_seq[3, masked_index] = self.vocab_size["hour"]  # mask hour
            feature_seq[4, masked_index] = self.vocab_size["day"]  # mask day
            feature_seq[5, masked_index] = self.vocab_size["pop"] # mask pop
            feature_seq[6, masked_index] = self.vocab_size["dist"] # mask dist

            masked_sequences.append((feature_seq, day_nums))
        return masked_sequences

    def ssl(self, embedding_1, pos_embedding, neg_embedding):
        def score(x1, x2):
            return torch.mean(torch.matmul(x1, x2.transpose(0,1)))

        def single_infoNCE_loss_simple(embedding1, pos_embedding, neg_embedding):
            pos = score(embedding1, pos_embedding)
            neg = score(embedding1, neg_embedding)
            one = torch.FloatTensor([1]).to(device)
            if settings.cl_loss == 'log':
                con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            elif settings.cl_loss == 'bpr':
                con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos-neg)))
            return con_loss

        ssl_loss = single_infoNCE_loss_simple(embedding_1, pos_embedding, neg_embedding)
        return ssl_loss

    def forward(self, sample, pos_sample, neg_sample, cand):
        # Process input sample
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]
        short_term_features = short_term_sequence[0][:, :- 1]
        short_term_cat_seq = short_term_sequence[0][1, :- 1]
        short_term_dist_seq = short_term_sequence[0][-1, :- 1]
        target = short_term_sequence[0][0, -1]
        user_id = short_term_sequence[0][2, 0]

        # Random mask long-term sequences
        long_term_sequences = self.feature_mask(long_term_sequences, settings.mask_prop)

        # Long-term
        long_term_out = []
        for seq in long_term_sequences:
            output = self.encoder(feature_seq=seq[0])
            long_term_out.append(output)
        long_term_catted = torch.cat(long_term_out, dim=0)

        # Short-term
        short_term_state = self.encoder(feature_seq=short_term_features)

        # # pop or user embed cat with long_short_state from transformer
        h_all = torch.cat((short_term_state, long_term_catted)) # [length, total_embed_size]
        if settings.info_enhance == 'user':
            user_embed = self.embedding.user_embed(user_id)
            embedding = torch.unsqueeze(self.embedding(short_term_features), 0)
            output, _ = self.lstm(embedding)
            short_term_enhance = torch.squeeze(output)
            enhance_embed = self.enhance_val * user_embed + (1 - self.enhance_val) * self.tryone_line2(torch.mean(short_term_enhance, dim=0)) 
            final_att = self.final_attention(enhance_embed, h_all, h_all)
            final_pref = self.predict_layer(final_att) # [dim]
        elif settings.info_enhance == 'cat':
            cat_seq_embedding = torch.unsqueeze(self.embedding.cat_embed(short_term_cat_seq),0)
            output, _ = self.lstm_pop(cat_seq_embedding)
            short_term_enhance = torch.squeeze(output)
            enhance_embed = torch.mean(short_term_enhance, dim=0) # [dim]
            final_att = self.final_attention(enhance_embed, h_all, h_all)
            final_pref = self.predict_layer(final_att) # [dim]
        elif settings.info_enhance == 'dist':
            dist_seq_embedding = torch.unsqueeze(self.embedding.dist_embed(short_term_dist_seq),0)
            output, _ = self.lstm_pop(dist_seq_embedding)
            short_term_enhance = torch.squeeze(output)
            enhance_embed = torch.mean(short_term_enhance, dim=0) # [dim]
            final_att = self.final_attention(enhance_embed, h_all, h_all)
            final_pref = self.predict_layer(final_att) # [dim]
        elif settings.info_enhance == 'all': # cat and dist
            seq_embedding = torch.unsqueeze(self.embedding.cat_embed(short_term_cat_seq),0) + torch.unsqueeze(self.embedding.dist_embed(short_term_dist_seq),0)
            output, _ = self.lstm_pop(seq_embedding)
            short_term_enhance = torch.squeeze(output)
            enhance_embed = torch.mean(short_term_enhance, dim=0) # [dim]
            final_att = self.final_attention(enhance_embed, h_all, h_all)
            final_pref = self.predict_layer(final_att) # [dim]
        elif settings.info_enhance == 'none':
            final_att = torch.sum(h_all, 0)
            final_pref = self.predict_layer(final_att) # [dim]
        
        # Final predict
        if settings.cl_enhance == 'none':
            cand_rep = self.embedding.poi_embed(cand[0].long())
        elif settings.cl_enhance == 'cat':
            cand_rep = self.embedding.poi_embed(cand[0].long()) + self.embedding.cat_embed(cand[1].long())
        elif settings.cl_enhance == 'dist':
            cand_rep = self.embedding.poi_embed(cand[0].long()) + self.embedding.dist_embed(cand[2].long())
        elif settings.cl_enhance == 'all':
            cand_rep = self.embedding.poi_embed(cand[0].long()) + self.embedding.cat_embed(cand[1].long()) + self.embedding.dist_embed(cand[2].long())
        output = torch.matmul(final_pref.unsqueeze(0), cand_rep.transpose(0,1)).squeeze(0)

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)
        pred_loss = self.loss_function(pred, label)

        # contrastive learning
        if settings.cl_enhance == 'none':
            pos_embedding = self.embedding.poi_embed((pos_sample[0].long())) # [#sample, dim]
            neg_embedding = self.embedding.poi_embed((neg_sample[0].long())) # [#sample, dim]
        elif settings.cl_enhance == 'cat':
            pos_embedding = self.embedding.poi_embed((pos_sample[0].long())) + self.embedding.cat_embed((pos_sample[1].long())) # [#sample, dim]
            neg_embedding = self.embedding.poi_embed((neg_sample[0].long())) + self.embedding.cat_embed((neg_sample[1].long())) # [#sample, dim]
        elif settings.cl_enhance == 'dist':
            pos_embedding = self.embedding.poi_embed((pos_sample[0].long())) + self.embedding.dist_embed((pos_sample[2].long())) # [#sample, dim]
            neg_embedding = self.embedding.poi_embed((neg_sample[0].long())) + self.embedding.dist_embed((neg_sample[2].long())) # [#sample, dim]
        elif settings.cl_enhance == 'all':
            pos_embedding = self.embedding.poi_embed((pos_sample[0].long())) + self.embedding.cat_embed((pos_sample[1].long())) + self.embedding.dist_embed((pos_sample[2].long())) # [#sample, dim]
            neg_embedding = self.embedding.poi_embed((neg_sample[0].long())) + self.embedding.cat_embed((neg_sample[1].long())) + self.embedding.dist_embed((neg_sample[2].long())) # [#sample, dim]
        # print(pos_sample)
        ssl_loss = self.ssl(final_pref.unsqueeze(0), pos_embedding, neg_embedding)

        if torch.isnan(ssl_loss):
            loss = pred_loss
        else:
            loss = pred_loss + ssl_loss * settings.cl_weight
        return loss, output

    def predict(self, sample, pos_sample, neg_sample, cand):
        _, pred_raw = self.forward(sample, pos_sample, neg_sample, cand)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0, -1]

        return ranking, target
