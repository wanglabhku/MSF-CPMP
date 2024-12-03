import torch
from torch import nn
import torch.nn.functional as F
import copy
from dgllife.model import GATPredictor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ConfigTrans(object):

    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 1
        self.num_epochs = 100
        self.batch_size = 128
        self.pad_size = 1
        self.learning_rate = 0.001
        self.embed = 128
        self.dim_model = 128
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 8
        self.num_encoder = 6
        self.feature_dim = 105
        self.weight_decay = 0.05


config = ConfigTrans()


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)]).to(device)
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(config.batch_size * self.num_head, -1, self.dim_head)
        K = K.view(config.batch_size * self.num_head, -1, self.dim_head)
        V = V.view(config.batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
        context = context.view(config.batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

# transformer
class Transformer_test(nn.Module):
    def __init__(self):
        super(Transformer_test, self).__init__()
        self.position_embedding = Positional_Encoding(config.embed, config.pad_size)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(config.num_encoder)])

    def forward(self, x):
        out = self.position_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        return out

class Siamese_network(nn.Module):
    def __init__(self):
        super(Siamese_network,self).__init__()
        self.transformer1 = Transformer_test().to(device)
        self.transformer2 = Transformer_test().to(device)
        self.max_pooling1 = nn.MaxPool1d(kernel_size=2)
        self.max_pooling2 = nn.MaxPool1d(kernel_size=2)
    def forward(self,x):
        out1 = self.transformer1(x)
        out1 = self.max_pooling1(out1)
        out2 = self.transformer1(x)
        out2 = self.max_pooling1(out2)
        out = torch.mul(out1,out2)
        return out


class Model_TGCN(nn.Module):
    def __init__(self):
        super(Model_TGCN, self).__init__()
        self.num_fc = nn.Linear(103, 128)
        self.norm1 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(2 + 128, 1)
        self.norm2 = nn.BatchNorm1d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x_t, x_g, x_l):
        num_out = self.num_fc(x_l)
        num_out = self.norm1(num_out)
        p = torch.concat([x_t,x_g,num_out],dim = -1)
        out = self.fc1(p)
        out = self.norm2(out)
        out = self.sig(out)
        return out


class GlobalAttention(nn.Module):
    def __init__(self, em_dim, seq_len, reduction_ratio=4):
        super(GlobalAttention, self).__init__()
        self.Norm = nn.LayerNorm([em_dim, seq_len])
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GMP = nn.AdaptiveMaxPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(em_dim, em_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(em_dim // reduction_ratio, em_dim),
            nn.LayerNorm([em_dim])
        )
        self.Sigmoid = nn.Sigmoid()
        self.OUT = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.Norm(x)
        x = x.unsqueeze(0)
        avg_out = self.GAP(x).squeeze(-1)
        max_out = self.GMP(x).squeeze(-1)
        avg_out = self.FC(avg_out).unsqueeze(2)
        max_out = self.FC(max_out).unsqueeze(2)
        out = avg_out + max_out
        attention_weights = self.Sigmoid(out)
        return self.OUT(attention_weights * x)


class GCNWithAttention(nn.Module):
    def __init__(self, in_feats, hidden_feats,num_heads):
        super(GCNWithAttention, self).__init__()
        self.attention = GATPredictor(in_feats=in_feats,
                                      hidden_feats=hidden_feats,
                                      n_tasks=2,
                                      num_heads=num_heads,
                                      predictor_dropout=config.dropout,
                                      feat_drops=[config.dropout, config.dropout])

    def forward(self, g, feats):
        h_att = self.attention(g, feats)
        return h_att

if __name__ == '__main__':


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    model_Siamese = Siamese_network()
    gcn_net = GCNWithAttention(in_feats=74, hidden_feats=[60, 20], num_heads=[4, 4])
    model_tgcn = Model_TGCN()


    num_params1 = count_parameters(model_Siamese)
    num_params2 = count_parameters(gcn_net)
    num_params3 = count_parameters(model_tgcn)
    All_params = num_params1 + num_params2 + num_params3
    print(f"Total number of parameters: {All_params}")

    memory_size_bytes = All_params * 4
    memory_size_megabytes = memory_size_bytes / (1024 ** 2)

    print(f"Approximate memory usage: {memory_size_megabytes:.6f} MB")
