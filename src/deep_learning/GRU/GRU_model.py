import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
from torch.nn.utils.rnn import pad_sequence
warnings.filterwarnings('ignore')
from sklearn.preprocessing import scale
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GRU_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU_Classifier, self).__init__()
        self.em = nn.Embedding(num_embeddings=27, embedding_dim=128)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x1):
        x1 = self.em(x1)
        x1 = x1.to(torch.float32)
        out, _ = self.gru(x1)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sig(out)
        out = out.squeeze(1)
        return out

def d_loadar(x1,x2,y):
    X = []
    Y = []
    Z = []
    for x,y,z in zip(x1,x2,y):
        X.append(x)
        Y.append(y)
        Z.append(z)
        if len(X) == 8:
            o_x = X
            o_y = Y
            o_z = Z
            X = []
            Y = []
            Z = []

            numpy_array1 = np.array([item.cpu().detach().numpy() for item in o_x])
            numpy_array2 = np.array([item.cpu().detach().numpy() for item in o_y])
            numpy_array3 = np.array([item.cpu().detach().numpy() for item in o_z])

            x_res = torch.tensor(numpy_array1)
            y_res = torch.tensor(numpy_array2)
            z_res = torch.tensor(numpy_array3)



            yield (x_res,y_res,z_res)


def create_dataset_number(PATH_x,PATH_y):
    df = pd.read_csv(PATH_x)


    df_num = df.drop(columns=['Year', 'CycPeptMPDB_ID', 'Structurally_Unique_ID'
        , 'SMILES', 'Sequence', 'Sequence_LogP', 'Sequence_TPSA', 'Source',
                              'Original_Name_in_Source_Literature', 'HELM',
                              'HELM_URL', 'Molecule_Shape','Permeability','PAMPA']).values



    df[df['Permeability'] >= -6] = 1
    df[df['Permeability'] < -6] = 0
    y = df['Permeability'].values



    y = y.astype('float32')
    df_num = scale(df_num)
    return df_num,y


def create_dataset_list(PATH_x):
    df_list = pd.read_csv(PATH_x,usecols=['Sequence_LogP','Sequence_TPSA'])
    df_list['Sequence_LogP'] = df_list['Sequence_LogP'].apply(lambda x: eval(x))
    df_list['Sequence_TPSA'] = df_list['Sequence_TPSA'].apply(lambda x: eval(x))
    a = df_list['Sequence_LogP'].values
    b = df_list['Sequence_TPSA'].values
    max_len = max(len(x) for x in a)
    data_padded = np.zeros((len(a), max_len))
    for i, row in enumerate(a):
        data_padded[i, :len(row)] = row
    tensor_data = torch.tensor(data_padded, dtype=torch.float32)
    logp_list = torch.tensor(pad_sequence(tensor_data, batch_first=True, padding_value=0))
    data_padded = np.zeros((len(b), max_len))
    for i, row in enumerate(b):
        data_padded[i, :len(row)] = row
    tensor_data = torch.tensor(data_padded, dtype=torch.float32)
    tpsa_list = torch.tensor(pad_sequence(tensor_data, batch_first=True, padding_value=0))
    list_num = torch.cat([logp_list, tpsa_list], dim=1)
    list_num = scale(list_num)
    list_num = torch.tensor(list_num, dtype=torch.float32)
    return list_num

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def create_dataset_seq(PATH_x):
    df = pd.read_csv(PATH_x,usecols=['SMILES'])
    vocab = []
    datas = []
    for i, row in df.iterrows():
        data = row["SMILES"]
        tokens = smi_tokenizer(data).split(" ")
        if len(tokens) <= 128:
            di = tokens+["PAD"]*(128-len(tokens))
        else:
            di = tokens[:128]
        datas.append(di)
        vocab.extend(tokens)
    vocab = list(set(vocab))
    vocab = ["PAD"]+vocab
    with open("vocab.txt","w",encoding="utf8") as f:
        for i in vocab:
            f.write(i)
            f.write("\n")
    mlist = []
    word2id = {}
    for i,d in enumerate(vocab):
        word2id[d] = i
    for d_i in datas:
        mi = [word2id[d] for d in d_i]
        mlist.append(np.array(mi))
    return mlist
