import os
import torch.nn as nn
from models import Model_TGCN, ConfigTrans, GCNWithAttention,Siamese_network
from data_pretreatment import func
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import dgl
from dgllife.utils import *

torch.cuda.empty_cache()
config = ConfigTrans()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_data(df, node_featurizer):
    mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
    g = [mol_to_complete_graph(m, node_featurizer=node_featurizer) for m in mols]

    df[df['Permeability'] >= -6] = 1
    df[df['Permeability'] < -6] = 0

    y = np.array(list((df['Permeability'])))

    y = np.array(y, dtype=np.int64)
    return g, y


def collate(sample):
    _, list_num, graphs, labels, index = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return _, list_num, batched_graph, torch.tensor(labels), index


def train_test_val(dataloader, gcn_net, model_Siamese, model_tgcn):
    epoch_loss, epoch_acc = 0, 0
    mlist = []
    gcn_net.eval()
    model_Siamese.eval()
    model_tgcn.eval()
    for i, (X, list_num, graph, labels, index) in enumerate(dataloader):
        graph = graph.to(device)
        labels = labels.to(device)
        atom_feats = graph.ndata.pop('h').to(device)
        atom_feats, labels = atom_feats.to(device), labels.to(device)

        pred = gcn_net(graph, atom_feats)

        X = torch.cat(X, dim=0)
        X = torch.reshape(X, [config.batch_size, config.batch_size])
        X = X.to(device)
        list_num = torch.tensor([item.cpu().detach().numpy() for item in list_num]).cuda()
        list_num = torch.tensor(list_num)
        list_num = list_num.to(device)
        y = model_Siamese(X)
        y = model_tgcn(y, pred, list_num).to(device)
        y = torch.reshape(y, [config.batch_size])
        loss = nn.BCELoss()(y, labels.float())
        epoch_loss += loss.detach().item()

        true_label = labels.cpu().numpy()
        yy = [1 if m >= 0.5 else 0 for m in y.cpu().detach().numpy()]

        mlist.extend(yy)

        epoch_acc += sum(true_label == yy)
    epoch_acc = epoch_acc / true_label.shape[0]
    epoch_acc /= (i + 1)
    epoch_loss /= (i + 1)

    return epoch_acc, epoch_loss, mlist


if __name__ == '__main__':
    for num in range(1, 11):
        torch.cuda.empty_cache()
        PATH_x_train = '../../../data/data_splitClassifier/X_train{}.csv'.format(num)
        PATH_x_test = '../../../data/data_splitClassifier/X_test{}.csv'.format(num)
        PATH_x_val = '../../../data/data_splitClassifier/X_val{}.csv'.format(num)
        model_Siamese = Siamese_network().to(device)
        df_seq_train, y_train_tensor, y_true_train, list_num_train = func(PATH_x_train)
        df_seq_test, y_test_tensor, y_true_test, list_num_test = func(PATH_x_test)
        df_seq_val, y_val_tensor, y_true_val, list_num_val = func(PATH_x_val)

        node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
        n_feats = atom_featurizer.feat_size('feat')

        gcn_net = GCNWithAttention(in_feats=74, hidden_feats=[60, 20], num_heads=[4, 4]).to(device)
        model_tgcn = Model_TGCN().to(device)
        train_X = pd.read_csv(PATH_x_train)

        x_train, y_train = get_data(train_X, node_featurizer)
        train_data = list(zip(df_seq_train, list_num_train, x_train, y_train, [i for i in range(len(train_X))]))
        train_loader_ = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate,
                                   drop_last=True)

        test_X = pd.read_csv(PATH_x_test)
        x_test, y_test = get_data(test_X, node_featurizer)
        test_data = list(zip(df_seq_test, list_num_test, x_test, y_test, [i for i in range(len(test_X))]))
        test_loader_test = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate,
                                      drop_last=True)

        val_X = pd.read_csv(PATH_x_val)
        x_val, y_val = get_data(val_X, node_featurizer)
        val_data = list(zip(df_seq_val, list_num_val, x_val, y_val, [i for i in range(len(val_X))]))
        val_loader_val = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate,
                                    drop_last=True)

        optimizer = torch.optim.Adam([{'params': gcn_net.parameters()},
                                      {'params': model_Siamese.parameters()},
                                      {'params': model_tgcn.parameters()}],
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        for epoch in range(1, 501):
            gcn_net.train()
            model_Siamese.train()
            model_tgcn.train()
            train_epoch_loss, train_epoch_acc, train_epoch_r2 = 0, 0, 0
            for i, (X, list_num, graph, labels, index) in enumerate(train_loader_):
                graph = graph.to(device)
                train_labels = labels.to(device)
                atom_feats = graph.ndata.pop('h').to(device)
                atom_feats, train_labels = atom_feats.to(device), train_labels.to(device)
                train_pred = gcn_net(graph, atom_feats)
                X = torch.cat(X, dim=0)
                X = torch.reshape(X, [config.batch_size, config.batch_size])
                X = X.to(device)

                list_num = torch.tensor([item.cpu().detach().numpy() for item in list_num])
                list_num = torch.tensor(list_num)
                list_num = list_num.to(device)
                y = model_Siamese(X)
                y = model_tgcn(y, train_pred, list_num).to(device)
                y = torch.reshape(y, [config.batch_size])

                train_loss = nn.BCELoss()(y, train_labels.float())
                optimizer.zero_grad()
                train_loss.requires_grad_(True)
                train_loss.backward()
                optimizer.step()
                train_epoch_loss += train_loss.detach().item()

                train_true_label = train_labels.to('cpu').numpy()
                yy = [1 if i >= 0.5 else 0 for i in y.cpu().detach().numpy()]
                train_epoch_acc += sum(train_true_label == yy)

            train_epoch_acc = train_epoch_acc / train_true_label.shape[0]
            train_epoch_acc /= (i + 1)  # 求的是平均正确率
            train_epoch_loss /= (i + 1)  # 求的是平均损失率

            test_epoch_acc, test_epoch_loss, test_list = train_test_val(test_loader_test, gcn_net,
                                                                        model_Siamese, model_tgcn)
            val_epoch_acc, val_epoch_loss, val_list = train_test_val(val_loader_val, gcn_net,
                                                                     model_Siamese, model_tgcn)

            print(f"epoch: {epoch}, train_LOSS: {train_epoch_loss:.3f}, train_ACC: {train_epoch_acc:.3f}")
            print(f"epoch: {epoch}, test_LOSS : {test_epoch_loss:.3f} , test_ACC : {test_epoch_acc:.3f}")
            print(f"epoch: {epoch}, val_LOSS  : {val_epoch_loss:.3f}  , val_ACC  : {val_epoch_acc:.3f}")

            y_true_test = pd.read_csv(PATH_x_test, usecols=['Permeability'])
            y_true_test[y_true_test['Permeability'] >= -6] = 1
            y_true_test[y_true_test['Permeability'] < -6] = 0
            y_true_test = y_true_test[:688]
            t1, t2 = pd.DataFrame(test_list, columns=['predict']), pd.DataFrame(y_true_test['Permeability'].values,
                                                                                columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data_origin/MSF-CPMP/{num}/test/', exist_ok=True)
            pd.DataFrame(tt).to_csv(
                './pred_data_origin/MSF-CPMP/{}/test/experiment_{}_predicted_test_values.csv'.format(num, epoch),
                index=False)


            y_true_val = pd.read_csv(PATH_x_val, usecols=['Permeability'])
            y_true_val[y_true_val['Permeability'] >= -6] = 1
            y_true_val[y_true_val['Permeability'] < -6] = 0
            y_true_val = y_true_val[:624]
            t1, t2 = pd.DataFrame(val_list, columns=['predict']), pd.DataFrame(y_true_val['Permeability'].values,
                                                                               columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data_origin/MSF-CPMP/{num}/val/', exist_ok=True)
            pd.DataFrame(tt).to_csv(
                './pred_data_origin/MSF-CPMP/{}/val/experiment_{}_predicted_valid_values.csv'.format(num, epoch),
                index=False)

            fd_w = open("./pred_data_origin/result_train.txt", "a+")
            fd_t = open("./pred_data_origin/result_test.txt", "a+")
            fd_v = open("./pred_data_origin/result_val.txt", "a+")
            fd_w.write(f"epoch{epoch}   : train_loss{train_epoch_loss:.4f}, train_acc:{train_epoch_acc:.4f}\n")
            fd_t.write(f"epoch{epoch}   : test_loss{test_epoch_loss:.4f}  , test_acc :{test_epoch_acc:.4f}\n")
            fd_v.write(f"epoch{epoch}   : val_loss{val_epoch_loss:.4f}    , val_acc  :{val_epoch_acc:.4f}\n")

            fd_w.close()
            fd_t.close()
            fd_v.close()