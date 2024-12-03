import os
import torch
import torch.nn as nn
from MLP_process import MLP
import pandas as pd
from MLP_process import create_dataset_seq,create_dataset_number, d_loadar
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(3407)
# Define hyperparameter
batch_size = 64
learning_rate = 0.0001
epochs = 30
input_size = 128
hidden_size = 32
num_layers = 2
output_size = 103


def func(x, y):
    df_num, y_true = create_dataset_number(x, y)
    df_seq = create_dataset_seq(x)
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)
    y = torch.tensor([item for item in y_true]).to(torch.float)
    return df_seq, y, y_true

if __name__ == '__main__':
    for i in range(1, 11):
        PATH_x_train = '../../../data/data_splitClassifier/X_train{}.csv'.format(i)
        PATH_x_val = '../../../data/data_splitClassifier/X_val{}.csv'.format(i)
        PATH_x_test = '../../../data/data_splitClassifier/X_test{}.csv'.format(i)

        df_seq_train, y_train, y_true_train = func(PATH_x_train, PATH_x_train)
        df_seq_val, y_val, y_true_val = func(PATH_x_val, PATH_x_val)
        df_seq_test, y_test, y_true_test = func(PATH_x_test, PATH_x_test)


        model = MLP(input_size,output_size)
        criterion = nn.BCELoss()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            loss_avg = 0
            step = 0
            total = 0
            corrent = 0
            for X, y1 in d_loadar(df_seq_train, y_train):
                step += 1

                optimizer.zero_grad()
                X = X.to(torch.float32)
                y_pred = model(X)
                loss = criterion(y_pred, y1)
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                y_pred = torch.round(y_pred)
                total += y1.size(0)
                corrent += (y_pred == y1.float()).sum().item()
            train_epoch_acc = corrent / total#训练的平均正确率
            train_epoch_loss = loss_avg / step#训练的平均损失率

            def train_test_val(seq,y):
                model.eval()
                loss_avg =0
                step =0
                total = 0
                corrent = 0
                mlist1 = []
                mlist2 = []
                with torch.no_grad():
                    for X, y1 in d_loadar(seq, y):
                        step += 1
                        X = X.to(torch.float32)
                        y_pred = model(X)
                        loss = criterion(y_pred,y1)
                        loss_avg += loss.item()
                        mlist2.extend(y_pred.cpu().detach().numpy())
                        y_pred = torch.round(y_pred)
                        mlist1.extend(y_pred.cpu().detach().numpy())
                        total += y1.size(0)
                        corrent += (y_pred == y1.float()).sum().item()
                acc = corrent / total#平均正确率
                loss_avg = loss_avg / step#平均损失率

                return acc, loss_avg, mlist1, mlist2

            # os.makedirs(f'./model/cnn_fc/{i}/', exist_ok=True)
            #
            # torch.save(model, './model/cnn_fc/{}/{}_cnn_fc_model.pt'.format(i, epoch))
            val_epoch_acc, val_epoch_loss, val_list1, val_list2 = train_test_val(df_seq_val, y_val, )
            test_epoch_acc, test_epoch_loss, test_list1, test_list2 = train_test_val(df_seq_test, y_test, )


            print(f"epoch: {epoch}, train_LOSS: {train_epoch_loss:.3f},train_ACC: {train_epoch_acc:.3f}")
            print(f"epoch: {epoch}, test_LOSS : {test_epoch_loss:.3f},test_ACC: {test_epoch_acc:.3f}")
            print(f"epoch: {epoch}, val_LOSS  : {val_epoch_loss:.3f},val_ACC: {val_epoch_acc:.3f}")

            t1, t2 = pd.DataFrame(test_list1, columns=['predict']), pd.DataFrame(
                y_true_test[:688], columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data/MLP/test1/{i}/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data/MLP/test1/{}/experiment_{}_predicted_values.csv'.format(i, epoch + 1),
                                    index=False)

            t1, t2 = pd.DataFrame(test_list2, columns=['predict']), pd.DataFrame(
                y_true_test[:688],
                columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data/MLP/test2/{i}/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data/MLP/test2/{}/experiment_{}_predicted_values.csv'.format(i, epoch + 1),
                                    index=False)

            fd_w = open("./pred_data/result_train.txt", "a+")
            fd_t = open("./pred_data/result_test.txt", "a+")
            fd_v = open("./pred_data/result_val.txt", "a+")

            fd_w.write(f"epoch{epoch + 1}   :train_loss{train_epoch_loss:.4f}, train_acc:{train_epoch_acc:.4f}\n")
            fd_t.write(f"epoch{epoch + 1}   :test_loss{test_epoch_loss:.4f}, test_acc   :{test_epoch_acc:.4f}\n")
            fd_v.write(f"epoch{epoch + 1}   :val_loss{val_epoch_loss:.4f}, val_acc      :{val_epoch_acc:.4f}\n")

            fd_w.close()
            fd_t.close()
            fd_v.close()

