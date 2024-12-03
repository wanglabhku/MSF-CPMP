import os
import torch
import numpy as np
import torch.nn as nn
from lstm_model import LSTM
import pandas as pd
from lstm_model import create_dataset_seq,create_dataset_number,d_loadar
input_size = 128
hidden_size = 64
num_layers = 5
output_size = 1
learning_rate = 0.0001
num_epochs = 30
def func(x,y):
    df_num, y_true = create_dataset_number(x, y)
    list_num = torch.tensor(df_num, dtype=torch.float32)
    df_seq = create_dataset_seq(x)
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)
    y = torch.tensor([item for item in y_true]).to(torch.float)

    return df_seq,list_num,y,y_true

if __name__ == '__main__':
    for i in range(1,11):
        PATH_x_train = '../../../data/data_splitClassifier/X_train{}.csv'.format(i)
        PATH_x_test = '../../../data/data_splitClassifier/X_test{}.csv'.format(i)
        PATH_x_val = '../../../data/data_splitClassifier/X_val{}.csv'.format(i)

        df_seq_train,list_num_train,y_train,y_true_train = func(PATH_x_train,PATH_x_train)
        df_seq_test,list_num_test,y_test,y_true_test = func(PATH_x_test,PATH_x_test)
        df_seq_val,list_num_val,y_val,y_true_val = func(PATH_x_val,PATH_x_val)


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            model.train()
            loss_avg = 0
            step = 0
            total = 0
            train_epoch_loss, train_epoch_acc, train_epoch_r2 = 0, 0, 0
            for X, x2, y1 in d_loadar(df_seq_train,list_num_train,y_train):
                step += 1
                X = X.to(device)
                x2 = x2.to(device)
                y1 = y1.to(device)
                optimizer.zero_grad()
                x2 = x2.to(torch.float32)
                y_pred = model(X, x2)
                loss = criterion(y_pred, y1)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                train_true_label = y1.to('cpu').numpy()
                train_true_label = np.ravel(train_true_label)
                yy = [1 if i >= 0.5 else 0 for i in y_pred.detach().to('cpu').numpy()]
                train_epoch_acc += sum(train_true_label == yy)
            train_epoch_acc = train_epoch_acc / train_true_label.shape[0]
            train_epoch_acc /= step#计算平均正确率
            train_epoch_loss = loss_avg / step#计算平均损失率

            # os.makedirs(f'./model/lstm_fc/{i}/', exist_ok=True)
            # torch.save(model, './model/lstm_fc/{}/第{}个lstm_model.pt'.format(i,epoch))

            def train_test_val(seq,num,y):
                model.eval()
                loss_avg = 0
                step = 0
                total = 0
                corrent = 0
                mlist1 = []
                mlist2 = []
                with torch.no_grad():
                    for X, x2, y in d_loadar(seq,num,y):
                        step += 1
                        X = X.to(device)
                        x2 = x2.to(device)
                        y = y.to(device)
                        y_pred = model(X, x2)
                        loss = criterion(y_pred,y)
                        loss_avg += loss.item()
                        mlist2.extend(y_pred.cpu().detach().numpy())
                        y_pred = torch.round(y_pred)
                        mlist1.extend(y_pred.cpu().detach().numpy())
                        total += y.size(0)
                        corrent += (y_pred == y.float()).sum().item()

                # pred = model(seq.to(device), num.to(device))
                # acc = 100 * corrent / total
                acc = corrent / total  # 平均正确率
                loss_avg = loss_avg / step  # 平均损失率
                return acc, loss_avg, mlist1, mlist2


            val_epoch_acc, val_epoch_loss, val_list1, val_list2 = train_test_val(df_seq_val,list_num_val,y_val)
            test_epoch_acc, test_epoch_loss, test_list1, test_list2 = train_test_val(df_seq_test,list_num_test,y_test)

            print(f"epoch: {epoch+1}, train_LOSS: {train_epoch_loss:.4f},train_ACC: {train_epoch_acc:.4f}")
            print(f"epoch: {epoch+1}, test_LOSS : {test_epoch_loss:.4f},test_ACC: {test_epoch_acc:.4f}")
            print(f"epoch: {epoch+1}, val_LOSS  : {val_epoch_loss:.4f},val_ACC: {val_epoch_acc:.4f}")

            y_true_test = pd.read_csv(PATH_x_test, usecols=['Permeability'])
            y_true_test[y_true_test['Permeability'] >= -6] = 1
            y_true_test[y_true_test['Permeability'] < -6] = 0

            t1, t2 = pd.DataFrame(test_list1, columns=['predict']), pd.DataFrame(
                y_true_test['Permeability'].values[:688], columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data/LSTM/test1/{i}/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data/LSTM/test1/{}/experiment_{}_predicted_values.csv'.format(i, epoch + 1),
                                    index=False)

            t1, t2 = pd.DataFrame(test_list2, columns=['predict']), pd.DataFrame(
                y_true_test['Permeability'].values[:688],
                columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data/LSTM/test2/{i}/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data/LSTM/test2/{}/experiment_{}_predicted_values.csv'.format(i, epoch + 1),
                                    index=False)

            fd_w = open("./pred_data/result_train.txt", "a+")
            fd_t = open("./pred_data/result_test.txt", "a+")
            fd_v = open("./pred_data/result_val.txt", "a+")

            fd_w.write(f"epoch{epoch+1}   :train_loss{train_epoch_loss:.4f}, train_acc:{train_epoch_acc:.4f}\n")
            fd_t.write(f"epoch{epoch+1}   :test_loss{test_epoch_loss:.4f}, test_acc   :{test_epoch_acc:.4f}\n")
            fd_v.write(f"epoch{epoch+1}   :val_loss{val_epoch_loss:.4f}, val_acc      :{val_epoch_acc:.4f}\n")

            fd_w.close()
            fd_t.close()
            fd_v.close()
