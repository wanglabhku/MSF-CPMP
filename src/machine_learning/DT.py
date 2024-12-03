from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd
from tqdm import tqdm
import time
import os

average_epoch_acc = 0

if __name__ == '__main__':
    fd_t = open("./pred_data/dt/result_test.txt", "a+")
    for i in tqdm(range(1,11),desc="Processing epochs"):
        time.sleep(0.1)
        print('\n')
        epoch_acc = 0
        X_train = pd.read_csv('../../data/data_splitClassifier_235/X_train{}.csv'.format(i))
        y_train = pd.read_csv('../../data/data_splitClassifier_235/y_train{}.csv'.format(i)).to_numpy().reshape(-1)
        test_x = pd.read_csv('../../data/data_splitClassifier_235/X_test{}.csv'.format(i))
        test_y = pd.read_csv('../../data/data_splitClassifier_235/y_test{}.csv'.format(i))

        test_y = test_y['target'].to_numpy()

        numtest = (test_y.shape[0] // 16) * 16

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train.astype('int'))


        y_pred = clf.predict(test_x[:numtest])

        epoch_acc += sum(y_pred == (test_y[:numtest]))
        epoch_acc = (epoch_acc/y_pred.shape[0]) * 100
        average_epoch_acc += epoch_acc
        print(f"epoch: {i},epoch_acc: {epoch_acc:.3f}")


        t1,t2 = pd.DataFrame(y_pred,columns=['predict']),pd.DataFrame(test_y[:numtest],columns=['true'])
        tt = pd.concat([t1, t2], axis=1)

        os.makedirs(f'./pred_data/dt/', exist_ok=True)
        pd.DataFrame(tt).to_csv('./pred_data/dt/experiment_{}_predicted_values.csv'.format(i))

        os.makedirs(f'./pred_model/dt_model/', exist_ok=True)
        with open('./pred_model/dt_model/{}_DT_model_rbf.pkl'.format(i), 'wb') as f:
            pickle.dump(clf, f)
        fd_t.write(f"epoch:{i}, epoch_acc:{epoch_acc:.4f}\n")
    fd_t.write(f"average_epoch_acc: {average_epoch_acc / i:.4f}\n")
    fd_t.close()

    print(f"average_epoch_acc: {average_epoch_acc / i:.3f}")