from sklearn.naive_bayes import GaussianNB
import pickle
import pandas as pd
from tqdm import tqdm

import time
import os

average_epoch_acc = 0

if __name__ == '__main__':

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

        clf = GaussianNB()
        clf.fit(X_train, y_train.astype('int'))

        y_pred = clf.predict_proba(test_x[:numtest])
        y_pred = y_pred[:,1]

        yy = [1 if i >= 0.5 else 0 for i in y_pred]

        epoch_acc += sum(y_pred == (test_y[:numtest]))
        epoch_acc = (epoch_acc/y_pred.shape[0]) * 100
        average_epoch_acc += epoch_acc
        print(f"epoch: {i},epoch_acc: {epoch_acc:.3f}")

        t1,t2 = pd.DataFrame(y_pred,columns=['predict']),pd.DataFrame(test_y[:numtest],columns=['true'])
        tt = pd.concat([t1, t2], axis=1)

        os.makedirs(f'./pred_data/GaussianNB/', exist_ok=True)
        pd.DataFrame(tt).to_csv('./pred_data/GaussianNB/experiment_{}_predicted_values.csv'.format(i))

        os.makedirs(f'./pred_model/GaussianNB_model/', exist_ok=True)
        with open('./pred_model/GaussianNB_model/{}_GaussianNB_model_rbf.pkl'.format(i), 'wb') as f:
            pickle.dump(clf, f)

    print(f"average_epoch_acc: {average_epoch_acc / i:.3f}")