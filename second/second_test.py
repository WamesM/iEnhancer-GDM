import sys
sys.path.append( 'D:/pycharm_pro/gan_enhancer/second/' )
from model import get_model, get_model_onehot, model5, model5_onehot
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,matthews_corrcoef,confusion_matrix, roc_curve
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model



names = ['second']
for name in names:
    for i in [2]:
        model = model5()
        model.load_weights("./model/our_model_3/%sModel%s.tf" % (name, i))
        Data_dir = 'D:/pycharm_pro/gan_enhancer/%s/second_index/' % name
        test = np.load(Data_dir+'%s_test_enhancers3.npz' % name)
        X_en_tes,  y_tes = test['X_en_tes'], test['y_tes']

        print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
        # model.fit(X_en_tes, y_tes)
        y_pred1 = model.predict([X_en_tes])
        y_pred = np.where(y_pred1 > 0.5, 1, 0)

        acc = accuracy_score(y_tes, y_pred)
        sn = recall_score(y_tes, y_pred)
        mcc = matthews_corrcoef(y_tes, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_tes, y_pred).ravel()
        sp = tn / (tn + fp)
        auc = roc_auc_score(y_tes, y_pred1)
        aupr = average_precision_score(y_tes, y_pred1)
        f1 = f1_score(y_tes, np.round(y_pred1.reshape(-1)))
        print("ACC : ", acc)
        print("SN : ", sn)
        print("SP : ", sp)
        print("MCC : ", mcc)
        print("AUC : ", auc)
        print("AUPR : ", aupr)
        print("f1_score : ", f1)



