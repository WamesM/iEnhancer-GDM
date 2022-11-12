import sys
sys.path.append( 'D:/pycharm_pro/gan_enhancer/' )
from model import get_model, get_model_onehot, model5, model5_onehot
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve, precision_recall_curve
import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import matplotlib.pyplot as plt
# import umap
# import umap.plot
# from matplotlib.backends.backend_pdf import PdfPages


names = ['first']
for name in names:
    for i in [1]:
        model = model5_onehot()
        model.load_weights("./model/our_model_onehot/%sModel%s.tf" % (name, i))
        Data_dir = 'D:/pycharm_pro/gan_enhancer/%s/first_index/' % name
        test = np.load(Data_dir+'%s_test_onehot.npz' % name)
        X_en_tes,  y_tes = test['X_en_tes'], test['y_tes']

        print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
        # model.fit(X_en_tes, y_tes)
        y_pred1 = model.predict([X_en_tes])
        # print(y_pred1)
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
