import sys
sys.path.append( 'D:/pycharm_pro/gan_enhancer/' )
# In[ ]:
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
from model import get_model, get_model_onehot, get_model_2, model5
import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score, f1_score,recall_score,matthews_corrcoef,confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot


# class roc_callback(Callback):
#     def __init__(self, val_data,name):
#         self.en = val_data[0]
#         self.y = val_data[1]
#         self.name = name
#
#     def on_train_begin(self, logs={}):
#         return
#
#     def on_train_end(self, logs={}):
#         return
#
#     def on_epoch_begin(self, epoch, logs={}):
#         return
#
#     def on_epoch_end(self, epoch, logs={}):
#         # y_pred1 = self.model.predict([self.en])
#         # y_pred = np.where(y_pred1 > 0.5, 1, 0)
#         # sn_val = recall_score(self.y, y_pred)
#         # tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
#         # sp_val = tn / (tn + fp)
#         # mcc_val = matthews_corrcoef(self.y, y_pred)
#         # auc_val = roc_auc_score(self.y, y_pred1)
#         # aupr_val = average_precision_score(self.y, y_pred1)
#         # f1_val = f1_score(self.y, np.round(y_pred1.reshape(-1)))
#         self.model.save_weights(
#             "./model/our_model/%sModel%d.tf" % (self.name, epoch))
#         # print('\r sn_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
#         # print('\r sp_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
#         # print('\r mcc_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
#         # print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
#         # print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
#         # print('\r f1_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
#         return
#
#     def on_batch_begin(self, batch, logs={}):
#         return
#
#     def on_batch_end(self, batch, logs={}):
#         return






t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
batch_size = 32
names = ['first']
name=names[0]
Data_dir = 'D:/pycharm_pro/gan_enhancer/%s/first_index/' % name
train = np.load(Data_dir + '%s_train_enhancers3.npz' % name)
test = np.load(Data_dir + '%s_test_enhancers3.npz' % name)
X_en_tra, y_tra = train['X_en_tra'], train['y_tra']
X_en_tes, y_tes = test['X_en_tes'],test['y_tes']
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# acc_score = []
# auc_score = []
# sn_score = []
# sp_score = []
# mcc_score = []
for i,(tra, val) in enumerate(kfold.split(X_en_tra, y_tra)):
    print('\n\n第%d折' % i)
    # X_en_tra, X_en_val,y_tra, y_val=train_test_split(
    #     X_en_tra,y_tra,test_size=0.3,stratify=y_tra,random_state=5)
    model=None
    model=model5()
    model.summary()
    print ('Traing %s cell line specific model ...'%name)
    # back = roc_callback(val_data=[X_en_tra, y_tra], name=name)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')
    filepath = 'D:/pycharm_pro/gan_enhancer/first/model/our_model_3_test/%sModel%d.tf' % (name,i)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
    callbacks_list = [checkpoint]
    back = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto')
    # history=model.fit(X_en_tra[tra], y_tra[tra], validation_data=(X_en_tra[val], y_tra[val]), epochs=90, batch_size=batch_size,
    #                   callbacks=[callbacks_list, back])
    history=model.fit(X_en_tra, y_tra, validation_data=(X_en_tes, y_tes), epochs=90, batch_size=batch_size,
                      callbacks=[callbacks_list, back])
    acc = history.history['val_accuracy']
    loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    pyplot.title('Model%d.tf' %(i) )
    pyplot.plot(epochs, acc, 'red', label='Validation acc')
    pyplot.plot(epochs, loss, 'blue', label='Validation loss')
    pyplot.legend()
    pyplot.show()
#     prd_acc = model.predict(X_en_tra[val])
#     pre_acc2 = []
#     for i in prd_acc:
#         pre_acc2.append(i[0])
#
#     prd_lable = []
#     for i in pre_acc2:
#         if i > 0.5:
#             prd_lable.append(1)
#         else:
#             prd_lable.append(0)
#     prd_lable = np.array(prd_lable)
#     obj = confusion_matrix(y_tra[val], prd_lable)
#     tp = obj[0][0]
#     fn = obj[0][1]
#     fp = obj[1][0]
#     tn = obj[1][1]
#     sn = tp / (tp + fn)
#     sp = tn / (tn + fp)
#     mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
#     sn_score.append(sn)
#     sp_score.append(sp)
#     mcc_score.append(mcc)
#     ###########################
#     pre_test_y = model.predict(X_en_tra[val], batch_size=batch_size)
#     test_auc = roc_auc_score(y_tra[val], pre_test_y)
#     auc_score.append(test_auc)
#     print("test_auc: ", test_auc)
#
#     score, acc = model.evaluate(X_en_tra[val], y_tra[val], batch_size=batch_size)
#     acc_score.append(acc)
#     print('val score:', score)
#     print('val accuracy:', acc)
#     print('***********************************************************************\n')
# print('***********************print final result*****************************')
# print(acc_score, auc_score)
# mean_acc = np.mean(acc_score)
# mean_auc = np.mean(auc_score)
# mean_sn = np.mean(sn_score)
# mean_sp = np.mean(sp_score)
# mean_mcc = np.mean(mcc_score)
# # print('mean acc:%f\tmean auc:%f'%(mean_acc,mean_auc))
#
# line = 'acc\tsn\tsp\tmcc\tauc:\n%.2f\t%.2f\t%.2f\t%.4f\t%.4f' % (
# 100 * mean_acc, 100 * mean_sn, 100 * mean_sp, mean_mcc, mean_auc)
# print('5-fold result:\n' + line)
t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("开始时间:"+t1+"结束时间："+t2)