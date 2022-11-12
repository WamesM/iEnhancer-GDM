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

for i,(tra, val) in enumerate(kfold.split(X_en_tra, y_tra)):
    print('\n\n第%d折' % i)

    model=None
    model=model5()
    model.summary()
    print ('Traing %s cell line specific model ...'%name)

    filepath = 'D:/pycharm_pro/gan_enhancer/first/model/our_model_3_test/%sModel%d.tf' % (name,i)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
    callbacks_list = [checkpoint]
    back = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto')

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

t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("开始时间:"+t1+"结束时间："+t2)