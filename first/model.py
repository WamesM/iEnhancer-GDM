# import numpy as np
# A = np.load('data_train.npz')
# print(A['X_en_tra'])
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import os
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras
# from layers import MultiHeadAttention, Attention
import numpy as np
from keras_multi_head import MultiHeadAttention


MAX_LEN_en = 200
# NB_WORDS = 16385
NB_WORDS = 65
# NB_WORDS = 257
# NB_WORDS = 1025
# NB_WORDS = 4097
# NB_WORDS = 16385
# NB_WORDS = 65537
EMBEDDING_DIM = 100
embedding_matrix = np.load('D:/pycharm_pro/My-Enhancer-classification/embedding/embedding_matrix3.npy')
embedding_matrix_one_hot = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])


class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weight = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        #print("WQ.shape", WQ.shape)

        #print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        QK = K.softmax(QK)

        #print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def model5():
    enhancers = Input(shape=(MAX_LEN_en,))
    emb_en = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       embedding_matrix], trainable=True)(enhancers)
    # emb_en = Embedding(5, 4, weights=[embedding_matrix_one_hot],
    #                               trainable=False)(enhancers)#cpu
    enhancer_conv_layer1 = Convolution1D(
                                        filters=128,
                                        kernel_size=5,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer1 = MaxPooling1D(pool_size=int(4))
    enhancer_conv_layer2 = Convolution1D(
                                        filters=64,
                                        kernel_size=5,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer2 = MaxPooling1D(pool_size=int(2))
    enhancer_conv_layer3 = Convolution1D(
                                        filters=128,
                                        kernel_size=7,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer3 = MaxPooling1D(pool_size=int(4))
    enhancer_conv_layer4 = Convolution1D(
                                        filters=64,
                                        kernel_size=7,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer4 = MaxPooling1D(pool_size=int(2))
    enhancer_conv_layer5 = Convolution1D(
                                        filters=128,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer5 = MaxPooling1D(pool_size=int(4))
    enhancer_conv_layer6 = Convolution1D(
                                        filters=64,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer6 = MaxPooling1D(pool_size=int(2))
    # enhancer_conv_layer7 = Convolution1D(
    #                                     filters=16,
    #                                     kernel_size=3,
    #                                     padding="same",  # "same"
    #                                     )
    # enhancer_max_pool_layer7 = MaxPooling1D(pool_size=int(4))


    enhancer = Sequential()
    enhancer.add(enhancer_conv_layer1)
    enhancer.add(Activation("relu"))
    enhancer.add(enhancer_max_pool_layer1)
    enhancer.add(BatchNormalization())
    enhancer.add(Dropout(0.2))
    enhancer.add(enhancer_conv_layer2)
    enhancer.add(Activation("relu"))
    enhancer.add(enhancer_max_pool_layer2)
    enhancer.add(BatchNormalization())
    enhancer.add(Dropout(0.2))
    enhancer_out = enhancer(emb_en)

    enhancer1 = Sequential()
    enhancer1.add(enhancer_conv_layer3)
    enhancer1.add(Activation("relu"))
    enhancer1.add(enhancer_max_pool_layer3)
    enhancer1.add(BatchNormalization())
    enhancer1.add(Dropout(0.2))
    enhancer1.add(enhancer_conv_layer4)
    enhancer1.add(Activation("relu"))
    enhancer1.add(enhancer_max_pool_layer4)
    enhancer1.add(BatchNormalization())
    enhancer1.add(Dropout(0.2))
    enhancer_out1 = enhancer1(emb_en)

    enhancer2 = Sequential()
    enhancer2.add(enhancer_conv_layer5)
    enhancer2.add(Activation("relu"))
    enhancer2.add(enhancer_max_pool_layer5)
    enhancer2.add(BatchNormalization())
    enhancer2.add(Dropout(0.2))
    enhancer2.add(enhancer_conv_layer6)
    enhancer2.add(Activation("relu"))
    enhancer2.add(enhancer_max_pool_layer6)
    enhancer2.add(BatchNormalization())
    enhancer2.add(Dropout(0.2))
    enhancer_out2 = enhancer2(emb_en)

    # enhancer3 = Sequential()
    # enhancer3.add(enhancer_conv_layer7)
    # enhancer3.add(Activation("relu"))
    # enhancer3.add(enhancer_max_pool_layer7)
    # enhancer3.add(BatchNormalization())
    # enhancer3.add(Dropout(0.5))
    # enhancer_out3 = enhancer3(emb_en)




    enhancer_out4 = tf.concat([enhancer_out, enhancer_out1, enhancer_out2],axis = 2)





    # l_gru_1 = Bidirectional(GRU(64, return_sequences=True))(enhancer_out)
    # l_att = AttLayer(64)(l_gru_1)
    # bn = BatchNormalization()(l_att_1)
    # dt = Dropout(0.2)(bn)

    l_gru = Bidirectional(LSTM(32, return_sequences=True))(enhancer_out4)
    # l_att = Self_Attention(32)(l_gru)
    l_att = MultiHeadAttention(head_num=64, name='Multi-Head-Attention')(l_gru)
    l_att = Flatten()(l_att)
    #多头和自注意力需要展开层
    # l_att = AttLayer(64)(l_gru)
    # l_gru = Bidirectional(SimpleRNN(64, return_sequences=True))(enhancer_out)
    # l_att = AttLayer(64)(l_gru)
    bn2 = BatchNormalization()(l_att)
    dt2 = Dropout(0.2)(bn2)
    # dt = BatchNormalization()(dt2)
    # dt = Dropout(0.5)(dt)
    dt = Dense(64,kernel_initializer="glorot_uniform")(dt2)
    # dt = GlobalAveragePooling1D()(dt2)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([enhancers], preds)
    adam = tensorflow.keras.optimizers.Adam(lr=4e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model
    # 5e-6(测试集)  4e-5(训练集)

def model5_onehot():
    enhancers = Input(shape=(MAX_LEN_en,))
    # emb_en = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
    #                    embedding_matrix], trainable=True)(enhancers)
    emb_en = Embedding(5, 4, weights=[embedding_matrix_one_hot],
                                  trainable=False)(enhancers)#cpu
    enhancer_conv_layer1 = Convolution1D(
                                        filters=128,
                                        kernel_size=5,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer1 = MaxPooling1D(pool_size=int(4))
    enhancer_conv_layer2 = Convolution1D(
                                        filters=64,
                                        kernel_size=5,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer2 = MaxPooling1D(pool_size=int(2))
    enhancer_conv_layer3 = Convolution1D(
                                        filters=128,
                                        kernel_size=7,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer3 = MaxPooling1D(pool_size=int(4))
    enhancer_conv_layer4 = Convolution1D(
                                        filters=64,
                                        kernel_size=7,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer4 = MaxPooling1D(pool_size=int(2))
    enhancer_conv_layer5 = Convolution1D(
                                        filters=128,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer5 = MaxPooling1D(pool_size=int(4))
    enhancer_conv_layer6 = Convolution1D(
                                        filters=64,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer6 = MaxPooling1D(pool_size=int(2))
    # enhancer_conv_layer7 = Convolution1D(
    #                                     filters=16,
    #                                     kernel_size=3,
    #                                     padding="same",  # "same"
    #                                     )
    # enhancer_max_pool_layer7 = MaxPooling1D(pool_size=int(4))


    enhancer = Sequential()
    enhancer.add(enhancer_conv_layer1)
    enhancer.add(Activation("relu"))
    enhancer.add(enhancer_max_pool_layer1)
    enhancer.add(BatchNormalization())
    enhancer.add(Dropout(0.2))
    enhancer.add(enhancer_conv_layer2)
    enhancer.add(Activation("relu"))
    enhancer.add(enhancer_max_pool_layer2)
    enhancer.add(BatchNormalization())
    enhancer.add(Dropout(0.2))
    enhancer_out = enhancer(emb_en)

    enhancer1 = Sequential()
    enhancer1.add(enhancer_conv_layer3)
    enhancer1.add(Activation("relu"))
    enhancer1.add(enhancer_max_pool_layer3)
    enhancer1.add(BatchNormalization())
    enhancer1.add(Dropout(0.2))
    enhancer1.add(enhancer_conv_layer4)
    enhancer1.add(Activation("relu"))
    enhancer1.add(enhancer_max_pool_layer4)
    enhancer1.add(BatchNormalization())
    enhancer1.add(Dropout(0.2))
    enhancer_out1 = enhancer1(emb_en)

    enhancer2 = Sequential()
    enhancer2.add(enhancer_conv_layer5)
    enhancer2.add(Activation("relu"))
    enhancer2.add(enhancer_max_pool_layer5)
    enhancer2.add(BatchNormalization())
    enhancer2.add(Dropout(0.2))
    enhancer2.add(enhancer_conv_layer6)
    enhancer2.add(Activation("relu"))
    enhancer2.add(enhancer_max_pool_layer6)
    enhancer2.add(BatchNormalization())
    enhancer2.add(Dropout(0.2))
    enhancer_out2 = enhancer2(emb_en)

    # enhancer3 = Sequential()
    # enhancer3.add(enhancer_conv_layer7)
    # enhancer3.add(Activation("relu"))
    # enhancer3.add(enhancer_max_pool_layer7)
    # enhancer3.add(BatchNormalization())
    # enhancer3.add(Dropout(0.5))
    # enhancer_out3 = enhancer3(emb_en)




    enhancer_out4 = tf.concat([enhancer_out, enhancer_out1, enhancer_out2],axis = 2)





    # l_gru_1 = Bidirectional(GRU(64, return_sequences=True))(enhancer_out)
    # l_att = AttLayer(64)(l_gru_1)
    # bn = BatchNormalization()(l_att_1)
    # dt = Dropout(0.2)(bn)

    l_gru = Bidirectional(LSTM(32, return_sequences=True))(enhancer_out4)
    # l_att = Self_Attention(32)(l_gru)
    l_att = MultiHeadAttention(head_num=64, name='Multi-Head-Attention')(l_gru)
    l_att = Flatten()(l_att)
    #多头和自注意力需要展开层
    # l_att = AttLayer(64)(l_gru)
    # l_gru = Bidirectional(SimpleRNN(64, return_sequences=True))(enhancer_out)
    # l_att = AttLayer(64)(l_gru)
    bn2 = BatchNormalization()(l_att)
    dt2 = Dropout(0.2)(bn2)
    # dt = BatchNormalization()(dt2)
    # dt = Dropout(0.5)(dt)
    dt = Dense(64,kernel_initializer="glorot_uniform")(dt2)
    # dt = GlobalAveragePooling1D()(dt2)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([enhancers], preds)
    adam = tensorflow.keras.optimizers.Adam(lr=4e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model
    # 5e-6(测试集)  4e-5(训练集)

