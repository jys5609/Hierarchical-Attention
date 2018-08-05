from en_data_generator import EnDataGenerator
from cn_data_generator import CnDataGenerator
from theano import tensor as T
from keras.preprocessing import sequence as ksq
from keras.models import Model
from keras.layers import *
from utils import *
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import *
from keras.layers.wrappers import *
import keras.optimizers

np.random.seed(1)

class Tracker():
    def __init__(self, embed_dim, en_segment_length, cn_segment_length):
        self.exist_model = False

        self.en_generator = EnDataGenerator()
        self.cn_generator = CnDataGenerator()

        slots, slot_loc, slot_value, slot_value_vector = self.en_generator.get_general()

        self.index = -1

        self.slots = slots
        self.slot_loc = slot_loc
        self.slot_value = slot_value
        self.slot_value_vector = slot_value_vector
        self.slot_num = len(slots)

        self.embed_dim = embed_dim
        self.en_segment_length = en_segment_length
        self.cn_segment_length = cn_segment_length

        value_num = [0 for i in range(len(slots))]
        for topic in sorted(slot_loc.keys()):
            for slot in sorted(slot_loc[topic].keys()):
                value_num[slot_loc[topic][slot]] = len(slot_value[topic][slot])

        self.value_num = value_num

        value_vec_list = []
        value_list = []
        for topic in sorted(slot_loc.keys()):
            for slot in sorted(slot_loc[topic].keys()):
                vec_list = []

                for i in range(len(slot_value_vector[topic][slot])):
                    value_vec = slot_value_vector[topic][slot][i]
                    value = slot_value[topic][slot][i]
                    vec_list.append(value_vec)
                    value_list.append(value)
                vec_list = np.array(vec_list)  # (#value, 200)
                vec_list = np.reshape(vec_list, (1, vec_list.shape[0], vec_list.shape[1]))  # (1, #value, 200)
                vec_list = np.reshape(vec_list, (vec_list.shape[2], vec_list.shape[1], 1))  # (200, #value, 1)
                vec_list = np.reshape(vec_list, (1, vec_list.shape[1], vec_list.shape[0]))  # (256, #value, 200)
                value_vec_list.append(vec_list)

        self.value_vec_list = value_vec_list
        self.value_list = value_list

        self.exist_model = False

    def load_data(self, datatype, uttr_accumulate):
        SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot, X_len = self.en_generator.get_data(datatype, uttr_accumulate)

        X = ksq.pad_sequences(X, maxlen=self.en_segment_length, dtype='float32')
        X_ins = ksq.pad_sequences(X_ins, maxlen=self.en_segment_length, dtype='float32')
        Y = np.array(Y)

        X_len = np.array(X_len)
        X_len = X_len.reshape((X_len.shape[0], 1))

        X_slot = np.array(X_slot)
        X_slot = np.reshape(X_slot, (X_slot.shape[0], X_slot.shape[1], 1))
        X_slot = np.tile(X_slot, self.embed_dim)

        return SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot, X_len

    def cn_load_data(self, datatype, uttr_accumulate):
        SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_len = self.cn_generator.get_data(datatype, uttr_accumulate)

        X = ksq.pad_sequences(X, maxlen=self.cn_segment_length, dtype='float32')
        X_ins = ksq.pad_sequences(X_ins, maxlen=self.cn_segment_length, dtype='float32')
        Y = np.array(Y)

        X_len = np.array(X_len)
        X_len = X_len.reshape((X_len.shape[0], 1))

        return SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_len

    def lambda_normalize(self, x):
        norm = K.sum(K.square(x), axis=-1, keepdims=True)
        norm = K.switch(norm <= 0, np.array(1.0), norm)
        x = x / K.sqrt(norm)
        return x

    def lambda_normalize_out_shape(self, input_shape):
        return input_shape

    def lambda_tile_embed(self, x):
        return T.tile(x, self.embed_dim)

    def lambda_tile_embed_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(input_shape) == 3
        shape[-1] *= self.embed_dim
        return tuple(shape)

    def lambda_mean(self, x):
        x, word_num = x
        return K.sum(x, axis=1, keepdims=True) * 1.0 / K.expand_dims(word_num, axis=2)

    def lambda_mean_output_shape(self, input_shape):
        input_shape, input_word_shape = input_shape
        shape = list(input_shape)
        assert len(input_shape) == 3
        return tuple([shape[0], 1, shape[2]])

    def lambda_mul(self, i):
        x, y = i
        return x * y

    def lambda_mul_output_shape(self, input_shape):
        return input_shape[1]

    def lambda_sum_keepdims(self, x):
        return K.sum(x, axis=1, keepdims=True)

    def lambda_sum_keepdims_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(input_shape) == 3
        return tuple([shape[0], 1, shape[2]])

    def lambda_sum(self, x):
        return K.sum(x, axis=-1)

    def lambda_sum_output_shape(self, input_shape):
        return input_shape[:-1]

    def lambda_broadcast(self, x):
        self.index += 1
        x = T.tile(x, self.embed_dim)
        return x * T.patternbroadcast(K.variable(self.value_vec_list[self.index]), [True, False, False])

    def lambda_broadcast_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] *= self.embed_dim
        return tuple(shape)

    def cosine_sim(self, y_pred, y_true):
        pred_norm = T.sum(T.square(y_pred), axis=2, keepdims=True)
        pred_norm = T.switch(pred_norm <= 0, 1.0, pred_norm)
        y_pred = y_pred / T.sqrt(pred_norm)
        true_norm = T.sum(T.square(y_true), axis=2, keepdims=True)
        true_norm = T.switch(true_norm <= 0, 1.0, true_norm)
        y_true = y_true / T.sqrt(true_norm)
        return -K.mean(K.sum((y_true * y_pred), axis=2))

    def model1(self, input_x, input_x_ins, lstm_units, dropout, segment_length):
        x = concatenate([input_x, input_x_ins], axis=-1)
        x = Masking()(x)
        x = Dropout(dropout)(x)
        x = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(x)
        x = Dropout(dropout)(x)

        # Attention to given utterance words
        attention = [TimeDistributed(Dense(1))(x) for i in range(self.slot_num)]  # (None, 520, 1)
        attention = [MaskEatingLambda(lambda_mask_zero)(attention[i]) for i in range(self.slot_num)]
        attention = [Reshape((segment_length,))(attention[i]) for i in range(self.slot_num)]  # (None, 520)
        attention = [Masking()(attention[i]) for i in range(self.slot_num)]
        attention = [Activation('softmax')(attention[i]) for i in range(self.slot_num)]  # (None, 520)
        attention = [MaskEatingLambda(lambda_mask_zero)(attention[i]) for i in range(self.slot_num)]
        attention = [Reshape((segment_length, 1,))(attention[i]) for i in range(self.slot_num)]  # (None, 520, 1) * 30
        attention = [Masking()(attention[i]) for i in range(self.slot_num)]

        # Weigted-sum output based on attention
        vector_list = [Lambda(self.lambda_tile_embed, output_shape=self.lambda_tile_embed_output_shape)(attention[i]) for i in range(self.slot_num)]
        vector_list = [multiply([vector_list[i], input_x]) for i in range(self.slot_num)]
        vector_list = [Lambda(self.lambda_sum_keepdims, output_shape=self.lambda_sum_keepdims_output_shape)(vector_list[i]) for i in range(self.slot_num)]
        vector_list = [Reshape((self.embed_dim,))(vector_list[i]) for i in range(self.slot_num)]
        vector = [Reshape((1, self.embed_dim,))(vector_list[i]) for i in range(self.slot_num)]
        vector = concatenate(vector, axis=1)  # (None, 30, 200)

        # Normalize for each output vector
        vector = Lambda(self.lambda_normalize, output_shape=self.lambda_normalize_out_shape)(vector)  # (None, 30, 200)

        # For minimizing cosine similarity
        out_prob = concatenate(attention, axis=-1, name='out_prob')  # (None, 520, 30)

        return x, vector, vector_list, attention, out_prob

    def model2(self, input_en, input_en_ins, input_cn, input_cn_ins, lstm_units, dropout):
        _, _, en_vector_list, _, _ = self.model1(input_en, input_en_ins, lstm_units, dropout, self.en_segment_length)
        _, _, cn_vector_list, _, _ = self.model1(input_cn, input_cn_ins, lstm_units, dropout, self.cn_segment_length)

        # Attention to ontology values
        value_attention = [concatenate([en_vector_list[i], cn_vector_list[i]], axis=-1) for i in range(self.slot_num)]
        value_attention = [Dense(100)(value_attention[i]) for i in range(self.slot_num)]
        value_attention = [Dense(self.value_num[i])(value_attention[i]) for i in range(self.slot_num)]
        value_attention = [Activation('softmax')(value_attention[i]) for i in range(self.slot_num)]
        value_attention = [Reshape((self.value_num[i], 1))(value_attention[i]) for i in range(self.slot_num)]

        value_entropy = concatenate(value_attention, axis=1, name='value_entropy')  # (None, #total_value, 1)

        # Weigted-sum output based on attention
        value_out = [Lambda(self.lambda_broadcast, output_shape=self.lambda_broadcast_output_shape)(value_attention[i]) for i in
                     range(self.slot_num)]  # (None, #value, 200) * 30
        value_out = [Lambda(self.lambda_sum_keepdims, output_shape=self.lambda_sum_keepdims_output_shape)(value_out[i]) for i in
                     range(self.slot_num)]  # (None, 1, 200) * 30
        value_out = concatenate(value_out, axis=1)  # (None, 30, 200)

        # Normalize for each output vector
        value_out = Lambda(self.lambda_normalize, output_shape=self.lambda_normalize_out_shape, name='value_out')(
            value_out)  # (None, 30, 200)

        return value_out, value_attention, value_entropy

    def model3(self, input_en, input_en_ins, input_word_num, lstm_units, dropout):
        x, vector, vector_list, attention, out_prob = self.model1(input_en, input_en_ins, lstm_units, dropout, self.en_segment_length)

        vector = Reshape((self.slot_num, self.embed_dim, 1))(vector)  # (None, 30, 200, 1)

        sentinel = [TimeDistributed(Dense(self.embed_dim))(x) for i in range(self.slot_num)]  # (None, 520, 200) * 30
        sentinel = [MaskEatingLambda(lambda_mask_zero)(sentinel[i]) for i in range(self.slot_num)]  # (None, 520, 200)
        sentinel = [Lambda(self.lambda_mean, output_shape=self.lambda_mean_output_shape)([sentinel[i], input_word_num])
                    for i in range(self.slot_num)]  # (None, 200) * 30
        sentinel = [Reshape((1, self.embed_dim,))(sentinel[i]) for i in range(self.slot_num)]  # (None, 1, 200) * 30
        sentinel = concatenate(sentinel, axis=1)  # (None, 30, 200)

        sentinel = Lambda(self.lambda_normalize, output_shape=self.lambda_normalize_out_shape)(
            sentinel)  # (None, 30, 200)

        sentinel = Reshape((self.slot_num, self.embed_dim, 1))(sentinel)  # (None, 30, 200, 1)

        vector_concat = concatenate([vector, sentinel], axis=-1)  # (None, 30, 200, 2)

        # Attention weight between Model1 and sentinel
        sentinel_attention = Dropout(dropout)(out_prob)
        sentinel_attention = LSTM(100, return_sequences=False)(sentinel_attention)  # (None, 100)
        sentinel_attention = [Dense(50)(sentinel_attention) for i in range(self.slot_num)]  # (None, 50) * 30
        sentinel_attention = [Dense(2)(sentinel_attention[i]) for i in range(self.slot_num)]  # (None, 2) * 30
        sentinel_attention = [Activation('softmax')(sentinel_attention[i]) for i in range(self.slot_num)]  # (None, 2) * 30
        sentinel_attention = [Reshape((1, 2))(sentinel_attention[i]) for i in range(self.slot_num)]  # (None, 1, 2) * 30
        sentinel_attention = concatenate(sentinel_attention, axis=1)  # (None, 30, 2)
        sentinel_attention = Reshape((self.slot_num, 1, 2), name='attention')(sentinel_attention)  # (None, 30, 1, 2)

        # final output vector with weighted sum
        out_embed = Lambda(self.lambda_mul, output_shape=self.lambda_mul_output_shape)(
            [sentinel_attention, vector_concat])  # (None, 30, 200, 2)
        out_embed = Lambda(self.lambda_sum, output_shape=self.lambda_sum_output_shape)(out_embed)  # (None, 30, 200)
        out_embed = Lambda(self.lambda_normalize, output_shape=self.lambda_normalize_out_shape, name='out_embed')(
            out_embed)  # (None, 30, 200)

        return out_embed, out_prob

    def model4(self, input_en, input_en_ins, input_x_slot, input_word_num, input_cn, input_cn_ins, lstm_units, dropout):
        out_embed, out_prob = self.model3(input_en, input_en_ins, input_word_num, lstm_units, dropout) # v_2

        value_out, value_attention, value_entropy = self.model2(input_en, input_en_ins, input_cn, input_cn_ins, lstm_units, dropout) # v_1

        model_attention = concatenate(value_attention, axis=1) # (None, #total_value, 1)
        model_attention = Dropout(dropout)(model_attention)
        model_attention = LSTM(100, return_sequences=False)(model_attention) # (None, 100)
        model_attention = [Dense(50)(model_attention) for i in range(self.slot_num)] # (None, 50) * 30
        model_attention = [Dense(2)(model_attention[i]) for i in range(self.slot_num)] # (None, 2) * 30
        model_attention = [Activation('softmax')(model_attention[i]) for i in range(self.slot_num)] # (None, 2) * 30
        model_attention = [Reshape((1,2))(model_attention[i]) for i in range(self.slot_num)] # (None, 1, 2) * 30
        model_attention = concatenate(model_attention, axis=1) # (None, 30, 2)
        model_attention = Reshape((self.slot_num, 1, 2), name='model_attention')(model_attention) # (None, 30, 1, 2)

        # final output vector with weighted sum
        out_embed = Reshape((self.slot_num, self.embed_dim, 1))(out_embed) # (None, 30, 200, 1)
        value_out = Reshape((self.slot_num, self.embed_dim, 1))(value_out) # (None, 30, 200, 1)
        out_concat = concatenate([out_embed, value_out], axis=-1) # (None, 30, 200, 2)

        final_out_embed = Lambda(self.lambda_mul, output_shape=self.lambda_mul_output_shape)([model_attention, out_concat]) # (None, 30, 200, 2)
        final_out_embed = Lambda(self.lambda_sum, output_shape=self.lambda_sum_output_shape)(final_out_embed) # (None, 30, 200)

        final_out_embed = multiply([final_out_embed, input_x_slot], name='final_out_embed')

        return final_out_embed, out_prob, value_entropy

    def create_model(self, learning_rate, lstm_units, dropout):
        # Inputs
        input_en = Input(shape=(self.en_segment_length, self.embed_dim,), dtype='float32')
        input_en_ins = Input(shape=(self.en_segment_length, self.slot_num + 7))
        input_x_slot = Input(shape=(self.slot_num, self.embed_dim), dtype='float32')
        input_word_num = Input(shape=(1,))

        input_cn = Input(shape=(self.cn_segment_length, self.embed_dim,), dtype='float32')
        input_cn_ins = Input(shape=(self.cn_segment_length, 2 * self.slot_num + 7))

        final_out_embed, out_prob, value_entropy = self.model4(input_en, input_en_ins, input_x_slot, input_word_num, input_cn, input_cn_ins, lstm_units, dropout)

        self.model = Model(inputs=[input_en, input_en_ins, input_x_slot, input_word_num, input_cn, input_cn_ins],
                           outputs=[final_out_embed, out_prob, value_entropy])

        optim = keras.optimizers.Adam(lr=learning_rate)

        self.model.compile(
            loss={'final_out_embed': self.cosine_sim, 'out_prob': 'mse', 'value_entropy': 'mse'}, optimizer=optim, metrics=[],
            loss_weights={'final_out_embed': 1.0, 'out_prob': 0.0, 'value_entropy': 0.0})
        self.model.summary()
        self.exist_model = True

    def train(self, X, X_ins, X_slot, X_len, cn_X, cn_X_ins, Y,
              valid_X, valid_X_ins, valid_X_slot, valid_X_len, valid_cn_X, valid_cn_X_ins, valid_Y, nb_epoch,
              weight_file):
        assert self.exist_model

        train_temp_prob = np.zeros((X.shape[0], self.en_segment_length, self.slot_num))
        valid_temp_prob = np.zeros((valid_X.shape[0], self.en_segment_length, self.slot_num))

        train_value_entropy = np.zeros((X.shape[0], sum(self.value_num), 1))
        valid_value_entropy = np.zeros((valid_X.shape[0], sum(self.value_num), 1))

        print('X dim: ', X.shape)
        print('X_ins dim: ', X_ins.shape)
        print('X_slot dim: ', X_slot.shape)
        print('X_len dim: ', X_len.shape)
        print('Y_dim: ', Y.shape)
        print()

        print('valid_X dim: ', valid_X.shape)
        print('valid_X_ins dim: ', valid_X_ins.shape)
        print('valid_X_slot dim: ', valid_X_slot.shape)
        print('valid_X_len dim: ', valid_X_len.shape)
        print('valid_Y_dim: ', valid_Y.shape)

        checkpointer = ModelCheckpoint(filepath='weight/{epoch:02d}_' + weight_file, verbose=1, save_weights_only=True,
                                       save_best_only=True)

        self.model.fit([X, X_ins, X_slot, X_len, cn_X, cn_X_ins],
                       [Y, train_temp_prob, train_value_entropy],
                       batch_size=128, epochs=nb_epoch, validation_split=0.,
                       validation_data=([valid_X, valid_X_ins, valid_X_slot, valid_X_len, valid_cn_X, valid_cn_X_ins],
                                        [valid_Y, valid_temp_prob, valid_value_entropy]), verbose=1, callbacks=[checkpointer])

        print('weight saved to ' + '{epoch:02d}_' + weight_file)
