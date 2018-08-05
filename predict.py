import os
from model import *
from utils import *
from json_formatter import JSONFormatter

np.random.seed(1)

class Predictor(Tracker):
    def load_weight(self, weight_file):
        # print weight_file
        assert self.exist_model
        self.model.load_weights('weight/' + weight_file)

    def predict(self, X, X_ins, X_slot, X_len, cn_X, cn_X_ins):
        assert self.exist_model

        return self.model.predict([X, X_ins, X_slot, X_len, cn_X, cn_X_ins])

    def calculate_entropy(self, prob):
        entropy = np.zeros((prob.shape[0], self.slot_num))
        for i in range(prob.shape[0]):
            for j in range(prob.shape[2]):
                log_prob = np.log(prob[i, :, j])
                entropy[i][j] = -np.dot(prob[i, :, j], log_prob)

        return entropy

    def calculate_value_entropy(self, prob):
        # input: (#data, 6216, 1)
        # output: (#data, 30)
        entropy = np.zeros((prob.shape[0], self.slot_num))

        for i in range(prob.shape[0]):
            s = 0
            for j in range(self.slot_num):
                log_prob = np.log(prob[i, s:s + self.value_num[j]])
                log_prob = np.reshape(log_prob, (log_prob.shape[0]))

                entropy[i][j] = -np.dot(
                    np.reshape(prob[i, s:s + self.value_num[j]], (prob[i, s:s + self.value_num[j]].shape[0])), log_prob)
                s += self.value_num[j]
        return entropy

    # calculate all cossim
    def calculate_all_cossim(self, out_embed, X_topic):
        total_cossim = np.zeros((len(X_topic), self.slot_num))

        for i in range(len(X_topic)):
            topic = X_topic[i]
            for slot in self.slot_loc[topic].keys():
                (_, nearest_cossim) = self.find_nearest_value_cossim(out_embed[i][self.slot_loc[topic][slot]], topic,
                                                                     slot)
                total_cossim[i][self.slot_loc[topic][slot]] = nearest_cossim

        return total_cossim

    # fine nearest value and corresponding cosine similarity for given word vector
    def find_nearest_value_cossim(self, vector, topic, slot):
        candidate_vectors = self.slot_value_vector[topic][slot]

        candidate_vectors_norm = np.linalg.norm(candidate_vectors, axis=1)
        candidate_vectors_norm[candidate_vectors_norm == 0] = 1
        candidate_vectors_norm = np.divide(candidate_vectors.T, candidate_vectors_norm).T

        vector_norm = np.linalg.norm(vector)
        vector_norm = vector if (vector_norm == 0) else np.divide(vector, vector_norm)

        cos_distance = np.dot(candidate_vectors_norm, vector_norm)
        nearest_loc = np.argmax(cos_distance)

        max_cos_distance = cos_distance[nearest_loc]
        max_cos_value = self.slot_value[topic][slot][nearest_loc]

        return max_cos_value, max_cos_distance

    # calculate total value and cosine similarity for predicted embed output
    def calculate_value_cossim(self, out_embed, dev_X_topic):
        total_value = []

        for i in range(len(out_embed)):
            cur_value = {}
            topic = dev_X_topic[i]

            for (t, s) in self.slots:
                if t == topic:
                    slot_loc = self.slot_loc[t][s]
                    (nearest_value, nearest_cossim) = self.find_nearest_value_cossim(out_embed[i][slot_loc], t, s)
                    cur_value[s] = nearest_value

            total_value.append(cur_value)

        return total_value

    # calculate accuracy for given slot
    def calculate_slot_accuracy(self, slot, entropy_thres, cossim_thres, total_value,
                                total_entropy, total_cossim, dev_X_topic, dev_Y_text):
        total_segs = 0.0
        correct_segs = 0.0

        for i in range(len(dev_Y_text)):
            topic = dev_X_topic[i]

            if slot in self.slot_loc[topic].keys():
                slot_loc = self.slot_loc[topic][slot]
                total_segs += 1
                value_exist = (total_entropy[i][slot_loc] < entropy_thres)\
                              and (total_cossim[i][slot_loc] > cossim_thres)
                if value_exist:
                    if slot in dev_Y_text[i].keys():
                        if total_value[i][slot] == dev_Y_text[i][slot][0]:
                            correct_segs += 1
                else:
                    if slot not in dev_Y_text[i].keys():
                        correct_segs += 1
        accuracy = correct_segs / total_segs
        return accuracy

    # calculate fscore for given slot

    def calculate_slot_fscore(self, slot, entropy_thres, cossim_thres, total_value,
                              total_entropy, total_cossim, dev_X_topic, dev_Y_text):
        pred_slots = 0.0
        ref_slots = 0.0
        correct_slots = 0.0

        for i in range(len(dev_Y_text)):
            topic = dev_X_topic[i]

            if slot in self.slot_loc[topic].keys():
                slot_loc = self.slot_loc[topic][slot]
                value_exist = (total_entropy[i][slot_loc] < entropy_thres) and (total_cossim[i][slot_loc] > cossim_thres)

                if value_exist:
                    pred_slots += 1
                    if slot in dev_Y_text[i].keys():
                        if total_value[i][slot] == dev_Y_text[i][slot][0]:
                            correct_slots += 1

                if slot in dev_Y_text[i].keys():
                    ref_slots += 1

        if pred_slots == 0 or ref_slots == 0:
            return 0.
        precision = correct_slots / pred_slots
        recall = correct_slots / ref_slots

        if precision + recall == 0:
            return 0.

        fscore = 2 * precision * recall / (precision + recall)
        return fscore

    # decide theshold by given total value, entropy, and cossim
    def decide_threshold(self, total_value, total_entropy, value_entropy, total_cossim, dev_X_topic, dev_Y_text,
                         criteria):
        entropy_max_thres = {}
        value_entropy_max_thres = {}
        cossim_max_thres = {}
        entropy_candidate_thres = {}
        value_entropy_candidate_thres = {}
        cossim_candidate_thres = {}

        assert criteria in ['accuracy', 'fscore']

        if criteria == 'accuracy':
            criteria_function = self.calculate_slot_accuracy
        elif criteria == 'fscore':
            criteria_function = self.calculate_slot_fscore

        for topic, slot in self.slots:
            entropy_max_thres[self.slot_loc[topic][slot]] = 0.0
            entropy_candidate_thres[self.slot_loc[topic][slot]] = set()

            value_entropy_max_thres[self.slot_loc[topic][slot]] = 0.0
            value_entropy_candidate_thres[self.slot_loc[topic][slot]] = set()

            cossim_max_thres[self.slot_loc[topic][slot]] = 0.0
            cossim_candidate_thres[self.slot_loc[topic][slot]] = set()

        count = [0 for i in range(30)]

        for i in range(len(dev_X_topic)):
            topic = dev_X_topic[i]
            for k in dev_Y_text[i].keys():
                count[self.slot_loc[topic][k]] += 1

            for slot in self.slot_loc[topic].keys():
                entropy_candidate_thres[self.slot_loc[topic][slot]].add(total_entropy[i, self.slot_loc[topic][slot]])
                value_entropy_candidate_thres[self.slot_loc[topic][slot]].add(
                    value_entropy[i, self.slot_loc[topic][slot]])
                cossim_candidate_thres[self.slot_loc[topic][slot]].add(total_cossim[i, self.slot_loc[topic][slot]])

        slot_list = self.slots

        for topic, slot in slot_list:
            entropy_slot_max_thres = entropy_max_thres[self.slot_loc[topic][slot]]
            value_entropy_slot_max_thres = value_entropy_max_thres[self.slot_loc[topic][slot]]
            cossim_slot_max_thres = cossim_max_thres[self.slot_loc[topic][slot]]
            slot_max_acc = criteria_function(slot, entropy_slot_max_thres, cossim_slot_max_thres,
                                             total_value, total_entropy, total_cossim, dev_X_topic, dev_Y_text)

            for entropy_candidate in entropy_candidate_thres[self.slot_loc[topic][slot]]:
                value_entropy_candidate = 0
                for cossim_candidate in cossim_candidate_thres[self.slot_loc[topic][slot]]:
                    cur_slot_max_acc = criteria_function(slot, entropy_candidate, cossim_candidate,
                                                         total_value, total_entropy, total_cossim, dev_X_topic, dev_Y_text)
                    if cur_slot_max_acc > slot_max_acc:
                        entropy_slot_max_thres = entropy_candidate
                        value_entropy_slot_max_thres = value_entropy_candidate
                        cossim_slot_max_thres = cossim_candidate
                        slot_max_acc = cur_slot_max_acc
                    entropy_max_thres[self.slot_loc[topic][slot]] = entropy_slot_max_thres
                    value_entropy_max_thres[self.slot_loc[topic][slot]] = value_entropy_slot_max_thres
                    cossim_max_thres[self.slot_loc[topic][slot]] = cossim_slot_max_thres

            print (topic, slot, count[self.slot_loc[topic][slot]], slot_max_acc, entropy_slot_max_thres, value_entropy_slot_max_thres, cossim_slot_max_thres)

        return entropy_max_thres, value_entropy_max_thres, cossim_max_thres

    # return total solution by decided threshold

    def get_solution(self, total_value, entropy_thresholds, cossim_thresholds, total_entropy,
                     total_cossim, dev_X_topic):
        total_solution = []

        for i in range(len(total_value)):
            topic = dev_X_topic[i]
            cur_solution = {}

            for slot in self.slot_loc[topic].keys():
                slot_loc = self.slot_loc[topic][slot]
                value_exist = (total_entropy[i][slot_loc] < entropy_thresholds[slot_loc])\
                              and (total_cossim[i][slot_loc] > cossim_thresholds[slot_loc])

                if value_exist:
                    cur_solution[slot] = [total_value[i][slot]]

            total_solution.append(cur_solution)
        return total_solution

    # calculate final accuracy
    def calculate_accuracy(self, total_solution, dev_Y_text):
        total_segs = 0.
        correct_segs = 0.

        for i in range(len(dev_Y_text)):
            total_segs += 1
            if total_solution[i] == dev_Y_text[i]:
                correct_segs += 1
        print ('correct', correct_segs)
        print ('total', total_segs)
        return correct_segs / total_segs

    def predict_write_json(self, dstctype, datatype, entropy_thresholds, cossim_thresholds, json_file):
        assert self.exist_model
        SUID, X, X_ins, X_text, X_topic, _, Y_text, X_slot, X_len = self.load_data(datatype, uttr_accumulate=True)
        _, cn_X, cn_X_ins, _, _, _, _, _ = self.cn_load_data(datatype, uttr_accumulate=True)

        pred = self.predict(X, X_ins, X_slot, X_len, cn_X, cn_X_ins)

        total_entropy = self.calculate_entropy(pred[1])
        value_entropy = self.calculate_value_entropy(pred[2])
        total_entropy += value_entropy
        total_cossim = self.calculate_all_cossim(pred[0], X_topic)
        total_value = self.calculate_value_cossim(pred[0], X_topic)
        solution = self.get_solution(total_value, entropy_thresholds, cossim_thresholds,
                                     total_entropy, total_cossim, X_topic)

        dataset_name = 'dstc' + str(dstctype) + '_' + datatype
        jf = JSONFormatter(dataset_name, 0.1, SUID, solution)
        jf.dump_to_file(json_file)
        print ('Dumped to ', json_file)

import argparse, sys

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm', dest='lstm_units', type=int, action='store', default=100, help='Number of LSTM units')
    parser.add_argument('--lr', dest='learning_rate', type=float, action='store', default=0.005, help='Learning rate')
    parser.add_argument('--dropout', dest='dropout', type=float, action='store', default=0.5, help='Dropout')
    parser.add_argument('--epoch', dest='nb_epoch', type=int, action='store', default=300, help='Training epoch')
    parser.add_argument('--type', dest='dstctype', type=int, action='store', default=5, help='DSTC type')
    parser.add_argument('--criteria', dest='criteria', action='store', default='f', help='Decide criteria')

    args = parser.parse_args()

    print ('LSTM units: ', args.lstm_units)
    print ('Learning rate: ', args.learning_rate)
    print ('Dropout: ', args.dropout)
    print('Number of Epoch: ', args.nb_epoch)
    print('DSTC type: ', args.dstctype)
    print ('--------------------------------')

    if args.criteria == 'a':
        criteria = 'accuracy'
    else:
        criteria = 'fscore'

    weight_file = str(args.nb_epoch) \
                  + '_dstc' + str(args.dstctype) \
                  + '_lstm' + str(args.lstm_units) \
                  + '_lr' + str(args.learning_rate)[2:] \
                  + '_dr' + str(args.dropout)[2:] \
                  + '.hdf5'

    dev_json_file = 'dev_json/' + str(args.nb_epoch) + '_dev' \
                    + '_dstc' + str(args.dstctype) \
                    + '_lstm' + str(args.lstm_units) \
                    + '_lr' + str(args.learning_rate)[2:] \
                    + '_dr' + str(args.dropout)[2:] \
                    + '_' + criteria + '.json'

    test_json_file = 'test_json/' + str(args.nb_epoch) + '_test' \
                     + '_dstc' + str(args.dstctype) \
                     + '_lstm' + str(args.lstm_units) \
                     + '_lr' + str(args.learning_rate)[2:] \
                     + '_dr' + str(args.dropout)[2:] \
                     + '_' + criteria + '.json'

    print ('Weight file : ' + weight_file)

    predictor = Predictor(embed_dim=200, en_segment_length=520, cn_segment_length=790)
    if not os.path.exists('weight/' + weight_file):
        predictor.load_weight(weight_file)

    predictor.create_model(args.learning_rate, args.lstm_units, args.dropout)
    predictor.load_weight(weight_file)

    SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot, X_len = predictor.load_data('dev', uttr_accumulate=False)
    _, cn_X, cn_X_ins, _, _, _, _, _ = predictor.cn_load_data('dev', uttr_accumulate=False)

    pred = predictor.predict(X, X_ins, X_slot, X_len, cn_X, cn_X_ins)

    total_entropy = predictor.calculate_entropy(pred[1])
    total_entropy += predictor.calculate_value_entropy(pred[2])
    value_entropy = predictor.calculate_value_entropy(pred[2])

    total_cossim = predictor.calculate_all_cossim(pred[0], X_topic)
    total_value = predictor.calculate_value_cossim(pred[0], X_topic)

    entropy_thresholds, value_entropy_thresholds, cossim_thresholds \
        = predictor.decide_threshold(total_value, total_entropy, value_entropy, total_cossim, X_topic, Y_text, criteria)

    print ('Entropy Threshold: ', entropy_thresholds)
    print ('Cossim Threshold: ', cossim_thresholds)
    predictor.predict_write_json(args.dstctype, 'dev', entropy_thresholds, cossim_thresholds, dev_json_file)
    predictor.predict_write_json(args.dstctype, 'test', entropy_thresholds, cossim_thresholds, test_json_file)

if __name__ == '__main__':
    main(sys.argv)
