import numpy as np
from scripts.ontology_reader import *
from scripts.dataset_walker import *
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from gensim.models import word2vec

class EnDataGenerator():
    """English Data generator"""
    def __init__(self):
        self.embed_dim = 200
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_model = word2vec.Word2Vec.load('word2vec/en_word2vec_%ddim' % self.embed_dim)
        self.build_ontology()

    def build_ontology(self):
        onto_file = 'scripts/config/ontology_dstc5.json'
        tagsets = OntologyReader(onto_file).get_tagsets()

        print('building vocabulary')

        self.slots = []  # [[TOPIC, SLOT]]
        self.topic_list = [] # [TOPIC]
        self.slot_loc = {}  # {TOPIC: {SLOT: slot location number}}

        i = 0
        for topic in sorted(tagsets.keys()):
            self.slot_loc[topic] = {}
            self.topic_list.append(topic)
            for slot in sorted(tagsets[topic].keys()):
                self.slots.append([topic, slot])
                self.slot_loc[topic][slot] = i
                i += 1
        self.slot_num = len(self.slots)

        self.slot_value = defaultdict(self.ll)  # {TOPIC: {SLOT: [slot value list]}}
        self.slot_value_vector = defaultdict(self.ll)  # {TOPIC: {SLOT: [slot vector list]}}
        self.slot_voc = defaultdict(self.ls)  # {TOPIC: {SLOT: {w_normalized slot value set}}}

        for topic in tagsets:
            for slot in tagsets[topic]:
                for value in tagsets[topic][slot]:
                    wnormed = self.w_normalize(value)
                    vec = self.get_vector(wnormed)

                    self.slot_value[topic][slot].append(value)
                    self.slot_value_vector[topic][slot].append(vec)
                    self.slot_voc[topic][slot].update(wnormed)

                self.slot_voc[topic][slot].update(self.w_normalize(slot))
                self.slot_voc[topic][slot].update(self.w_normalize(topic))

        # transform list of 1d vectors to 2d matrix
        for topic in tagsets:
            for slot in tagsets[topic]:
                for i in range(len(self.slot_value_vector[topic][slot])):
                    self.slot_value_vector[topic][slot][i] = self.slot_value_vector[topic][slot][i] / np.sqrt(
                        np.sum(np.square(self.slot_value_vector[topic][slot][i])))
                self.slot_value_vector[topic][slot] = np.array(self.slot_value_vector[topic][slot])

        # update total ontology vocabulary
        onto_voc = set()
        for topic in tagsets:
            for slot in tagsets[topic]:
                onto_voc.update(self.slot_voc[topic][slot])

        print('complete ontology vocabulary')

        value_num = [0 for i in range(len(self.slots))]
        for topic in sorted(self.slot_loc.keys()):
            for slot in sorted(self.slot_loc[topic].keys()):
                value_num[self.slot_loc[topic][slot]] = len(self.slot_value[topic][slot])

    def ll(self):
        return defaultdict(list)

    def ls(self):
        return defaultdict(set)

    def w_normalize(self, tokens):
        data = tokens.lower()
        data = re.sub('[^0-9a-z ]+', ' ', data)
        data = [self.lemmatizer.lemmatize(w) for w in data.split()]
        return data

    def get_index(self, w):
        if w in self.word2vec_model.wv.vocab:
            return self.word2vec_model[w]
        else:
            return np.zeros(self.embed_dim)

    def get_indices(self, x):
        return np.array([self.get_index(w) for w in x])

    def get_vector(self, x):
        return np.sum(self.get_indices(x), axis=0)

    def get_lvec(self, label, topic):
        answer_values = [np.zeros(self.embed_dim) for x in range(self.slot_num)]

        for slot, value in label.items():

            if not (slot in self.slot_loc[topic].keys()):
                continue
            slot_where = self.slot_loc[topic][slot]

            answer_values[slot_where] = self.get_vector(self.w_normalize(value[0]))

        return answer_values

    def get_answer(self, label, topic):
        answer_values = [[0 for j in range(len(self.slot_value[self.slots[i][0]][self.slots[i][1]]))] for i in range(self.slot_num)]

        for slot in label.keys():
            for j in range(len(label[slot])):
                answer_values[self.slot_loc[topic][slot]][self.slot_value[topic][slot].index(label[slot][j])] = 1

        result = []
        for i in range(len(answer_values)):
            result.extend(answer_values[i])
        return result

    def get_word_in_slot(self, w, topic, speaker):
        # consider if the word w exist in each slot vocabulary
        # False: -1, True: 1

        res = -np.ones(self.slot_num + 7)

        for slot in self.slot_loc[topic].keys():
            if w in self.slot_voc[topic][slot]:
                res[self.slot_loc[topic][slot]] = 1

        res[self.slot_num + self.topic_list.index(topic)] = 1

        if speaker == 'guide':
            res[-2] = 1
        elif speaker == 'tourist':
            res[-1] = 1

        return res

    def get_in_slot(self, x, topic, speaker_list):
        result = []
        eou_index = 0

        for w in x:
            speaker = speaker_list[eou_index]
            result.append(self.get_word_in_slot(w, topic, speaker))
            if w == '%EOU':
                eou_index += 1

        return np.array(result)

    def slot_check(self, topic):
        slot_on = np.zeros((self.slot_num))
        for slot in self.slot_loc[topic].keys():
            slot_on[self.slot_loc[topic][slot]] = 1
        return slot_on

    def data_walker(self, dataset, lang):
        uttr_segs = defaultdict(list)
        topic_segs = {}
        label_segs = {}
        speaker_segs = defaultdict(list)

        for session in dataset:
            sid = session.log['session_id']
            segment_id = None
            for (uttr, trans, label) in session:
                uid = uttr['utter_index']
                target_bio = uttr['segment_info']['target_bio'].lower()
                if target_bio in ['b', 'o']:
                    if segment_id is not None:
                        print (len(uttr_segs), '------', segment_id)
                        segment_id = None
                if target_bio == 'o':
                    continue
                if segment_id is None:
                    segment_id = 's'+str(sid).zfill(5)+'u'+str(uid).zfill(5)
                    label_segs[segment_id] = label['frame_label']
                    topic_segs[segment_id] = uttr['segment_info']['topic']
                if lang == 'en':
                    uttr_segs[segment_id].append(self.w_normalize(uttr['transcript'])+['%EOU'])
                    speaker_segs[segment_id].append(uttr['speaker'].lower())
                else:
                    if len(trans['translated']) > 0:
                        uttr_segs[segment_id].append(self.w_normalize(trans['translated'][0]['hyp'])+['%EOU'])
                        speaker_segs[segment_id].append(uttr['speaker'].lower())
                    elif len(trans['translated']) == 0:
                        uttr_segs[segment_id].append(['%EOU'])
                        speaker_segs[segment_id].append('None')

        return uttr_segs, label_segs, topic_segs, speaker_segs

    def data_generate(self, dataset, lang, uttr_accumulate):
        uttr_segs, label_segs, topic_segs, speaker_segs = self.data_walker(dataset, lang)

        SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot = [[] for i in range(8)]

        for suid in sorted(uttr_segs.keys()):
            topic = topic_segs[suid]
            y_text = label_segs[suid]
            y = self.get_lvec(y_text, topic)
            speaker_list = speaker_segs[suid]

            if uttr_accumulate:
                x = []
                s_index = 0
                sid = int(suid[1:6])
                uid = int(suid[7:])
                for uttr in uttr_segs[suid]:
                    id = 's' + str(sid).zfill(5) + 'u' + str(uid).zfill(5)
                    SUID.append(id)
                    x += uttr

                    X.append(self.get_indices(x))
                    X_ins.append(self.get_in_slot(x, topic, speaker_list))
                    X_topic.append(topic)
                    x_text = ' '.join(x)
                    x_text = x_text.replace('%EOU', '\n')
                    X_text.append(x_text)

                    Y.append(y)
                    Y_text.append(y_text)

                    X_slot.append(self.slot_check(topic))

                    s_index += 1
                    uid += 1
            else:
                x = [item for sublist in uttr_segs[suid] for item in sublist]

                SUID.append(suid)
                X.append(self.get_indices(x))
                X_ins.append(self.get_in_slot(x, topic, speaker_list))
                X_text.append(uttr_segs[suid])
                X_topic.append(topic)

                Y.append(y)
                Y_text.append(y_text)

                X_slot.append(self.slot_check(topic))

        return SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot

    def write_file(self, datatype, lang, uttr_accumulate):
        if lang == 'en':
            dataset = dataset_walker('dstc5' + '_' + datatype, dataroot='data', translations=False, labels=True)
        else:
            dataset = dataset_walker('dstc5' + '_' + datatype, dataroot='data', translations=True, labels=True)

        SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot = self.data_generate(dataset, lang, uttr_accumulate)

        return SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot

    def get_general(self):
        return self.slots, self.slot_loc, self.slot_value, self.slot_value_vector

    def get_X_len(self, X):
        X_len = []
        for sentence in X:
            X_len.append(len(sentence))
        return X_len

    def get_data(self, datatype, acc):
        if datatype == 'train':
            lang = 'en'
        else:
            lang = 'cn'
        SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot = self.write_file(datatype, lang, acc)
        X_len = self.get_X_len(X)
        return SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_slot, X_len