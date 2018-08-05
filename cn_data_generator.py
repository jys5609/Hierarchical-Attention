import jieba
import numpy as np
from scripts.ontology_reader import *
from scripts.dataset_walker import *
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from gensim.models import word2vec

class CnDataGenerator():
    """Chinese Data generator"""
    def __init__(self):
        self.embed_dim = 200
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        self.en_word2vec_model = word2vec.Word2Vec.load('word2vec/en_word2vec_%ddim' % self.embed_dim)
        self.cn_word2vec_model = word2vec.Word2Vec.load('word2vec/cn_word2vec_%ddim' % self.embed_dim)

        self.build_ontology()

    def build_ontology(self):
        onto_file = 'scripts/config/ontology_dstc5.json'
        en_tagsets = OntologyReader(onto_file).get_tagsets()
        tagsets = OntologyReader(onto_file).get_translated_tagsets()

        print ('building vocabulary')

        self.slots = [] # [[TOPIC, SLOT]]
        self.topic_list = []  # [TOPIC]
        self.slot_loc = {} # {TOPIC: {SLOT: slot location number}}
        i = 0
        for topic in sorted(tagsets.keys()):
            self.slot_loc[topic] = {}
            self.topic_list.append(topic)
            for slot in sorted(tagsets[topic].keys()):
                self.slots.append([topic, slot])
                self.slot_loc[topic][slot] = i
                i+=1

        self.slot_num = len(self.slots)

        self.slot_value = defaultdict(self.ll)	# {TOPIC: {SLOT: [slot value list]}}
        self.slot_value_vector = defaultdict(self.ll) # {TOPIC: {SLOT: [slot vector list]}}
        self.slot_voc = defaultdict(self.ls) # {TOPIC: {SLOT: {w_normalized slot value set}}}
        value_match_cn_to_en ={}
        value_match_en_to_cn ={}
        value_match_en_to_cn_list = {}

        for topic in tagsets:
            value_match_en_to_cn_list[topic] = {}
            for slot in tagsets[topic]:
                value_match_en_to_cn_list[topic][slot]={}
                for value_list in tagsets[topic][slot]:
                    cn_value = value_list['translated_cn'][0]
                    en_value = value_list['entry_en']
                    value_match_cn_to_en[cn_value] = en_value
                    value_match_en_to_cn[en_value] = cn_value
                    value_match_en_to_cn_list[topic][slot][en_value] = cn_value

                    wnormed = self.token_full_sen(cn_value)
                    vec = self.get_vector(wnormed)

                    self.slot_value[topic][slot].append(cn_value)
                    self.slot_value_vector[topic][slot].append(vec)
                    self.slot_voc[topic][slot].update(wnormed)

        # transform list of 1d vectors to 2d matrix
        for topic in tagsets:
            for slot in tagsets[topic]:
                self.slot_value_vector[topic][slot] = np.array(self.slot_value_vector[topic][slot])

        # English slot voc
        for topic in en_tagsets:
            for slot in en_tagsets[topic]:
                for value in en_tagsets[topic][slot]:
                    wnormed = self.w_normalize(value)
                    self.slot_voc[topic][slot].update(wnormed)

                self.slot_voc[topic][slot].update(self.w_normalize(slot))
                self.slot_voc[topic][slot].update(self.w_normalize(topic))

        self.en_slot_value = defaultdict(self.ll)    # {TOPIC: {SLOT: [slot value list]}}
        self.en_slot_value_vector = defaultdict(self.ll) # {TOPIC: {SLOT: [slot vector list]}}

        for topic in en_tagsets:
            for slot in en_tagsets[topic]:
                for value in en_tagsets[topic][slot]:
                    wnormed = self.w_normalize(value)
                    vec = self.en_get_vector(wnormed)

                    self.en_slot_value[topic][slot].append(value)
                    self.en_slot_value_vector[topic][slot].append(vec)

        # transform list of 1d vectors to 2d matrix
        for topic in en_tagsets:
            for slot in en_tagsets[topic]:
                for i in range(len(self.en_slot_value_vector[topic][slot])):
                    self.en_slot_value_vector[topic][slot][i] = self.en_slot_value_vector[topic][slot][i] / np.sqrt(np.sum(np.square(self.en_slot_value_vector[topic][slot][i])))
                self.en_slot_value_vector[topic][slot] = np.array(self.en_slot_value_vector[topic][slot])

    def ll(self):
        return defaultdict(list)

    def ls(self):
        return defaultdict(set)

    def find_all(self, x, value):
        index_list = []
        for i in range(len(x)):
            if x[i] == value:
                index_list.append(i)
        return index_list

    def token_full_sen(self, sen):
        output = []
        for token in jieba.tokenize(sen):
            word = token[0]
            output.append(word)

        result = []
        for i in output:
            if ord(i[0]) < 128:
                result.append(i)
            else:
                result.extend(list(i))

        return result

    def token_char(self, words):
        return list(words)

    def w_normalize(self, tokens):
        data = tokens.lower()
        data = re.sub('[^0-9a-z ]+', ' ', data)
        data = [self.lemmatizer.lemmatize(w) for w in data.split()]
        if data == []:
            data = ['']
        return data

    def en_get_index(self, w):
        if len(w.split(' ')) == 0:
            return np.zeros(self.embed_dim)
        elif len(w.split(' ')) == 1:
            if self.w_normalize(w.lower())[0] in self.en_word2vec_model.wv.vocab:
                return self.en_word2vec_model[self.w_normalize(w.lower())[0]]
            elif self.stemmer.stem(self.w_normalize(w)[0]) in self.en_word2vec_model.wv.vocab:
                return self.en_word2vec_model[self.stemmer.stem(self.w_normalize(w)[0])]
            else:
                return np.zeros(self.embed_dim)

        else:
            word = np.zeros(self.embed_dim)
            for i in w.split(' '):
                if self.w_normalize(i.lower())[0] in self.en_word2vec_model.wv.vocab:
                    word += self.en_word2vec_model[self.w_normalize(i.lower())[0]]
                elif self.stemmer.stem(self.w_normalize(i)[0]) in self.en_word2vec_model.wv.vocab:
                    word += self.en_word2vec_model[self.stemmer.stem(self.w_normalize(i)[0])]
                else:
                    word += np.zeros(self.embed_dim)
            if np.sqrt(np.sum(np.square(word))) == 0:
                return np.zeros(self.embed_dim)
            word = word / np.sqrt(np.sum(np.square(word)))
            return word

    def en_get_indices(self, x):
        return np.array([self.en_get_index(w) for w in x])

    def en_get_vector(self, x):
        return np.sum(self.en_get_indices(x), axis=0)

    def get_index(self, w):
        if w in self.cn_word2vec_model.wv.vocab:
            return self.cn_word2vec_model[w]
        else:
            if w != '%EOU':
                if w.lower() in self.cn_word2vec_model.wv.vocab:
                    return self.cn_word2vec_model[w.lower()]
            return np.zeros(self.embed_dim)

    def get_indices(self, x):
        return np.array([self.get_index(w) for w in x])

    def get_vector(self, x):
        return np.sum(self.get_indices(x), axis=0)

    def get_answer(self, label, topic):
        answer_values = [[0 for j in range(len(self.en_slot_value[self.slots[i][0]][self.slots[i][1]]))] for i in range(self.slot_num)]

        for slot in label.keys():
            for j in range(len(label[slot])):
                answer_values[self.slot_loc[topic][slot]][self.en_slot_value[topic][slot].index(label[slot][j])] = 1

        result = []
        for i in range(len(answer_values)):
            result.extend(answer_values[i])
        return result

    def get_lvec(self, label, topic):

        answer_values = [np.zeros(self.embed_dim) for x in range(self.slot_num)]

        for slot, value in label.iteritems():

            if not (slot in self.slot_loc[topic].keys()):
                continue
            slot_where = self.slot_loc[topic][slot]

            answer_values[slot_where] = self.en_get_vector(self.w_normalize(value[0]))

        return answer_values

    def en_get_word_in_slot(self, w, topic):
        # consider if the word w exist in each slot vocabulary
        # False: -1, True: 1

        res = -np.ones(self.slot_num)

        for slot in self.slot_loc[topic].keys():
            for w_part in w.split(' '):
                if w_part in self.slot_voc[topic][slot]:
                    res[self.slot_loc[topic][slot]] = 1

        return res

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

    def get_in_slot(self, x, x_align, topic, speaker_list):
        en_ins = np.array([self.en_get_word_in_slot(w, topic) for w in x_align])

        cn_result = []
        eou_index = 0

        for w in x:
            speaker = speaker_list[eou_index]
            cn_result.append(self.get_word_in_slot(w, topic, speaker))
            if w == '%EOU':
                eou_index += 1

        cn_ins = np.array(cn_result)

        x_ins = np.concatenate((en_ins, cn_ins), axis=1)

        return x_ins

    def data_walker(self, dataset, lang):
        uttr_segs = defaultdict(list)
        topic_segs = {}
        label_segs = {}
        speaker_segs = defaultdict(list)
        align_segs = defaultdict(list)

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

                # For train set
                if lang == 'en':
                    if len(trans['translated'])>0:
                        uttr_segs[segment_id].append(self.token_char(trans['translated'][0]['hyp'])+['%EOU'])
                        speaker_segs[segment_id].append(uttr['speaker'].lower())

                        # for alignment information
                        align_list = ['' for i in range(len(self.token_char(trans['translated'][0]['hyp'])+['%EOU']))]
                        align_list[-1] = '%EOU'

                        for align_info in trans['translated'][0]['align']:
                            if not align_info[0] in self.w_normalize(uttr['transcript']):
                                if len(self.w_normalize(align_info[0])) == 1:
                                    align_info[0] = self.w_normalize(align_info[0])[0]
                                elif len(self.w_normalize(align_info[0])) == 0:
                                    align_info[0] = ''
                                else:
                                    align_info[0] = ' '.join(self.w_normalize(align_info[0]))

                            for align_index in align_info[1]:
                                align_list[align_index] = align_info[0]

                        align_segs[segment_id].append(align_list)

                    else:
                        assert False

                # For valid & test set
                else:
                    if len(trans['translated'])>0:
                        uttr_list = self.token_full_sen(uttr['transcript']) + ['%EOU']
                        uttr_segs[segment_id].append(uttr_list)
                        speaker_segs[segment_id].append(uttr['speaker'].lower())


                        en_list = trans['translated'][0]['hyp'].split(' ')
                        align_list = ['' for i in range(len(uttr_list))]
                        align_list[-1] = '%EOU'

                        for align_info in trans['translated'][0]['align']:
                            for i in range(len(self.token_full_sen(align_info[0]))):
                                word = []
                                for j in align_info[1]:
                                    word.append(en_list[j])

                                if self.token_full_sen(align_info[0])[i] in (self.token_full_sen(uttr['transcript']) + ['%EOU']):
                                    indexs = self.find_all(self.token_full_sen(uttr['transcript']) + ['%EOU'], self.token_full_sen(align_info[0])[i])
                                    for k in indexs:
                                        if align_list[k] == '':
                                            align_list[k] = ' '.join(word)
                                        else:
                                            align_list[k] = align_list[k] + ' ' + ' '.join(word)

                        align_segs[segment_id].append(align_list)

                    else:
                        uttr_list = self.token_full_sen(uttr['transcript']) + ['%EOU']
                        uttr_segs[segment_id].append(uttr_list)
                        speaker_segs[segment_id].append('None')
                        align_list = ['' for i in range(len(uttr_list))]
                        align_list[-1] = '%EOU'
                        align_segs[segment_id].append(align_list)

        return uttr_segs, label_segs, topic_segs, speaker_segs, align_segs

    def data_generate(self, dataset, lang, uttr_accumulate):
        uttr_segs, label_segs, topic_segs, speaker_segs, align_segs = self.data_walker(dataset, lang)

        SUID, X, X_ins, X_text, X_topic, Y, Y_text = [[] for i in range(7)]

        for suid in sorted(uttr_segs.keys()):
            topic = topic_segs[suid]
            y_text = label_segs[suid]
            y = self.get_answer(y_text, topic)
            speaker_list = speaker_segs[suid]

            if uttr_accumulate:
                x = []
                x_align = []
                sid = int(suid[1:6])
                uid = int(suid[7:])

                for i in range(len(uttr_segs[suid])):
                    uttr = uttr_segs[suid][i]
                    align = align_segs[suid][i]
                    id = 's' + str(sid).zfill(5) + 'u' + str(uid).zfill(5)
                    SUID.append(id)
                    x += uttr
                    x_align += align
                    X.append(self.get_indices(x))
                    X_ins.append(self.get_in_slot(x, x_align, topic, speaker_list))
                    X_topic.append(topic)
                    x_text = ' '.join(x)
                    x_text = x_text.replace('%EOU', '\n')
                    X_text.append(x_text)

                    Y.append(y)
                    Y_text.append(y_text)
                    uid += 1

            else:
                x = [item for sublist in uttr_segs[suid] for item in sublist]
                x_align = [item for sublist in align_segs[suid] for item in sublist]

                SUID.append(suid)
                X.append(self.get_indices(x))
                X_ins.append(self.get_in_slot(x, x_align, topic, speaker_list))
                X_text.append(uttr_segs[suid])
                X_topic.append(topic)

                Y.append(y)
                Y_text.append(y_text)

        return SUID, X, X_ins, X_text, X_topic, Y, Y_text

    def write_file(self, datatype, lang, uttr_accumulate):
        if lang == 'en':
            dataset = dataset_walker('dstc5' + '_' + datatype, dataroot='data', translations=True, labels=True)
        else:
            dataset = dataset_walker('dstc5' + '_' + datatype, dataroot='data', translations=True, labels=True)

        SUID, X, X_ins, X_text, X_topic, Y, Y_text = self.data_generate(dataset, lang, uttr_accumulate)

        return SUID, X, X_ins, X_text, X_topic, Y, Y_text

    def get_general(self):
        return self.slots, self.slot_loc, self.en_slot_value, self.en_slot_value_vector

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
        SUID, X, X_ins, X_text, X_topic, Y, Y_text = self.write_file(datatype, lang, acc)
        X_len = self.get_X_len(X)
        return SUID, X, X_ins, X_text, X_topic, Y, Y_text, X_len