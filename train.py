from model import *
from utils import *
import argparse, sys

np.random.seed(1)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm', dest='lstm_units', type=int, action='store', default=100, help='Number of LSTM units')
    parser.add_argument('--lr', dest='learning_rate', type=float, action='store', default=0.005, help='Learning rate')
    parser.add_argument('--dropout', dest='dropout', type=float, action='store', default=0.5, help='Dropout')
    parser.add_argument('--epoch', dest='nb_epoch', type=int, action='store', default=300, help='Training epoch')
    parser.add_argument('--type', dest='dstctype', type=int, action='store', default=5, help='DSTC type')

    args = parser.parse_args()

    print ('LSTM units: ', args.lstm_units)
    print ('Learning rate: ', args.learning_rate)
    print ('Dropout: ', args.dropout)
    print('Number of Epoch: ', args.nb_epoch)
    print('DSTC type: ', args.dstctype)
    print ('--------------------------------')

    model = Tracker(embed_dim=200, en_segment_length=520, cn_segment_length=790)
    model.create_model(args.learning_rate, args.lstm_units, args.dropout)

    train_SUID, train_X, train_X_ins, train_X_text, train_X_topic, train_Y, train_Y_text, train_X_slot, train_X_len \
        = model.load_data('train', uttr_accumulate=False)
    dev_SUID, dev_X, dev_X_ins, dev_X_text, dev_X_topic, dev_Y, dev_Y_text, dev_X_slot, dev_X_len \
        = model.load_data('dev', uttr_accumulate=False)

    _, train_cn_X, train_cn_X_ins, _, _, _, _, _ = model.cn_load_data('train', uttr_accumulate=False)
    _, dev_cn_X, dev_cn_X_ins, _, _, _, _, _ = model.cn_load_data('dev', uttr_accumulate=False)

    weight_file = 'dstc'+str(args.dstctype)\
                  + '_lstm'+str(args.lstm_units)\
                  + '_lr'+str(args.learning_rate)[2:]\
                  + '_dr'+str(args.dropout)[2:]\
                  +'.hdf5'

    print ('Weight file : '+ weight_file)

    model.train(train_X, train_X_ins, train_X_slot, train_X_len, train_cn_X, train_cn_X_ins, train_Y,
                dev_X, dev_X_ins, dev_X_slot, dev_X_len, dev_cn_X, dev_cn_X_ins, dev_Y, args.nb_epoch, weight_file)

if __name__=='__main__':
    main(sys.argv)

