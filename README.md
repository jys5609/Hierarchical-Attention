# Hierarchical-Attention 
**Cross-language Neural Dialog State Tracker for Large Ontologies using Hierarchical Attention** (TASLP 2018), Youngsoo Jang, Jiyeon Ham, Byung-Jun Lee and Kee-Eung Kim. Accepted at ***TASLP 2018***. TASLP anthology:[[PDF]](https://ieeexplore.ieee.org/document/8401898/).  

This code has been written using Python 3.6.5 & Theano 1.0.2 & Keras 2.2.0.

## Setting the Dialog State Tracking Challenge 5 (DSTC5) dataset
Make a directory 'data', then put the DSTC5 data set in 'data'. If you need to ask for a dataset, contact the organizing Committees (http://workshop.colips.org/dstc5/committee.html).

## Training
```console
❱❱❱ python train.py --lstm 100 --lr 0.005 --dropout 0.3 --epoch 100
```
The option you can choose are:
- '--lstm' the number of lstm units (default = 100)
- '--lr' learning rate (default = 0.005)
- '--dropout' dropout rate (default = 0.5)
- '--epoch' the number of epoch (default = 300)

After then, there will be created weight file with named '{#epoch}_dstc5_lstm{#lstm}_lr{#lr}_dr{#dropout}.h5' (ex. 100_dstc5_lstm100_lr005_dr3.h5)

## Predict with finding threshold
```console
❱❱❱ predict.py -lstm 100 -lr 0.005 --dropout 0.3 --epoch 100
```
The arguments are same with training step.
The option you can choose are:
- '--lstm' the number of lstm units (default = 100)
- '--lr' learning rate (default = 0.005)
- '--dropout' dropout rate (default = 0.5)
- '--epoch' the number of epoch (default = 300)

After then, there will be created 2 json files like below.
* 100_dev_dstc5_lstm100_lr005_dr3_fscore.json </br>
* 100_test_dstc5_lstm100_lr005_dr3_fscore.json </br>

## Make a result
For validation result,
```console
❱❱❱ bash dev_run.sh 100_dev_dstc5_lstm100_lr005_dr3_fscore.json
```
For test result,
```console
❱❱❱ bash test_run.sh 100_test_dstc5_lstm100_lr005_dr3_fscore.json
```
Then, you can see the result and there will be created the result files.
* 100_dev_dstc5_lstm100_lr005_dr3_fscore.score.csv
* 100_test_dstc5_lstm100_lr005_dr3_fscore.score.csv
