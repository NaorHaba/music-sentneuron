# Learning to Generate Music with Sentiment

This repository contains the source code to reproduce the results of the [ISMIR'19](https://ismir2019.ewi.tudelft.nl/) 
paper [Learning to Generate Music with Sentiment](http://www.lucasnferreira.com/papers/2019/ismir-learning.pdf). 
This paper proposes a deep learning method to generate music with a given sentiment (positive or negative). 
It uses a new dataset called VGMIDI which contains 95 labelled pieces according to sentiment as well as 728 
non-labelled pieces. All pieces are piano arrangements of video game soundtracks in MIDI format.

## Installing Dependencies

This project depends on tensorflow (2.0), numpy and music21, so you need to install them first:

```
$ pip3 install tensorflow tensorflow-gpu numpy music21
```

## Dowload Data

The VGMIDI dataset has its own [GitHub repo](https://github.com/lucasnfe/vgmidi). You can download it as follws:

```
$ git clone https://github.com/lucasnfe/vgmidi.git
```

## Results

This paper has two main results: (a) sentiment analysis of symbolic music and (b) generation of symbolic 
music with sentiment.

### (a) Sentiment Analysis 

1. Train a generative LSTM on unlabelled pieces:

```
python3 train_generative.py --train vgmidi/unlabelled/train/ --test vgmidi/unlabelled/test/ --embed 256 --units 4096 --layers 1 --batch 64 --lrate 0.00001 --seqlen 256 --drop 0.05 --epochs 10
```

2. Use the final cell states of the generative LSTM to encode the labelled pieces and train a Logistic Regression to classify sentiment in symbolic music:

```
python3 train_classifier.py --model training_checkpoints --ch2ix char2idx.json --embed 256 --units 4096 --layers 1 --train vgmidi/labelled/vgmidi_sent_train.csv --test vgmidi/labelled/vgmidi_sent_test.csv --cellix 1
```

### (b) Generative 

TODO
