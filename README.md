# Multilabel Text Classification using novel CNN Bi-LSTM framework

Text classification is a modelling approach where we have series of sequences as input to predict the class for the particular sequence. This predictive modelling approach tends to pose a challenge in a way that the input sequences that go as inputs do not have a constant length. This invariable length of the sequences accounts for very large vocabulary size and hence it usually requires model to learn long term contexts. 

Now, lot of algorithms and solutions for binary and multi class text classification prevails but in real life tweet or even a sentence and even most of the problems can be represented as multi-label classification problem.

This multi-label classification approach finds its use in lots of major areas such as :

1- Categorizing genre for movies by OTT platforms.
2- Text Classifications by Banking and financial institutions.
3- Automatic caption generation.

Hence, need arises for a well to do AI driven approach. Here we present a deep learning framework that has been used for classifying the sentences into various labels. The aim of the article is to familiarize the audience as to how the CNN and Bi-LSTM networks in combinations is  used for providing a novel multi-label classifier.
The CNN is used as feature extractor and Bi-LSTM as seq2seq learner to get us the desired output.

Below is the flow diagram of the framewok that is used for classifying sentences :

![Alt text]('C:/Users/Diwas/Desktop/12.PNG'?raw=true "CNN Bi-LSTM Architectural flow")


