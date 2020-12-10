## Welcome to our project page! It's time to pay ATTENTION.

### Project Information
University: Northwestern University

Professor: Prof. Bryan Pardo

Project Members & Contact Information:
  Yao Gu email: YaoGu2024 at u.northwestern.edu 
  Zijian Zhao email: zijianzhao2019 at u.northwestern.edu 
  Yuxiang(Alvin) Chen  email: yuxiangchen2021 at u.northwestern.edu

Github Repository:
  Here is our [GitHub Repository](https://github.com/yaogood/stock-price-prediction)

### Motivation:

Stock index value prediction is a highly challenging study for both research and industry. Applying Deep Learning techniques to financial time series datasets has been a popular methodology. Existing prediction models are mostly based on complex Multi-Layer Perceptrons, Recurrent or Convolutional Neural Networks. However, Attention is a very useful mechanism as the length of the input data increases. It has been widely used in NLP tasks but underutilized in stock time-series prediction models. Thus, we built three deep learning models with different extent of application of attention to mine the same financial time series datasets and predict stock index price moves and evaluate their performances to examine how applicaiton of attention affect model perfromance.

The three models we chose are: 
  1. LSTM Encoder-Decoder Network; 
  2. LSTM Encoder-Decoder Network with Attention; 
  3. Transformer; 
  
This research problem is an important one because we can verify what extent of attention mechanism is the most appropriate for financial time series prediction and effective model construction. No research has been conducted with a focus on comparing these three models with different attention applications. The conclusion will be meaningful for future new model development. This is also a rewarding problem to solve because mining huge amounts of financial datasets can bring a lot of benefits for investing judgements. 

### Model Details

Here are brief description of our three models:

For the baseline, we used a LSTM based model. It's a standard seq2seq architecture, in which two recurrent neural networks work together to transform one sequence to another. The encoder condenses an input sequence into a hidden vector, and the decoder unfolds that vector into a new sequence. In our settings, the encoder is a 3-layer LSTM whose inputs are 30 days of stock data, the decoder is a single LSTM cell which will be used repeatedly to generate the stock index of the future 5 days. The optimizer is an Adam optimizer with default parameters, we used MSE loss to train the model.

Our LSTM + Attention model basically has the same architecture as the LSTM model, but it has one more attention layer which can calculate the weight using the encoder output. In our model we used Bahdanau Attention. The other settings are the same as the first LSTM model, a defualt Adam optimzier and MSE criterion.

Our transformer model is build based on the paper [Attention Is All You Need]({%https://arxiv.org/abs/1706.03762%}). We replace the embedding layer of NLP tasks with a 1-D convolutional layer for projecting the time-series input into a length=dmodel vector.   Our transformermodel uses 6 layers transformer encoder and 3 layers decoder,8 heads self-attention, and dmodel=512. We use an SGD optimizer with CosineAnnealing learning rate decay and MSEloss to train the model.

### Data Used

This graph shows the candidate variables we explored for our model.

![Input Variables](./images/variables_used.PNG)

We used daily S&P500 data between 2008/07/02 and 2016/09/30. This dataset is from our referenced paper [A deep learning framework for financial time series using stacked autoencoders and long-short term memory]({%https://www.researchgate.net/publication/318991900_A_deep_learning_framework_for_financial_time_series_using_stacked_autoencoders_and_long-short_term_memory%})

### Expriment  Method

All 3 models are constructed to tackle the same task. They take 30 days of input varaible data and predict the close price daily percentage move of the next day. We keep the task same to compare the performance of three model studctures. We separated the data into three sets - 80% for training set, 10% for validation set and 10% for testing set. training set and validation set are generated by randomly split, and test set contains the data from specific date. 

We applied subsection prediction method \cite{subsection_method}. This method has three parts. Firstly, we used training set to train the model. Secondly, we used the validation set to verify the optimal model setting. Lastly, we used testing set to measure the performance of the model. We used 3 classical performance metrics for financial time series data - MAPE, R and Theil U


### Results & Next Step
Performance Metrics Comparison Table

|    | MAPE | R | Theil U |
| -- | ---  | - | ------- |
| RNN*  | 0.018  | 0.847 | 0.847|
| WSAEs-LSTM*  | 0.011  | 0.946 | 0.007
| LSTM Encoder Decoder             | 0.011  | 0.872 | 0.008 |
| LSTM Encoder Decoder + Attention | 0.0075  | 0.947 | 0.005 |
| Transformer                      | 0.0055  | 0.980 | 0.0038 |

* models are models from our referenced paper [A deep learning framework for financial time series using stacked autoencoders and long-short term memory]({%https://www.researchgate.net/publication/318991900_A_deep_learning_framework_for_financial_time_series_using_stacked_autoencoders_and_long-short_term_memory%})

Daily Close Price Predictions Comparison Graph
![comparison graph](./images/comparison.jpg)

The three performance metrics and the graph match with each other and we conclude  that  among  the  three  models,  transformer model performs the best in predicting S&P500 index daily movement in our experiment setting. Although this doesnot mean transformer will always perform the best in any financial  time-series  prediction  tasks  (eg.   different  data  freqeuncy, input variables), the results still indicate that transformer has a high potential to be applied in financial time-series prediction tasks and should be payed more attention to in future researches on financial time series. We also plan to see whether this conclusion hold for other financial products and possibly further develop our Transformer model to generate daily adjusted investment strategy.
