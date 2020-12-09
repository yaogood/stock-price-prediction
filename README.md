## Welcome to our Project Page!

We conducted this research to compare how popular models in Natural Language Processing tasks perform in forecasting financial time series! 

The three models we compared are:

1. Encoder-Decoder LSTM
2. Encoder-Attention-Decoder LSTM
3. Transformer

Here is our [GitHub Repository](https://github.com/yaogood/stock-price-prediction)

### Model Tasks

All 3 models are constructed to tackle the same task.

They take 30 days of input varaible data and predict the close price daily percentage move of the next day. If this task could be well conducted, trading strategy with daily adjustment could be constructed from it.

We keep the task same to compare the performance of three model studctures.

### Data Used

This graph shows the candidate variables we explored for our model

![Input Variables](./images/variables_used.PNG)


### Model Details


### Results

Result Table

|    | MAPE | R | Theil U |
| -- | ---  | - | ------- |
| LSTM Encoder Decoder             | Content Cell  | |
| LSTM Encoder Decoder + Attention | Content Cell  | |
| Transformer                      |               | |

