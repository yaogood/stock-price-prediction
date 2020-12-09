import os
import plotly.graph_objects as go
import plotly.express as px 

def plot_finance(df, dir):
    plot_candlestick(df, dir)
    plot_timeseries(df, dir, ['Volume'])
    plot_timeseries(df, dir, ['MACD'])
    plot_timeseries(df, dir, ['CCI'])
    plot_timeseries(df, dir, ['ATR'])
    plot_timeseries(df, dir, ['BOLL'])
    plot_timeseries(df, dir, ['EMA20'])
    plot_timeseries(df, dir, ['MA5', 'MA10'])
    plot_timeseries(df, dir, ['MTM6', 'MTM12'])
    plot_timeseries(df, dir, ['ROC'])
    plot_timeseries(df, dir, ['SMI'])
    plot_timeseries(df, dir, ['WVAD'])
    plot_timeseries(df, dir, ['US_Dollar_Index'])
    plot_timeseries(df, dir, ['Federal_Fund_Rate'])


def plot_candlestick(df, dir=None): # dataframe must contain 
    fig = go.Figure(data=[go.Candlestick(x=df['Ntime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.update_layout(
        title = 'Index price of years from 2008 to 2016',
        xaxis_title="date",
        yaxis_title='price',
    )
    fig.write_html(file=os.path.join(dir, 'candle_stick.html'), default_width='60%', default_height='60%')


def plot_timeseries(df,  dir=None, labels=['']):
    fig = go.Figure()
    for label in labels:
        fig.add_trace(go.Scatter(x=df['Ntime'], y=df[label], name=label))

    y_title = labels[0]
    if len(labels) > 1:
        y_title = labels[0][:-1]
    fig.update_layout(
        title = y_title + ' values of years from 2008 to 2016',
        xaxis_title="date",
        yaxis_title=y_title,
        # legend_title="Legend Title",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )

    fig.write_html(file=os.path.join(dir, labels[0]+'.html'), default_width='60%', default_height='60%')
    # fig.show()


