import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
from pandas import Timestamp


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import BlackLittermanModel
from pypfopt import risk_models
from pypfopt import expected_returns

# Title of the app
st.title("Portfolio Return Analysis")

# Sidebar for user inputs
st.sidebar.header("Input Portfolio Details")

# User input for asset mix
asset_mix = {}
st.sidebar.subheader("Asset Allocation")
num_assets = st.sidebar.number_input("Number of assets in your portfolio", min_value=1, value=3, step=1)

for i in range(num_assets):
    ticker = st.sidebar.text_input(f"Asset {i+1} Ticker (e.g., AAPL, MSFT)", key=f"ticker_{i}")
    amount = st.sidebar.number_input(f"Investment Amount in {ticker} ($)", min_value=0, value=1000, step=100, key=f"amount_{i}")
    if ticker and amount:
        asset_mix[ticker] = amount

# Input start and end date
start_date = st.sidebar.date_input("Start Date", datetime(2014, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2023, 1, 1))

# Input benchmark ticker
benchmark = st.sidebar.text_input("Benchmark Ticker (e.g., SPY)", value="SPY")

# Button to trigger the portfolio return calculation
if st.sidebar.button("Calculate Portfolio Returns"):

    def adjust_dates(ticker, start_date, end_date):
      
      # Convert start_date and end_date to pandas Timestamps
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Download historical data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        first_valid_index = data['Adj Close'].first_valid_index()
        # last_valid_index = data['Adj Close'].last_valid_index()

        if first_valid_index is None:
            st.error(f"No data available for {ticker}. Please check the ticker symbol.")
            return start_date, end_date, False

        # Compare and adjust dates
        if start_date < first_valid_index:
            st.warning(f"Adjusted start date for {ticker} to {first_valid_index.date()} due to data availability.")
            start_date = first_valid_index

        # if end_date > last_valid_index:
        #     st.warning(f"Adjusted end date for {ticker} to {last_valid_index.date()} due to data availability.")
        #     end_date = last_valid_index

        return start_date, end_date, True 
   
    # Function to calculate portfolio returns and benchmark comparison
    def portfolio_ret(asset_mix, start_date, end_date, benchmark):

        # Adjust dates for all assets in the portfolio
        for ticker in asset_mix.keys():
            start_date, end_date, valid = adjust_dates(ticker, start_date, end_date)
            if not valid:
                return

        data = yf.download(list(asset_mix.keys()), start=start_date, end=end_date)
        if isinstance(data.columns, pd.MultiIndex):
            missing_data_tickers = []
            for ticker in asset_mix.keys():
                first_valid_index = data['Adj Close'][ticker].first_valid_index()
                if first_valid_index is None or first_valid_index.strftime('%Y-%m-%d') > str(start_date):
                    missing_data_tickers.append(ticker)

            if missing_data_tickers:
                st.error(f"No data available for the following tickers starting from {start_date}: {', '.join(missing_data_tickers)}")
                return
        else:
            first_valid_index = data['Adj Close'].first_valid_index()
            if first_valid_index is None or first_valid_index.strftime('%Y-%m-%d') > str(start_date):
                st.error(f"No data available for {list(asset_mix.keys())[0]} starting from {start_date}")
                return

        # Portfolio return calculation
        total_portfolio_amt = sum(asset_mix.values())
        weight = {ticker: value / total_portfolio_amt for ticker, value in asset_mix.items()}
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Adj Close'].fillna(data['Close'])

        if len(weight) > 1:
            weights = list(weight.values())
            weight_ret = data.pct_change().mul(weights, axis=1)
            portfo_ret = weight_ret.sum(axis=1)
        else:
            weights = [1]
            data = data['Adj Close'].fillna(data['Close'])
            portfo_ret = data.pct_change()

        # Benchmark calculation
        bench_mark = yf.download(benchmark, start=start_date, end=end_date)
        bench_mark = bench_mark['Adj Close'].fillna(bench_mark['Close'])
        bench_mark_ret = bench_mark.pct_change()

        # Visualizations
        st.plotly_chart(go.Figure(data=[go.Pie(labels=list(asset_mix.keys()),
                                                 values=list(asset_mix.values()),
                                                 hole=.65,
                                                 marker=dict(colors=px.colors.qualitative.G10))]),
                                     use_container_width=True)

        fig2 = portfolio_vs_benchmark(portfo_ret, bench_mark_ret)

        if len(weights) > 1:
            fig1 = perform_portfo_analysis(data, weight)
            st.plotly_chart(fig1, use_container_width=True)

        st.plotly_chart(fig2, use_container_width=True)

    def portfolio_vs_benchmark(portfo_ret, bench_mark_ret):
        port_cum_ret = (((portfo_ret + 1).cumprod() - 1) * 100).round(2)
        bench_cum_ret = (((bench_mark_ret + 1).cumprod() - 1) * 100).round(2)

        portfo_vol = portfo_ret.std() * np.sqrt(252).round(2)
        bench_vol = bench_mark_ret.std() * np.sqrt(252).round(2)

        excess_port_ret = portfo_ret - 0.01 / 252
        portfo_sharpe = (excess_port_ret.mean() / portfo_ret.std()) * np.sqrt(252).round(2)

        excess_bench_ret = bench_mark_ret - 0.01 / 252
        bench_sharpe = (excess_bench_ret.mean() / bench_mark_ret.std()) * np.sqrt(252).round(2)

        fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                             column_titles=['Cumulative returns', 'Portfolio Risk-Reward'],
                             column_widths=[0.5, 0.5],
                             shared_xaxes=False, shared_yaxes=False)

        fig2.add_trace(go.Scatter(x=port_cum_ret.index, y=port_cum_ret, mode='lines', name='Portfolio', showlegend=False,
                                   hovertemplate='%{y:.2f}%'), row=1, col=1)

        fig2.add_trace(go.Scatter(x=bench_cum_ret.index, y=bench_cum_ret, mode='lines', name='Benchmark', showlegend=False,
                                   hovertemplate='%{y:.2f}%'), row=1, col=1)

        fig2.add_trace(go.Scatter(x=[portfo_vol, bench_vol],
                                   y=[port_cum_ret.iloc[-1], bench_cum_ret.iloc[-1]],
                                   mode='markers+text',
                                   name='Returns',
                                   marker=dict(size=75, color=[portfo_sharpe, bench_sharpe],
                                               colorscale='bluered',
                                               colorbar=dict(title='Sharpe Ratio'),
                                               showscale=True),
                                   text=['Portfolio', 'Benchmark'],
                                   textposition='middle center',
                                   textfont=dict(size=12, color='white'),
                                   hovertemplate='%{y:.2f}%<br>Annualized Volatility:%{x:.2f}%<br>Sharpe Ratio: %{marker.color:.2f}',
                                   showlegend=False),
                                   row=1, col=2)

        fig2.update_layout(title="Portfolio vs Benchmark", xaxis_title='Date', yaxis_title='Cumulative Returns (%)', hovermode='x unified')
        return fig2

    def perform_portfo_analysis(data, weights):
        indi_cumsum = pd.DataFrame()
        indi_vol = pd.Series(dtype=float)
        indi_sharpe = pd.Series(dtype=float)

        for ticker, weight in weights.items():
            if ticker in data.columns:
                indi_cumsum[ticker] = ((data[ticker].pct_change() + 1).cumprod() - 1)
                indi_vol[ticker] = data[ticker].pct_change().std() * np.sqrt(252) * 100
                excess_ret = data[ticker].pct_change() - 0.01 / 252
                indi_sharpe[ticker] = (excess_ret.mean() / data[ticker].pct_change().std()) * np.sqrt(252).round(2)

        fig1 = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                             column_titles=['Historical Performance Assets', 'Risk-Reward'],
                             column_widths=[0.5, 0.5],
                             shared_xaxes=False, shared_yaxes=False)

        for ticker in indi_cumsum.columns:
            fig1.add_trace(go.Scatter(x=indi_cumsum.index, y=indi_cumsum[ticker],
                                       mode='lines', name=ticker, showlegend=False,
                                       hovertemplate='%{y:.2f}%'), row=1, col=1)

        sharpe_color = [indi_sharpe[ticker] for ticker in indi_cumsum.columns]

        fig1.add_trace(go.Scatter(x=indi_vol.tolist(), y=indi_cumsum.iloc[-1].tolist(),
                                   mode='markers+text',
                                   marker=dict(size=75, color=sharpe_color,
                                               colorscale='Bluered_r',
                                               colorbar=dict(title='Sharpe Ratio'),
                                               showscale=True),
                                   name='Returns',
                                   text=indi_cumsum.columns.tolist(),
                                   textfont=dict(color='white'),
                                   showlegend=False,
                                   hovertemplate='%{y:.2f}%<br>Annualized volatility: %{x:.2f}%<br>Sharpe Ratio: %{marker.color:.2f}',
                                   textposition='middle center'),
                          row=1, col=2)

        fig1.update_yaxes(title_text='Returns (%)', col=1)
        fig1.update_yaxes(title_text='Returns (%)', col=2)
        fig1.update_xaxes(title_text='Date', col=1)
        fig1.update_xaxes(title_text='Annualized Volatility (%)', col=2)

        fig1.update_layout(title="Portfolio Analysis", hovermode='x unified')


        return fig1
     # Run the portfolio function
    portfolio_ret(asset_mix, start_date, end_date, benchmark)

# Function to calculate and visualize portfolio performance
if st.sidebar.button("Markowitz Portfolio Optimization"):

    st.title("Markowitz Mean_Variance Optimized Allocation")
    # Function to calculate portfolio returns and benchmark comparison
    def mar_mean_var(asset_mix, start_date, end_date, benchmark):

        if len(list(asset_mix.keys())) < 2:
          st.error("You must have at least two assets in your portfolio")
          st.write(len(list(asset_mix.keys())))

        else:
          # Download data for the given asset mix
          data = yf.download(list(asset_mix.keys()), start=start_date, end=end_date)['Adj Close']


          # Check for missing data in downloaded assets
          missing_data_tickers = [ticker for ticker in asset_mix.keys() if ticker not in data.columns]
          if missing_data_tickers:
              st.error(f"Missing data for: {', '.join(missing_data_tickers)}")
              return

          # Calculate portfolio returns
          total_portfolio_amt = sum(asset_mix.values())
          weights = {ticker: value / total_portfolio_amt for ticker, value in asset_mix.items()}

          # If we have more than one asset, proceed with weighting
          if len(weights) > 1:
              weight_values = list(weights.values())
              weight_ret = data.pct_change().mul(weight_values, axis=1)
              portfo_ret = weight_ret.sum(axis=1)
          else:
              portfo_ret = data.pct_change()

          # Download benchmark data
          bench_mark = yf.download(benchmark, start=start_date, end=end_date)['Adj Close']
          bench_mark_ret = bench_mark.pct_change()

          # Markowitz portfolio optimization
          mu = expected_returns.mean_historical_return(data)
          S = risk_models.sample_cov(data)
          ef = EfficientFrontier(mu, S)
          optimal_weights = ef.max_sharpe()
          cleaned_weights = ef.clean_weights()


          # Re-calculate optimized portfolio returns
          optimized_portfolio = pd.Series(cleaned_weights).values
          optimized_ret = (data.pct_change().mul(optimized_portfolio, axis=1)).sum(axis=1)

          # Visualization of initial and optimized portfolio allocation
          fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                              specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                              column_titles=['Initial Portfolio Allocation', 'Optimized Portfolio Allocation'],
                              column_widths=[0.5, 0.5])

          # Initial portfolio pie chart
          fig.add_trace(go.Pie(labels=list(asset_mix.keys()),
                              values=list(asset_mix.values()),
                              hole=.65,
                              name = 'Initial Portfolio',
                              marker=dict(colors=px.colors.qualitative.G10)),
                        row=1, col=1)

          # Optimized portfolio pie chart
          new_investment = total_portfolio_amt * np.array(list(cleaned_weights.values()))
          fig.add_trace(go.Pie(labels=list(cleaned_weights.keys()),
                              values=new_investment,
                              hole=.65,
                              name = 'Optimized Portfolio',
                              marker=dict(colors=px.colors.qualitative.G10)),
                        row=1, col=2)

          st.plotly_chart(fig, use_container_width=True)

          # Plot portfolio vs optimized portfolio vs benchmark
          fig2 = portfolio_vs_optp_benchmark(portfo_ret, bench_mark_ret, optimized_ret)
          st.plotly_chart(fig2, use_container_width=True)

    # Function to compare portfolio, optimized portfolio, and benchmark
    def portfolio_vs_optp_benchmark(portfo_ret, bench_mark_ret, optimized_ret):
        port_cum_ret = (((portfo_ret + 1).cumprod() - 1) * 100).round(2)
        bench_cum_ret = (((bench_mark_ret + 1).cumprod() - 1) * 100).round(2)
        opt_cum_ret = (((optimized_ret + 1).cumprod() - 1) * 100).round(2)

        # Calculate volatilities and Sharpe ratios
        portfo_vol = portfo_ret.std() * np.sqrt(252)
        bench_vol = bench_mark_ret.std() * np.sqrt(252)
        opt_vol = optimized_ret.std() * np.sqrt(252)

        excess_port_ret = portfo_ret - 0.01 / 252
        portfo_sharpe = (excess_port_ret.mean() / portfo_ret.std()) * np.sqrt(252)

        excess_bench_ret = bench_mark_ret - 0.01 / 252
        bench_sharpe = (excess_bench_ret.mean() / bench_mark_ret.std()) * np.sqrt(252)

        excess_opt_ret = optimized_ret - 0.01 / 252
        opt_sharpe = (excess_opt_ret.mean() / optimized_ret.std()) * np.sqrt(252)

        # Create subplots for cumulative returns and risk-reward comparison
        fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                             column_titles=['Cumulative returns', 'Portfolio Risk-Reward'],
                             column_widths=[0.5, 0.5],
                             shared_xaxes=False, shared_yaxes=False)

        # Cumulative returns
        fig2.add_trace(go.Scatter(x=port_cum_ret.index, y=port_cum_ret,
                                  mode='lines', name='Portfolio',
                                  hovertemplate='%{y:.2f}%', showlegend = False),
                       row=1, col=1)

        fig2.add_trace(go.Scatter(x=opt_cum_ret.index, y=opt_cum_ret,
                                  mode='lines', name='Optimized Portfolio',
                                  hovertemplate='%{y:.2f}%', showlegend = False),
                       row=1, col=1)

        fig2.add_trace(go.Scatter(x=bench_cum_ret.index, y=bench_cum_ret,
                                  mode='lines', name='Benchmark',
                                  hovertemplate='%{y:.2f}%', showlegend = False),
                       row=1, col=1)

        # Risk-Reward comparison
        fig2.add_trace(go.Scatter(x=[portfo_vol, opt_vol, bench_vol],
                                  y=[port_cum_ret.iloc[-1], opt_cum_ret.iloc[-1], bench_cum_ret.iloc[-1]],
                                  mode='markers+text',
                                  name='Returns',
                                  text=['Portfolio', 'Optimized Portfolio', 'Benchmark'],
                                  textposition='middle center',
                                  textfont=dict(size=12, color='white'),
                                  marker=dict(size=75, color=[portfo_sharpe, opt_sharpe, bench_sharpe],
                                              colorscale='Bluered',
                                              colorbar=dict(title='Sharpe Ratio'),
                                              showscale=True),
                                  hovertemplate='%{y:.2f}%<br>Volatility: %{x:.2f}%<br>Sharpe: %{marker.color:.2f}',
                                  showlegend = False),
                       row=1, col=2)

        fig2.update_layout(title="Cumulative Returns & Risk-Reward Comparison",xaxis_title='Date', yaxis_title='Cumulative Returns (%)', hovermode='x unified')
        return fig2

    # Run the Markowitz portfolio function
    mar_mean_var(asset_mix, start_date, end_date, benchmark)


# App Footer
st.sidebar.markdown("### Developed by SON_OF_IWUOHA")

