from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import plotly.graph_objects as go
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect

import os
from functools import wraps


# Fix the indentation and order o
app = Flask(__name__)

# SET SECRET KEY FIRST
app.config['SECRET_KEY'] = 'my_super_secret_key_123'

# THEN enable CSRF
csrf = CSRFProtect(app)



# NSE stock list
stock_list = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS","SBIN.NS", "AXISBANK.NS", "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS","LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "ULTRACEMCO.NS", "TITAN.NS", "ONGC.NS", "POWERGRID.NS",
    "BHARTIARTL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANITRANS.NS",
    "DMART.NS", "PIDILITIND.NS", "SUNPHARMA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "BAJAJFINSV.NS", "M&M.NS", "SBILIFE.NS", "ICICIPRULI.NS",
    "HDFCLIFE.NS", "SHREECEM.NS", "GRASIM.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "VEDL.NS", "COALINDIA.NS", "NTPC.NS", "BPCL.NS", "IOC.NS",
    "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "TVSMOTOR.NS", "MOTHERSUMI.NS", "BOSCHLTD.NS",
    "APOLLOHOSP.NS", "CIPLA.NS", "TORNTPHARM.NS", "BIOCON.NS", "LUPIN.NS",
    "AUROPHARMA.NS", "GLAXO.NS", "GSKCONS.NS", "HAVELLS.NS", "CROMPTON.NS",
    "JSPL.NS", "TATAMOTORS.NS", "TECHM.NS", "ZYDUSLIFE.NS", "ABB.NS",
    "SIEMENS.NS", "BEL.NS", "HAL.NS", "BHEL.NS", "IRCTC.NS",
    "CONCOR.NS", "GMRINFRA.NS", "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS",
    "UBL.NS", "MCDOWELL-N.NS", "MARICO.NS", "BRITANNIA.NS", "COLPAL.NS",
    "DABUR.NS", "EMAMILTD.NS", "PGHH.NS", "NESTLEIND.NS", "TRENT.NS",
    "PEL.NS", "IDFCFIRSTB.NS", "PNB.NS", "CANBK.NS", "BANKBARODA.NS",
    "FEDERALBNK.NS", "INDUSINDBK.NS", "YESBANK.NS", "IDBI.NS", "RBLBANK.NS",
    "AMBUJACEM.NS", "ACC.NS", "RAMCOCEM.NS", "JKCEMENT.NS", "INDIGO.NS",
    "SPICEJET.NS", "INTERGLOBE.NS", "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS"
]

# -------------------- Helper Functions --------------------
def fetch_stock_data(symbol, period="6mo"):
    try:
        data = yf.Ticker(symbol).history(period=period)
        if data.empty:
            app.logger.warning(f"[fetch_stock_data] No data returned for {symbol}")
            return None
        return data
    except Exception as e:
        app.logger.error(f"[fetch_stock_data] Error fetching data for {symbol}: {str(e)}")
        return None

def preprocess_data(data):
    try:
        df = data[['Close']].dropna().copy()
        scaler = MinMaxScaler()
        df['Scaled_Close'] = scaler.fit_transform(df[['Close']])
        return df, scaler
    except Exception as e:
        app.logger.error(f"[preprocess_data] Error: {str(e)}")
        raise

def prepare_lstm_input(data, time_steps=60):
    try:
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    except Exception as e:
        app.logger.error(f"[prepare_lstm_input] Error: {str(e)}")
        raise

def build_lstm_model(input_shape):
    try:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        app.logger.error(f"[build_lstm_model] Error: {str(e)}")
        raise

def create_candlestick_chart(data):
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index[-30:], 
            open=data['Open'][-30:],
            high=data['High'][-30:], 
            low=data['Low'][-30:], 
            close=data['Close'][-30:]
        )])
        fig.update_layout(
            title="Candlestick Chart (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )
        return fig.to_html(full_html=False)
    except Exception as e:
        app.logger.error(f"[create_candlestick_chart] Error: {str(e)}")
        return None

def create_historical_chart(data):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['Close'],
            mode="lines", 
            name="Close Price",
            line=dict(color="blue")
        ))
        fig.update_layout(
            title="Historical Price",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )
        return fig.to_html(full_html=False)
    except Exception as e:
        app.logger.error(f"[create_historical_chart] Error: {str(e)}")
        return None

def create_volume_chart(data):
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data.index, 
            y=data['Volume'],
            name="Volume",
            marker=dict(color="orange")
        ))
        fig.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_dark"
        )
        return fig.to_html(full_html=False)
    except Exception as e:
        app.logger.error(f"[create_volume_chart] Error: {str(e)}")
        return None

def create_moving_avg_chart(data):
    try:
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['Close'],
            mode="lines", 
            name="Close Price",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['SMA_50'],
            mode="lines", 
            name="50-Day SMA",
            line=dict(color="red")
        ))
        fig.update_layout(
            title="Moving Average",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )
        return fig.to_html(full_html=False)
    except Exception as e:
        app.logger.error(f"[create_moving_avg_chart] Error: {str(e)}")
        return None

def create_prediction_chart(dates, values):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, 
            y=values.flatten(),
            mode="lines+markers",
            name="Predicted Price",
            line=dict(color="green")
        ))
        fig.update_layout(
            title="Future Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )
        return fig.to_html(full_html=False)
    except Exception as e:
        app.logger.error(f"[create_prediction_chart] Error: {str(e)}")
        return None

# -------------------- Decorators -------------------
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            flash('Please log in to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# -------------------- Form Classes --------------------
class CalculatorForm(FlaskForm):
    symbol = StringField('Stock Symbol', validators=[DataRequired()])
    operation = SelectField('Operation', choices=[
        ('cagr', 'Compound Annual Growth Rate (CAGR)'),
        ('returns', 'Simple Returns'),
        ('volatility', 'Volatility'),
        ('moving_avg', 'Moving Averages'),
        ('sharpe', 'Sharpe Ratio'),
        ('max_drawdown', 'Maximum Drawdown')
    ], validators=[DataRequired()])
    years = FloatField('Years (for CAGR)', validators=[NumberRange(min=0.1)])
    timeframe = SelectField('Timeframe (for Returns)', choices=[
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly')
    ])
    short_window = FloatField('Short Window (days)', default=50)
    long_window = FloatField('Long Window (days)', default=200)
    submit = SubmitField('Calculate')


@app.route('/')
def home():
    return render_template('index.html', 
                         stocks=stock_list, 
                         selected_stock=None, 
                         username=session.get('username'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    selected_stock = None
    charts = {
        'candlestick': None,
        'prediction': None,
        'historical': None,
        'volume': None,
        'moving_avg': None
    }
    predictions = []
    error_message = None
    current_price = 0.0

    if request.method == 'POST':
        selected_stock = request.form.get('stock')
        prediction_days = int(request.form.get('days', 10))

        try:
            data = fetch_stock_data(selected_stock, period="6mo")
            app.logger.info(f"[predict] Data shape for {selected_stock}: {None if data is None else data.shape}")
            if data is None or data.empty:
                error_message = f"No data available for the selected stock ({selected_stock})."
            elif len(data) < 60:
                error_message = f"Not enough data (only {len(data)} points). Need at least 60 data points."
            else:
                current_price = round(data['Close'].iloc[-1], 2) if 'Close' in data and len(data['Close']) > 0 else 0.0
                charts['candlestick'] = create_candlestick_chart(data)
                charts['historical'] = create_historical_chart(data)
                charts['volume'] = create_volume_chart(data)
                charts['moving_avg'] = create_moving_avg_chart(data)
                processed_data, scaler = preprocess_data(data)
                scaled_close = processed_data['Scaled_Close'].values
                time_steps = 60
                
                X, y = prepare_lstm_input(scaled_close, time_steps)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                model = build_lstm_model((X.shape[1], 1))
                model.fit(X, y, epochs=5, batch_size=1, verbose=0)
                future_input = scaled_close[-time_steps:]
                predictions_scaled = []
                for _ in range(prediction_days):
                    future_input = future_input.reshape((1, time_steps, 1))
                    pred = model.predict(future_input, verbose=0)[0][0]
                    predictions_scaled.append(pred)
                    future_input = np.append(future_input[0][1:], pred)
                predictions_rescaled = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
                prediction_dates = [(datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d") 
                                 for i in range(prediction_days)]
                predictions = list(zip(prediction_dates, predictions_rescaled.flatten()))
                charts['prediction'] = create_prediction_chart(prediction_dates, predictions_rescaled)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            app.logger.error(f"[predict] Error in prediction: {str(e)}")

    return render_template("index.html",
        username=session.get('username'),
        stocks=stock_list,
        selected_stock=selected_stock,
        current_price=current_price,
        predictions=predictions,
        error_message=error_message,
        candlestick=charts['candlestick'],
        historical=charts['historical'],
        volume=charts['volume'],
        moving_avg=charts['moving_avg'],
        prediction=charts['prediction']
    )

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    form = CalculatorForm()
    operations = {
        'cagr': ('Compound Annual Growth Rate (CAGR)', 'Calculate growth rate over time'),
        'returns': ('Simple Returns', 'Calculate daily/weekly/monthly returns'),
        'volatility': ('Volatility', 'Measure price fluctuations'),
        'moving_avg': ('Moving Averages', 'Calculate and visualize moving averages'),
        'sharpe': ('Sharpe Ratio', 'Risk-adjusted return metric'),
        'max_drawdown': ('Maximum Drawdown', 'Worst peak-to-trough decline')
    }
    
    result = None
    chart = None
    calculation_details = None
    
    if form.validate_on_submit():
        symbol = form.symbol.data
        operation = form.operation.data
        
        try:
            period_map = {
                'cagr': '10y',
                'returns': '1y',
                'volatility': '1y',
                'moving_avg': '1y',
                'sharpe': '5y',
                'max_drawdown': '10y'
            }
            period = period_map.get(operation, '1y')
            stock_data = yf.Ticker(symbol).history(period=period)
            
            if stock_data.empty:
                flash(f"No data available for {symbol}", 'warning')
            else:
                if operation == 'cagr':
                    years = form.years.data if form.years.data else 5
                    start_price = stock_data['Close'].iloc[0]
                    end_price = stock_data['Close'].iloc[-1]
                    cagr = ((end_price / start_price) ** (1/years) - 1) * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index, 
                        y=stock_data['Close'],
                        name='Price',
                        line=dict(color='blue'))
                    )
                    fig.add_trace(go.Scatter(
                        x=[stock_data.index[0], stock_data.index[-1]],
                        y=[start_price, end_price],
                        name='CAGR Line',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title=f"{symbol} CAGR Calculation ({years} years)",
                        yaxis_title="Price",
                        template="plotly_white"
                    )
                    
                    result = f"{cagr:.2f}%"
                    calculation_details = {
                        'Start Price': f"₹{start_price:.2f}",
                        'End Price': f"₹{end_price:.2f}",
                        'Period': f"{years} years"
                    }
                    chart = fig.to_html(full_html=False)
                
                elif operation == 'returns':
                    timeframe = form.timeframe.data if form.timeframe.data else 'daily'
                    if timeframe == 'daily':
                        returns = stock_data['Close'].pct_change().dropna()
                    elif timeframe == 'weekly':
                        returns = stock_data['Close'].resample('W').last().pct_change().dropna()
                    else:  # monthly
                        returns = stock_data['Close'].resample('M').last().pct_change().dropna()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=returns*100,
                        name='Returns Distribution',
                        marker_color='green',
                        opacity=0.75
                    ))
                    fig.update_layout(
                        title=f"{symbol} {timeframe.capitalize()} Returns Distribution",
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        template="plotly_white"
                    )
                    
                    stats = returns.describe()
                    result = f"Mean: {stats['mean']*100:.2f}%"
                    calculation_details = {
                        'Timeframe': timeframe.capitalize(),
                        'Positive Days': f"{(returns > 0).mean()*100:.1f}%",
                        'Negative Days': f"{(returns < 0).mean()*100:.1f}%",
                        'Best Day': f"{returns.max()*100:.2f}%",
                        'Worst Day': f"{returns.min()*100:.2f}%"
                    }
                    chart = fig.to_html(full_html=False)
                
                elif operation == 'volatility':
                    returns = stock_data['Close'].pct_change().dropna()
                    daily_vol = returns.std()
                    annual_vol = daily_vol * np.sqrt(252)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        name='Price',
                        line=dict(color='blue'))
                    )
                    fig.update_layout(
                        title=f"{symbol} Price Volatility",
                        yaxis_title="Price",
                        template="plotly_white"
                    )
                    
                    result = f"{annual_vol*100:.2f}%"
                    calculation_details = {
                        'Daily Volatility': f"{daily_vol*100:.2f}%",
                        'Annualized Volatility': f"{annual_vol*100:.2f}%",
                        'Data Points': len(returns)
                    }
                    chart = fig.to_html(full_html=False)
                
                elif operation == 'moving_avg':
                    short_window = int(form.short_window.data) if form.short_window.data else 50
                    long_window = int(form.long_window.data) if form.long_window.data else 200
                    
                    stock_data['SMA_50'] = stock_data['Close'].rolling(short_window).mean()
                    stock_data['SMA_200'] = stock_data['Close'].rolling(long_window).mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        name='Price',
                        line=dict(color='blue'))
                    )
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['SMA_50'],
                        name=f'{short_window}-day MA',
                        line=dict(color='orange'))
                    )
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['SMA_200'],
                        name=f'{long_window}-day MA',
                        line=dict(color='red'))
                    )
                    fig.update_layout(
                        title=f"{symbol} Moving Averages",
                        yaxis_title="Price",
                        template="plotly_white"
                    )
                    
                    current_price = stock_data['Close'].iloc[-1]
                    ma50 = stock_data['SMA_50'].iloc[-1]
                    ma200 = stock_data['SMA_200'].iloc[-1]
                    
                    result = "Golden Cross" if ma50 > ma200 else "Death Cross"
                    calculation_details = {
                        'Current Price': f"₹{current_price:.2f}",
                        f'{short_window}-day MA': f"₹{ma50:.2f}",
                        f'{long_window}-day MA': f"₹{ma200:.2f}",
                        'Position': "Above MA" if current_price > ma50 else "Below MA"
                    }
                    chart = fig.to_html(full_html=False)
                
                elif operation == 'sharpe':
                    returns = stock_data['Close'].pct_change().dropna()
                    risk_free_rate = 0.05  # Assume 5% risk-free rate
                    excess_returns = returns - (risk_free_rate/252)
                    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
                    
                    result = f"{sharpe_ratio:.2f}"
                    calculation_details = {
                        'Annualized Return': f"{returns.mean()*252*100:.2f}%",
                        'Annualized Volatility': f"{returns.std()*np.sqrt(252)*100:.2f}%",
                        'Risk-Free Rate': f"{risk_free_rate*100:.2f}%"
                    }
                
                elif operation == 'max_drawdown':
                    cumulative = (1 + stock_data['Close'].pct_change()).cumprod()
                    peak = cumulative.expanding(min_periods=1).max()
                    drawdown = (cumulative/peak) - 1
                    max_drawdown = drawdown.min()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=drawdown*100,
                        name='Drawdown',
                        fill='tozeroy',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title=f"{symbol} Drawdown Analysis",
                        yaxis_title="Drawdown (%)",
                        template="plotly_white"
                    )
                    
                    result = f"{max_drawdown*100:.2f}%"
                    calculation_details = {
                        'Peak Date': peak.idxmax().strftime('%Y-%m-%d'),
                        'Trough Date': drawdown.idxmin().strftime('%Y-%m-%d'),
                        'Recovery Days': "N/A"
                    }
                    chart = fig.to_html(full_html=False)
                
                flash('Calculation completed successfully', 'success')
        
        except Exception as e:
            flash(f'Error performing calculation: {str(e)}', 'danger')
            app.logger.error(f"Calculator error: {str(e)}")
    
    return render_template('calculator.html',
                         username=session.get('username'),
                         stock_list=stock_list,
                         operations=operations,
                         form=form,
                         result=result,
                         chart=chart,
                         calculation_details=calculation_details)

@app.route('/debug/stocks')
def debug_stocks():
    return {'stocks': stock_list}

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/support')
def support():
    return render_template('support.html', username=session.get('username'))

if __name__=="__main__":
    app.run(debug=True)
