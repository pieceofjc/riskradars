import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pykrx import stock
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class AdvancedStockPredictor:
    def __init__(self, code):
        self.code = code
        self.name = stock.get_market_ticker_name(code)
        self.df = None
        self.df_monthly = None
        self.returns = None
        self.volatility = None
        
    def load_data(self, start_date="2022-06-30", end_date=None):
        """데이터 로드 및 전처리"""
        if end_date is None:
            end_date = pd.Timestamp.today().strftime("%Y%m%d")
        
        print(f"[데이터 로딩] {self.name} ({self.code})")
        self.df = stock.get_market_ohlcv_by_date(start_date, end_date, self.code)
        
        if self.df.empty:
            raise ValueError(f"{self.name} ({self.code}) 데이터가 없습니다.")
        
        self.df.index = pd.to_datetime(self.df.index)
        self.df_monthly = self.df['종가'].resample('M').mean().interpolate()
        self.returns = self.df_monthly.pct_change().dropna()
        self.volatility = self.returns.rolling(window=12).std() * np.sqrt(12)
    
    def detect_regime_changes(self):
        """주가 추세 변화 감지"""
        ma_short = self.df_monthly.rolling(window=6).mean()
        ma_long = self.df_monthly.rolling(window=12).mean()
        trend_signal = np.where(ma_short > ma_long, 1, -1)
        trend_changes = np.where(np.diff(trend_signal) != 0)[0]
        return trend_signal, trend_changes, ma_short, ma_long
    
    def calculate_downside_risk(self):
        """하방 위험 계산"""
        negative_returns = self.returns[self.returns < 0]
        var_95 = np.percentile(negative_returns, 5) if len(negative_returns) > 0 else 0
        cumulative_returns = (1 + self.returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return var_95, max_drawdown, drawdown
    
    def arima_with_volatility_adjustment(self, steps=36):
        """변동성 조정 ARIMA 예측"""
        model = ARIMA(self.df_monthly, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        
        current_volatility = self.volatility.iloc[-1]
        historical_volatility = self.volatility.mean()
        volatility_adjustment = current_volatility / historical_volatility
        
        adjusted_forecast = forecast.copy()
        if volatility_adjustment > 1.2:
            decline_factor = np.linspace(1.0, 0.85, steps)
            adjusted_forecast = forecast * decline_factor
        
        return forecast, adjusted_forecast
    
    def monte_carlo_simulation(self, steps=36, simulations=1000):
        """몬테카를로 시뮬레이션"""
        current_price = self.df_monthly.iloc[-1]
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        simulation_results = np.zeros((simulations, steps))
        
        for i in range(simulations):
            prices = [current_price]
            for _ in range(steps):
                random_return = np.random.normal(mean_return, std_return)
                next_price = prices[-1] * (1 + random_return)
                prices.append(next_price)
            simulation_results[i] = prices[1:]
        
        percentiles = np.percentile(simulation_results, [5, 25, 50, 75, 95], axis=0)
        return percentiles
    
    def mean_reversion_model(self, steps=36):
        """평균 회귀 모델"""
        long_term_mean = self.df_monthly.tail(60).mean()
        current_price = self.df_monthly.iloc[-1]
        half_life = 12
        reversion_speed = np.log(2) / half_life
        future_dates = pd.date_range(self.df_monthly.index[-1] + pd.DateOffset(months=1), periods=steps, freq='M')
        
        mean_reversion_forecast = []
        for t in range(1, steps + 1):
            forecasted_price = long_term_mean + (current_price - long_term_mean) * np.exp(-reversion_speed * t)
            mean_reversion_forecast.append(forecasted_price)
        
        return pd.Series(mean_reversion_forecast, index=future_dates)
    
    def comprehensive_forecast(self, steps=36):
        """종합 예측"""
        arima_forecast, arima_adj_forecast = self.arima_with_volatility_adjustment(steps)
        mc_percentiles = self.monte_carlo_simulation(steps)
        mr_forecast = self.mean_reversion_model(steps)
        
        future_dates = pd.date_range(self.df_monthly.index[-1] + pd.DateOffset(months=1), periods=steps, freq='M')
        
        ensemble_forecast = (
            0.5 * arima_adj_forecast + 
            0.2 * mc_percentiles[2] + 
            0.3 * mr_forecast
        )
        
        return {
            'dates': future_dates,
            'arima_basic': pd.Series(arima_forecast, index=future_dates),
            'arima_adjusted': pd.Series(arima_adj_forecast, index=future_dates),
            'monte_carlo': mc_percentiles,
            'mean_reversion': mr_forecast,
            'ensemble': pd.Series(ensemble_forecast, index=future_dates)
        }
