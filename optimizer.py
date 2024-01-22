import pandas as pd
import numpy as np
import optuna
import os
from strategy import Strategy, Filter

class Optimizer():
  def __init__(self):
    self.strategy = None
    self.symbols = []
    self.train_data = {}
    self.test_data = {}
  
  def fetch_data(self):
    historical_data = {}
    lengths = []
    self.symbols = [filename.replace('.csv', '') for filename in os.listdir('historical_data')]
    self.symbols.remove('SHIB')
    for symbol in self.symbols:
      data = pd.read_csv(f'historical_data/{symbol}.csv')
      historical_data[symbol] = data
      lengths.append(len(data))

    top = np.sort(lengths)[-5]
    
    for symbol in self.symbols:
      if len(historical_data[symbol]) < top:
        continue

      test_size = int(0.1 * top)
      self.train_data[symbol] = historical_data[symbol].iloc[-top:-test_size]
      self.test_data[symbol] = historical_data[symbol].iloc[-test_size:]

    self.symbols = list(self.train_data.keys())

  def optimize(self):
    def objective(trial):
      def make_filter(name):
        if trial.suggest_categorical(f'{name}_ignore', [True, False]):
          return Filter(ignore=True)

        return Filter(
          ignore=False,
          close_above_ema_slow=trial.suggest_categorical(f'{name}_close_above_ema_slow', [True, False]),
          close_above_ema_med=trial.suggest_categorical(f'{name}_close_above_ema_med', [True, False]),
          close_above_sma_slow=trial.suggest_categorical(f'{name}_close_above_sma_slow', [True, False]),
          close_above_sma_med=trial.suggest_categorical(f'{name}_close_above_sma_slow', [True, False]),
          rsi_below_30=trial.suggest_categorical(f'{name}_rsi_below_30', [True, False]),
        )
      
      self.strategy = Strategy(
        symbols=self.symbols,
        sma_fast=trial.suggest_int('sma_fast', 2, 20),
        sma_med=trial.suggest_int('sma_med', 20, 40),
        sma_slow=trial.suggest_int('sma_slow', 40, 100),
        ema_fast=trial.suggest_int('ema_fast', 2, 20),
        ema_med=trial.suggest_int('ema_med', 20, 40),
        ema_slow=trial.suggest_int('ema_slow', 40, 100),
        macd_fast=trial.suggest_int('macd_fast', 10, 20),
        macd_slow=trial.suggest_int('macd_slow', 20, 30),
        macd_signal=trial.suggest_int('macd_signal', 5, 20),
        ichimoku_low=trial.suggest_int('ichimoku_slow', 5, 15),
        ichimoku_med=trial.suggest_int('ichimoku_med', 15, 30),
        ichimoku_high=trial.suggest_int('ichimoku_high', 30, 50),

        bb_window=trial.suggest_int('bb_window', 10, 20),
        bb_dev=trial.suggest_float('bb_dev', 0.2, 2.5, step=0.1),
        keltner_window=trial.suggest_int('keltner_window', 10, 20),
        keltner_atr_window=trial.suggest_int('keltner_atr_window', 2, 20),

        rsi_window=trial.suggest_int('rsi_window', 2, 20),
        
        sma_fast_med_cross_filter=make_filter('sma_fast_med_cross_filter'),
        sma_fast_slow_cross_filter=make_filter('sma_fast_slow_cross_filter'),
        sma_med_slow_cross_filter=make_filter('sma_med_slow_cross_filter'),
        ema_fast_med_cross_filter=make_filter('ema_fast_med_cross_filter'),
        ema_fast_slow_cross_filter=make_filter('ema_fast_slow_cross_filter'),
        ema_med_slow_cross_filter=make_filter('ema_med_slow_cross_filter'),
        macd_cross_filter=make_filter('macd_cross_filter'),
        macd_zero_cross_filter=make_filter('macd_zero_cross_filter'),
        senkou_a_b_cross_filter=make_filter('senkou_a_b_cross_filter'),
        tenkan_kijun_cross_filter=make_filter('tenkan_kijun_cross_filter'),

        bb_breakout_filter=make_filter('bb_breakout_filter'),
        bb_bounce_filter=make_filter('bb_bounce_filter'),
        keltner_breakout_filter=make_filter('keltner_breakout_filter'),
        keltner_bounce_filter=make_filter('keltner_bounce_filter'),

        rsi_as_oversold_filter=make_filter('rsi_as_oversold_filter'),
        rsi_as_momentum_filter=make_filter('rsi_as_momentum_filter'),
      )
      self.strategy.make_ta_data(self.train_data)

      return self.strategy.backtest()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective)

  def test(self):
    pass

if __name__ == '__main__':
  
  optimizer = Optimizer()
  optimizer.fetch_data()
  optimizer.optimize()