import ta
import pandas as pd
import numpy as np

TAKE_PROFIT = 1.01
STOP_LOSS = 0.995

class Filter():
  def __init__(self, **kwargs):
    self.ignore = kwargs['ignore']
    if self.ignore:
      return

    self.close_above_ema_med = kwargs['close_above_ema_med']
    self.close_above_ema_slow = kwargs['close_above_ema_slow']
    self.close_above_sma_med = kwargs['close_above_sma_med']
    self.close_above_sma_slow = kwargs['close_above_sma_slow']

    self.rsi_below_30 = kwargs['rsi_below_30']
  def __call__(self, ta_row):
    if self.ignore:
      return False
    if self.close_above_ema_med and ta_row['ema_med'] < ta_row['close']:
      return False
    if self.close_above_ema_slow and ta_row['ema_slow'] < ta_row['close']:
      return False
    if self.close_above_sma_med and ta_row['sma_med'] < ta_row['close']:
      return False
    if self.close_above_sma_slow and ta_row['sma_slow'] < ta_row['close']:
      return False

    if self.rsi_below_30 and ta_row['rsi'] > 30:
      return False
    return True

class Strategy():

  def __init__(self, symbols, **kwargs):
    self.symbols = symbols

    self.sma_fast = kwargs['sma_fast']
    self.sma_med = kwargs['sma_med']
    self.sma_slow = kwargs['sma_slow']

    self.ema_fast = kwargs['ema_fast']
    self.ema_med = kwargs['ema_med']
    self.ema_slow = kwargs['ema_slow']

    self.macd_slow = kwargs['macd_slow']
    self.macd_fast = kwargs['macd_fast']
    self.macd_signal = kwargs['macd_signal']

    self.ichimoku_low = kwargs['ichimoku_low']
    self.ichimoku_med = kwargs['ichimoku_med']
    self.ichimoku_high = kwargs['ichimoku_high']

    self.bb_window = kwargs['bb_window']
    self.bb_dev = kwargs['bb_dev']
    self.keltner_window = kwargs['keltner_window']
    self.keltner_atr_window = kwargs['keltner_atr_window']

    self.rsi_window = kwargs['rsi_window']

    ####################################################################

    self.sma_fast_med_cross_filter = kwargs['sma_fast_med_cross_filter']
    self.sma_fast_slow_cross_filter = kwargs['sma_fast_slow_cross_filter']
    self.sma_med_slow_cross_filter = kwargs['sma_med_slow_cross_filter']

    self.ema_fast_med_cross_filter = kwargs['ema_fast_med_cross_filter']
    self.ema_fast_slow_cross_filter = kwargs['ema_fast_slow_cross_filter']
    self.ema_med_slow_cross_filter = kwargs['ema_med_slow_cross_filter']

    self.macd_cross_filter = kwargs['macd_cross_filter']
    self.macd_zero_cross_filter = kwargs['macd_zero_cross_filter']

    self.senkou_a_b_cross_filter = kwargs['senkou_a_b_cross_filter']
    self.tenkan_kijun_cross_filter = kwargs['tenkan_kijun_cross_filter']

    self.bb_breakout_filter = kwargs['bb_breakout_filter']
    self.bb_bounce_filter = kwargs['bb_bounce_filter']

    self.keltner_breakout_filter = kwargs['keltner_breakout_filter']
    self.keltner_bounce_filter = kwargs['keltner_bounce_filter']

    self.rsi_as_oversold_filter = kwargs['rsi_as_oversold_filter']
    self.rsi_as_momentum_filter = kwargs['rsi_as_momentum_filter']

    self.ta_data = {}
  
  def make_ta_data(self, price_data):
    for symbol in self.symbols:
      open_prices = price_data[symbol]['open']
      close_prices = price_data[symbol]['close']
      low_prices = price_data[symbol]['low']
      high_prices = price_data[symbol]['high']

      sma_fast = ta.trend.SMAIndicator(close_prices, window=self.sma_fast)
      sma_med = ta.trend.SMAIndicator(close_prices, window=self.sma_med)
      sma_slow = ta.trend.SMAIndicator(close_prices, window=self.sma_slow)

      ema_fast = ta.trend.EMAIndicator(close_prices, window=self.ema_fast)
      ema_med = ta.trend.EMAIndicator(close_prices, window=self.ema_med)
      ema_slow = ta.trend.EMAIndicator(close_prices, window=self.ema_slow)

      macd = ta.trend.MACD(close_prices, window_fast=self.macd_fast,\
                                    window_slow=self.macd_signal, window_sign=self.macd_signal)
      ichimoku = ta.trend.IchimokuIndicator(high_prices, low_prices, window1=self.ichimoku_low,\
                                            window2=self.ichimoku_med, window3=self.ichimoku_high)
      

      bb = ta.volatility.BollingerBands(close_prices, window=self.bb_window, window_dev=self.bb_dev)
      keltner = ta.volatility.KeltnerChannel(close_prices, low_prices, high_prices,\
                                             window=self.keltner_window, window_atr=self.keltner_atr_window)
      
      rsi = ta.momentum.RSIIndicator(close_prices, window=self.rsi_window)

      df = pd.DataFrame({
        'open': open_prices,
        'close': close_prices,
        'low_prices': low_prices,
        'high_prices': high_prices,
        
        'sma_fast': sma_fast.sma_indicator(),
        'sma_med': sma_med.sma_indicator(),
        'sma_slow': sma_slow.sma_indicator(),
        'ema_fast': ema_fast.ema_indicator(),
        'ema_med': ema_med.ema_indicator(),
        'ema_slow': ema_slow.ema_indicator(),

        'macd': macd.macd(),
        'macd_signal': macd.macd_signal(),

        'senkou_a': ichimoku.ichimoku_a(),
        'senkou_b': ichimoku.ichimoku_b(),
        'kijun_sen': ichimoku.ichimoku_base_line(),
        'tenkan_sen': ichimoku.ichimoku_conversion_line(),

        'upper_bb': bb.bollinger_hband(),
        'lower_bb': bb.bollinger_lband(),
        'lower_keltner': keltner.keltner_channel_hband(),
        'upper_keltner': keltner.keltner_channel_lband(),

        'rsi': rsi.rsi()
      }).dropna()

      print(len(df))

      self.ta_data[symbol] = {}
      for col in df.columns:
        self.ta_data[symbol][col] = df[col].to_numpy()

  def entry_condition(self, symbol, index):
    symbol_data = self.ta_data[symbol]
    ta_row = {}
    for key in symbol_data.keys():
      ta_row[key] = symbol_data[key][index]
    
    if ta_row['sma_fast'] > ta_row['sma_med'] and \
        self.sma_fast_med_cross_filter(ta_row):
      return True
    if ta_row['sma_fast'] > ta_row['sma_slow'] and \
        self.sma_fast_slow_cross_filter(ta_row):
      return True
    if ta_row['sma_med'] > ta_row['sma_slow'] and \
        self.sma_med_slow_cross_filter(ta_row):
      return True
    if ta_row['ema_fast'] > ta_row['ema_med'] and \
        self.ema_fast_med_cross_filter(ta_row):
      return True
    if ta_row['ema_fast'] > ta_row['ema_slow'] and \
        self.ema_fast_slow_cross_filter(ta_row):
      return True
    if ta_row['ema_med'] > ta_row['ema_slow'] and \
        self.ema_med_slow_cross_filter(ta_row):
      return True
    if ta_row['macd'] > ta_row['macd_signal'] and \
        self.macd_cross_filter(ta_row):
      return True
    if ta_row['macd'] > 0 and self.macd_zero_cross_filter(ta_row):
      return True
    if ta_row['senkou_a'] > ta_row['senkou_b'] and self.senkou_a_b_cross_filter(ta_row):
      return True
    if ta_row['tenkan_sen'] > ta_row['kijun_sen'] and self.tenkan_kijun_cross_filter(ta_row):
      return True

    if ta_row['upper_bb'] > ta_row['close'] and self.bb_breakout_filter(ta_row):
      return True
    if not index == 0 and symbol_data['lower_bb'][index - 1] < symbol_data['close'][index] \
        and symbol_data['lower_bb'][index] > symbol_data['close'][index] and \
        self.bb_bounce_filter(ta_row):
      return True
    if ta_row['upper_keltner'] > ta_row['close'] and self.keltner_breakout_filter(ta_row):
      return True
    if not index == 0 and symbol_data['lower_keltner'][index - 1] < symbol_data['close'][index] \
        and symbol_data['lower_keltner'][index] > symbol_data['close'][index] and \
        self.keltner_bounce_filter(ta_row):
      return True

    if ta_row['rsi'] < 30 and self.rsi_as_oversold_filter(ta_row):
      return True
    if ta_row['rsi'] > 70 and self.rsi_as_momentum_filter(ta_row):
      return True

    return False

  def backtest(self):
    balance = 5000

    enter_price = {}
    enter_index = {}
    amount = {}
    close = {}

    for symbol in self.symbols:
      enter_price[symbol] = 0
      amount[symbol] = 0
      close[symbol] = self.ta_data[symbol]['close']
    
    holding_times = []
    entries_per_day = []
    entries_today = 0
    daily_profits = []
    previous_day_account_value = 5000
    wins = 1
    losses = 1
    min_balance = 10000000
    
    def get_account_value(index):
      value = balance
      for symbol in self.symbols:
        value += close[symbol][index] * amount[symbol]
      return value

    length = len(close[self.symbols[0]])

    for i in range(length):
      if i % 1440 == 0:
        entries_per_day.append(entries_today)
        entries_today = 0
        daily_profits.append(((get_account_value(i) - previous_day_account_value) / previous_day_account_value) * 100)
        previous_day_account_value = get_account_value(i)
      
      for symbol in self.symbols:

        if amount[symbol] == 0 and self.entry_condition(symbol, i):
          qty = (0.5 * balance / len(self.symbols)) / close[symbol][i]
          amount[symbol] = qty
          enter_price[symbol] = close[symbol][i]
          enter_index[symbol] = i
          balance -= close[symbol][i] * qty
          if balance < min_balance:
            min_balance = balance
          entries_today += 1
        elif amount[symbol] > 0 and \
              (close[symbol][i] > enter_price[symbol] * TAKE_PROFIT or \
              close[symbol][i] < enter_price[symbol] * STOP_LOSS):
          if close[symbol][i] > enter_price[symbol]:
            wins += 1
          else:
            losses += 1
          balance += close[symbol][i] * amount[symbol]
          amount[symbol] = 0
          holding_times.append(i - enter_index[symbol])

    for symbol in self.symbols:
      if amount[symbol] > 0:
        balance += close[symbol][-1] * amount[symbol] 

    print('total trades: ', wins + losses)
    print('win rate: ', wins / (wins + losses))
    print()

    print('average holding time: ', np.mean(holding_times))
    print('median holding time: ', np.median(holding_times))
    print('0.1 quantile holding time: ', np.quantile(holding_times, 0.1))
    print('0.9 quantile holding time: ', np.quantile(holding_times, 0.9))
    print('holding time std dev: ', np.std(holding_times))
    print()

    print('average entries per day: ', np.mean(entries_per_day))
    print('median entries per day: ', np.median(entries_per_day))
    print('0.1 quantile entries per day: ', np.quantile(entries_per_day, 0.1))
    print('0.9 quantile entries per day: ', np.quantile(entries_per_day, 0.9))
    print('entries per day std dev: ', np.std(entries_per_day))
    print()

    print('average daily profit: ', np.mean(daily_profits))
    print('median daily profit: ', np.median(daily_profits))
    print('0.1 quantile daily profit: ', np.quantile(daily_profits, 0.1))
    print('0.9 quantile daily profit: ', np.quantile(daily_profits, 0.9))
    print('daily profit std dev: ', np.std(daily_profits))

    print('min balance: ', min_balance)
    print('final balance: ', balance)
    return ((balance - 5000) / 5000) * 100
  
  