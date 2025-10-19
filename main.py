# ai_trading_bot_ccxt_safe.py
# 完整可运行版本：多交易所、多币种、多周期策略 + AI LSTM + TG消息
# 自动安装依赖: ccxt, pandas, numpy, torch, scikit-learn, joblib, requests

import os
import time
import traceback
import requests
from datetime import datetime, timezone
from threading import Lock
import pandas as pd
import numpy as np
import ccxt
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import subprocess
import sys


# 简单日志函数，防止报错
def save_log(text):
    print(text)

# ================= 安全版依赖检查 =================
def check_and_install_packages(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except Exception as e:
            print(f"[依赖安装失败]{pkg}: {e}")
            save_log(f"[依赖安装失败]{pkg}: {e}")
            try:
                # ✅ 去掉 --user
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except Exception as e2:
                print(f"请手动安装依赖: {pkg}")
                save_log(f"[依赖安装失败2]{pkg}: {e2}")

# 必要依赖列表
required_packages = ["ccxt","pandas","numpy","torch","scikit-learn","joblib","requests"]

# 执行依赖检查
check_and_install_packages(required_packages)

# ================== 全局配置 ==================
EXCHANGES = {
    "binance": {"ccxt_id": "binance", "apiKey": os.environ.get("BINANCE_APIKEY"), "secret": os.environ.get("BINANCE_SECRET")},
    "okx": {"ccxt_id": "okx", "apiKey": os.environ.get("OKX_APIKEY"), "secret": os.environ.get("OKX_SECRET")},
}

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
DEFAULT_WEIGHTS = {"1m": 0.1, "5m": 0.2, "15m": 0.3, "1h": 0.2, "4h": 0.2}
DEFAULT_WEIGHTS = {""1m":0.1, "5m":0.2, "15m":0.3, "1h":0.2, "4h":0.2}
FEATURE_COLUMNS = [
    'MA5','MA10','MA20','MA50','MA200','MA5_10_cross','RSI','MACD','MACD_signal','MACD_hist',
    'ATR','BB_mid','BB_upper','BB_lower','CCI','Williams_R','CMF','OBV','Volume_Change',
    'K','D','J','MOM','ADX'
]

SIGNAL_THRESHOLD = 0.5
MAX_POSITION_RATIO = 0.3
STRATEGY_BUDGET = {"scalp":0.1,"short":0.1,"long":0.1, "combined": 0.3}
TRADING_FEE = 0.001
LOOP_DELAY = 60
MODEL_DIR = "models"
LOG_DIR = "logs"
HISTORY_DIR = "history"
SEQ_LEN = 50

BOT_TOKEN = os.environ.get("BOT_TOKEN","")
CHAT_ID = os.environ.get("CHAT_ID","")

SIM_BALANCE = 1000.0
POSITIONS = []
POS_LOCK = Lock()
BALANCE_LOCK = Lock()

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# ================== 日志 & TG ==================
def save_log(text):
    fn = f"{LOG_DIR}/log_{datetime.now().strftime('%Y%m%d')}.txt"
    try:
        with open(fn,"a",encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {text}\n")
    except:
        pass

def send_tg_cn(title,body):
    if not BOT_TOKEN or not CHAT_ID:
        save_log(f"[TG未配置] {title} {body}")
        return
    try:
        text = f"【{title}】\n{body}"
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        save_log(f"[TG发送错误]{e}")

# ================== CCXT 获取 K 线 ==================
def get_ccxt_klines(symbol, interval="1h", limit=500, exchange_key="binance"):
    ex_cfg = EXCHANGES[exchange_key]
    try:
        ex = getattr(ccxt, ex_cfg["ccxt_id"])({"apiKey": ex_cfg["apiKey"], "secret": ex_cfg["secret"]})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        save_log(f"[CCXT获取K线失败]{symbol}-{interval}-{exchange_key}: {e}")
        return None

# ================== 技术指标 ==================
def calculate_indicators(df):
    df = df.copy()
    try:
        for p in [5,10,20,50,200]:
            df[f"MA{p}"] = df["close"].rolling(p, min_periods=1).mean()
        df["MA5_10_cross"] = (df["MA5"] > df["MA10"]).astype(int)
        diff = df["close"].diff()
        gain = diff.clip(lower=0)
        loss = -diff.clip(upper=0)
        df["RSI"] = 100 - 100/(1 + gain.rolling(14,min_periods=1).mean()/(loss.rolling(14,min_periods=1).mean()+1e-9))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        df["ATR"] = (df["high"] - df["low"]).rolling(14, min_periods=1).mean()
        df["BB_mid"] = df["close"].rolling(20).mean()
        df["BB_std"] = df["close"].rolling(20).std()
        df["BB_upper"] = df["BB_mid"] + 2*df["BB_std"]
        df["BB_lower"] = df["BB_mid"] - 2*df["BB_std"]
        tp = (df["high"] + df["low"] + df["close"])/3
        sma_tp = tp.rolling(20,min_periods=1).mean()
        mad = tp.rolling(20,min_periods=1).apply(lambda x: np.mean(np.abs(x-np.mean(x))),raw=True)
        df["CCI"] = (tp - sma_tp)/(0.015*mad + 1e-9)
        high_max = df["high"].rolling(14,min_periods=1).max()
        low_min = df["low"].rolling(14,min_periods=1).min()
        df["Williams_R"] = -100*(high_max-df["close"])/(high_max-low_min +1e-9)
        mfv = ((df["close"]-df["low"])-(df["high"]-df["close"]))/(df["high"]-df["low"] +1e-9)*df["volume"]
        df["CMF"] = mfv.rolling(20,min_periods=1).sum()/df["volume"].rolling(20,min_periods=1).sum()
        obv = [0]
        for i in range(1,len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv
        df["Volume_Change"] = df["volume"].pct_change().fillna(0)
        low_min_9 = df["low"].rolling(9,min_periods=1).min()
        high_max_9 = df["high"].rolling(9,min_periods=1).max()
        df["K"] = 100*(df["close"] - low_min_9)/(high_max_9 - low_min_9 + 1e-9)
        df["D"] = df["K"].rolling(3,min_periods=1).mean()
        df["J"] = 3*df["K"] - 2*df["D"]
        df["MOM"] = df["close"].diff(10)
        up_move = df["high"].diff()
        down_move = df["low"].diff().abs()
        plus_dm = np.where((up_move>down_move)&(up_move>0), up_move, 0)
        minus_dm = np.where((down_move>up_move)&(down_move>0), down_move, 0)
        tr = np.maximum(df["high"]-df["low"], np.maximum(abs(df["high"]-df["close"].shift()), abs(df["low"]-df["close"].shift())))
        atr = pd.Series(tr).rolling(14,min_periods=1).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(14,min_periods=1).mean()/(atr+1e-9)
        minus_di = 100 * pd.Series(minus_dm).rolling(14,min_periods=1).mean()/(atr+1e-9)
        df["ADX"] = 100 * abs(plus_di - minus_di)/(plus_di + minus_di + 1e-9)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        save_log(f"[指标计算错误]{e}")
        return df

# ================== LSTM 模型 ==================
class LSTMModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)
        self.hidden_size=hidden_size
        self.num_layers=num_layers

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out,_ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out

# ================== 动态 TP/SL ==================
def dynamic_tp_sl(df,direction,signal_strength):
    atr = df["ATR"].iloc[-1] if "ATR" in df.columns else (df["high"].iloc[-1]-df["low"].iloc[-1])
    price = df["close"].iloc[-1]
    tp_ratio = 3 + abs(signal_strength) * 2
    sl_ratio = 2 - abs(signal_strength) * 0.5
    tp = price * (1 + direction * atr / (price + 1e-9) * tp_ratio)
    sl = price * (1 - direction * atr / (price + 1e-9) * sl_ratio)
    return tp, sl

# ================== 仓位管理 ==================
def allocate_position(symbol, strategy_signal, strategy_name, exchange_key="binance"):
    global SIM_BALANCE, POSITIONS
    try:
        if abs(strategy_signal) < SIGNAL_THRESHOLD:
            return
        with BALANCE_LOCK:
            total_notional = sum([p.get("notional", p.get("size", 0)) for p in POSITIONS])
            strategy_budget = STRATEGY_BUDGET.get(strategy_name, 0.1)
            max_allowed = max(0.0, SIM_BALANCE * MAX_POSITION_RATIO * strategy_budget)
            if total_notional >= max_allowed:
                return
            notional = min(max_allowed - total_notional, 50 + abs(strategy_signal) * 100)
        try:
            price = float(get_ccxt_klines(symbol, "1h", exchange_key=exchange_key)["close"].iloc[-1])
        except:
            price = 0.0
        direction = 1 if strategy_signal > 0 else -1
        try:
            tp, sl = dynamic_tp_sl(get_ccxt_klines(symbol, "1h", exchange_key=exchange_key), direction, abs(strategy_signal))
        except:
            tp, sl = None, None
        with POS_LOCK:
            POSITIONS.append({
                "symbol": symbol,
                "size": notional,
                "open_price": price,
                "notional": notional,
                "direction": direction,
                "tp": tp,
                "sl": sl,
                "strategy": strategy_name,
                "time": datetime.now(timezone.utc),
                "exchange": exchange_key
            })
        save_log(f"[模拟开仓]{symbol} {strategy_name} notional={notional:.2f} price={price:.6f} dir={direction}")
        try:
            tp_str = f"{tp:.6f}" if tp else "N/A"
            sl_str = f"{sl:.6f}" if sl else "N/A"
            dir_str = "多" if direction == 1 else "空"
            msg = f"币种:{symbol}\n方向:{dir_str}\n开仓价格:{price:.6f}\n仓位资金:{notional:.2f} USDT\n止盈:{tp_str}\n止损:{sl_str}\n策略:{strategy_name}"
            send_tg_cn("模拟开仓", msg)
        except Exception as e:
            save_log(f"[中文开仓通知异常]{e}")
    except Exception as e:
        save_log(f"[allocate_position异常]{e}\n{traceback.format_exc()}")

def check_positions():
    global SIM_BALANCE, POSITIONS
    to_remove = []
    current_prices = {}
    with POS_LOCK:
        for pos in POSITIONS:
            try:
                sym = pos["symbol"]
                exch = pos.get("exchange", "binance")
                price = float(get_ccxt_klines(sym, "1h", exchange_key=exch)["close"].iloc[-1])
                current_prices[sym] = price
            except:
                current_prices[sym] = pos.get("open_price", 0.0)
    with POS_LOCK:
        for pos in POSITIONS:
            try:
                sym = pos["symbol"]
                dir = pos.get("direction", 1)
                open_price = pos.get("open_price", 0)
                tp = pos.get("tp", None)
                sl = pos.get("sl", None)
                notional = pos.get("notional", pos.get("size", 0))
                strategy_name = pos.get("strategy", "")
                cur_price = current_prices.get(sym, open_price)
                trigger_tp = tp is not None and ((dir == 1 and cur_price >= tp) or (dir == -1 and cur_price <= tp))
                trigger_sl = sl is not None and ((dir == 1 and cur_price <= sl) or (dir == -1 and cur_price >= sl))
                if not (trigger_tp or trigger_sl):
                    continue
                profit = (cur_price - open_price) * dir * (notional / open_price) - notional * TRADING_FEE * 2
                with BALANCE_LOCK:
                    SIM_BALANCE += profit
                try:
                    dir_str = "多" if dir == 1 else "空"
                    msg = f"币种:{sym}\n方向:{dir_str}\n开仓:{open_price:.6f}\n当前:{cur_price:.6f}\n仓位:{notional:.2f} USDT\n止盈:{tp if tp else 0:.6f}\n止损:{sl if sl else 0:.6f}\n盈亏:{profit:.2f} USDT\n策略:{strategy_name}\n触发:{'止盈' if trigger_tp else '止损'}"
                    send_tg_cn("模拟平仓", msg)
                except:
                    pass
                to_remove.append(pos)
            except Exception as e:
                save_log(f"[check_positions异常]{e}")
    for pos in to_remove:
        if pos in POSITIONS:
            POSITIONS.remove(pos)

# ================== 多策略信号 ==================
def combined_signal(symbol, data_dict, model, scaler):
    signals = {}
    for strat, interval in [("scalp","1m"),("short","15m"),("long","1h")]:
        df = data_dict.get(interval)
        if df is None or len(df) < 20:
            signals[strat] = 0.0
            continue
        df_ind = calculate_indicators(df)
        features = df_ind[FEATURE_COLUMNS].values
        probs = ai_predict_seq(features, model, scaler)
        signal = probs[1] - probs[2]  # 多-空概率
        signals[strat] = signal
    # 加权组合信号
    combined = 0
    total_weight = 0
    for strat, signal in signals.items():
        interval = {"scalp":"1m", "short":"15m", "long":"1h"}[strat]
        weight = DEFAULT_WEIGHTS.get(interval, 0.2)
        combined += signal * weight
        total_weight += weight
    if total_weight > 0:
        combined /= total_weight
    return combined, signals

# ================== 模型加载与训练 ==================
def load_model_if_exists(symbol):
    """加载已保存模型和标准化器"""
    model_path = f"{MODEL_DIR}/{symbol.replace('/', '_')}_lstm.pt"
    scaler_path = f"{MODEL_DIR}/{symbol.replace('/', '_')}_scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = LSTMModel(len(FEATURE_COLUMNS), 64, 2, 3)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        scaler = joblib.load(scaler_path)
        save_log(f"[模型加载成功]{symbol}")
        return model, scaler
    else:
        save_log(f"[模型未找到]{symbol}")
        return None, None


def train_model_from_df(df, symbol):
    """从历史K线训练LSTM模型"""
    df = calculate_indicators(df)
    if len(df) < SEQ_LEN + 1:
        save_log(f"[训练数据不足]{symbol}")
        return None, None
    features = df[FEATURE_COLUMNS].values
    change = df["close"].pct_change().shift(-1).fillna(0)
    y = np.zeros(len(change))
    y[change > 0.001] = 1  # long
    y[change < -0.001] = 2  # short
    # 0 for neutral
    X_seq = []
    y_seq = []
    for i in range(SEQ_LEN, len(features)):
        X_seq.append(features[i - SEQ_LEN: i])
        y_seq.append(y[i - 1])  # predict the label for the period after the last in sequence
    if len(X_seq) < 1:
        save_log(f"[序列数据不足]{symbol}")
        return None, None
    scaler = StandardScaler()
    all_features_flat = np.vstack(X_seq).reshape(-1, len(FEATURE_COLUMNS))
    scaler.fit(all_features_flat)
    X_seq_scaled = np.array([scaler.transform(seq) for seq in X_seq])
    X_tensor = torch.tensor(X_seq_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = LSTMModel(len(FEATURE_COLUMNS), 64, 2, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(3):  # 训练3轮即可
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{MODEL_DIR}/{symbol.replace('/', '_')}_lstm.pt")
    joblib.dump(scaler, f"{MODEL_DIR}/{symbol.replace('/', '_')}_scaler.pkl")
    save_log(f"[模型训练完成]{symbol}")
    return model, scaler


def ai_predict_seq(features, model, scaler):
    """用AI模型预测多空概率"""
    if model is None or scaler is None:
        return [1.0, 0.0, 0.0]
    try:
        seq_len = min(SEQ_LEN, len(features))
        features_last = features[-seq_len:]
        X_scaled = scaler.transform(features_last)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(X_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        return [probs[0], probs[1], probs[2]]  # [neutral, long, short]
    except Exception as e:
        save_log(f"[AI预测错误]{e}")
        return [1.0, 0.0, 0.0]

# ============================================================
# =============== 回测模块（手动触发 / 全币种）=================
# ============================================================

def analyze_symbol(df, symbol, interval):
    model, scaler = load_model_if_exists(symbol)
    if model is None:
        model, scaler = train_model_from_df(df, symbol)
    if model is None:
        return 0
    df_ind = calculate_indicators(df)
    features = df_ind[FEATURE_COLUMNS].values
    probs = ai_predict_seq(features, model, scaler)
    signal = probs[1] - probs[2]
    if abs(signal) < SIGNAL_THRESHOLD:
        return 0
    return 1 if signal > 0 else -1

def backtest_all_symbols():
    """
    手动触发回测：对所有配置币种进行AI逻辑回测（不使用模拟资金）
    """
    try:
        results = []
        for symbol in SYMBOLS:
            df = get_ccxt_klines(symbol, "1h", limit=2000, exchange_key="binance")  # ✅ 获取2000根K线
            if df is None or len(df) < 100:
                save_log(f"[回测] {symbol} 数据不足，跳过")
                continue

            # === 应用与实盘一致的交易逻辑 ===
            signals = []
            for i in range(SEQ_LEN + 20, len(df)):  # 确保足够数据
                try:
                    df_slice = df.iloc[:i]
                    sig = analyze_symbol(df_slice, symbol, "1h")
                    signals.append(sig)
                except Exception:
                    signals.append(0)

            df = df.iloc[SEQ_LEN + 20 - len(signals):].copy()
            df["signal"] = signals

            # === 回测：按信号方向统计盈亏 ===
            profits = []
            last_pos = None
            entry_price = 0

            for i in range(1, len(df)):
                sig = df["signal"].iloc[i]
                close = df["close"].iloc[i]

                # 开仓
                if sig == 1 and last_pos is None:
                    last_pos = "LONG"
                    entry_price = close
                elif sig == -1 and last_pos is None:
                    last_pos = "SHORT"
                    entry_price = close

                # 平仓
                elif last_pos == "LONG" and sig == -1:
                    profit = (close - entry_price) / entry_price - TRADING_FEE * 2
                    profits.append(profit)
                    last_pos = "SHORT"
                    entry_price = close
                elif last_pos == "SHORT" and sig == 1:
                    profit = (entry_price - close) / entry_price - TRADING_FEE * 2
                    profits.append(profit)
                    last_pos = "LONG"
                    entry_price = close

            if len(profits) == 0:
                save_log(f"[回测] {symbol} 无有效信号")
                continue

            win_rate = sum([1 for p in profits if p > 0]) / len(profits)
            avg_win = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
            avg_loss = abs(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

            results.append({
                "symbol": symbol,
                "trades": len(profits),
                "win_rate": round(win_rate * 100, 2),
                "profit_factor": round(profit_factor, 2),
                "avg_profit": round(np.mean(profits) * 100, 2),
            })

         # === 输出汇总 ===
        df_result = pd.DataFrame(results)
        if len(df_result) == 0:
            save_log("⚠️ 回测无结果。")
            return

        summary = df_result.to_string(index=False)
        save_log("📊 [AI回测汇总]\n" + summary)

    except Exception as e:
        save_log(f"[回测错误] {e}")

# ================== 主循环 ==================
def main_loop():
    models = {}
    scalers = {}
    for sym in SYMBOLS:
        model, scaler = load_model_if_exists(sym)
        if model is None:
            df = get_ccxt_klines(sym, "1h")
            if df is not None:
                model, scaler = train_model_from_df(df, sym)
        models[sym] = model
        scalers[sym] = scaler
    while True:
        try:
            for exchange_key in EXCHANGES:
                data_dict_all = {}
                for sym in SYMBOLS:
                    data_dict_all[sym] = {}
                    for interval in INTERVALS:
                        df = get_ccxt_klines(sym, interval, exchange_key=exchange_key)
                        if df is not None:
                            data_dict_all[sym][interval] = df
                for sym in SYMBOLS:
                    model = models.get(sym)
                    scaler = scalers.get(sym)
                    if model is None or scaler is None:
                        continue
                    combined_sig, _ = combined_signal(sym, data_dict_all[sym], model, scaler)
                    allocate_position(sym, combined_sig, "combined", exchange_key=exchange_key)
            check_positions()
            time.sleep(LOOP_DELAY)
        except Exception as e:
            save_log(f"[主循环异常]{e}")
            time.sleep(LOOP_DELAY)

if __name__ == "__main__":
    import argparse
    from threading import Thread
    import time
    from flask import Flask

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="prod", choices=["prod", "backtest"], help="运行模式")
    args = parser.parse_args()

    if args.mode == "backtest":
        save_log("=== 开始回测模式 ===")
        backtest_all_symbols()
    else:
        app = Flask(__name__)

        @app.route("/")
        def index():
            return "AI Trading Bot Running"

        # ---------------- Flask端口启动函数 ----------------
def run_flask():
    import socket
    port = int(os.environ.get("PORT", 8080))
    # 自动寻找空闲端口（3000~9000）
    for p in range(3000, 9000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p
                break
    try:
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except Exception as e:
        save_log(f"[Flask启动失败] {e}")


# ---------------- Telegram监听 ----------------
def tg_listener():
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            params = {"timeout": 10, "offset": offset}
            updates = requests.get(url, params=params, timeout=15).json()
            
            for u in updates.get("result", []):
                offset = u["update_id"] + 1  # ✅ 防止重复读取
                msg = u.get("message", {}).get("text", "")
                if msg:
                    save_log(f"[TG收到] {msg}")
                    send_tg_cn("回复", f"收到指令：{msg}\n机器人运行中 ✅")
        except Exception as e:
            save_log(f"[TG监听错误] {e}")
        time.sleep(2)


# ---------------- 主启动 ----------------
if __name__ == "__main__":
    from threading import Thread

    # 启动 Flask
    Thread(target=run_flask, daemon=True).start()

    # 启动 Telegram 监听
    Thread(target=tg_listener, daemon=True).start()

    # 启动主循环
    Thread(target=main_loop, daemon=True).start()

    save_log("=== 机器人启动完成（Prod 版）===")

    # 主线程保活
    while True:
        time.sleep(60)
