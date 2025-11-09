import yaml
import logging
import requests
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, filename='system.log', format='%(asctime)s - %(message)s')

def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def calc_net_profit(diff, position, leverage, fee=0.001, slippage=0.5):
    gross = diff * position * leverage
    net = gross - fee*position - slippage
    return net

def check_risk(balance, min_balance):
    return balance >= min_balance

def send_telegram(msg, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg}
    try:
        requests.post(url, data=payload)
    except:
        logging.warning("Telegram message failed")
