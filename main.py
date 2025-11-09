import asyncio
from exchanges import ExchangeHandler
from utils import load_config, calc_net_profit, check_risk, send_telegram

async def main():
    cfg = load_config()
    ex_handler = ExchangeHandler(cfg['exchanges'])
    balance = cfg['risk']['starting_balance']

    while True:
        funding_rates = await ex_handler.fetch_funding_rates()
        prices = await ex_handler.fetch_prices()

        # 找最大费率差币种
        max_diff = 0
        best_pair = None
        high_ex, low_ex = None, None
        for symbol in funding_rates['binance']:
            if symbol in funding_rates['bybit']:
                diff = funding_rates['binance'][symbol] - funding_rates['bybit'][symbol]
                if abs(diff) > max_diff:
                    max_diff = abs(diff)
                    best_pair = symbol
                    if diff > 0:
                        high_ex, low_ex = 'binance','bybit'
                    else:
                        high_ex, low_ex = 'bybit','binance'

        if best_pair:
            net_profit = calc_net_profit(max_diff, cfg['risk']['position_size'], cfg['risk']['leverage'])
            if net_profit >= cfg['risk']['profit_threshold'] and check_risk(balance, cfg['risk']['min_balance']):
                await ex_handler.place_order(low_ex, best_pair, 'long', cfg['risk']['position_size'], cfg['risk']['leverage'])
                await ex_handler.place_order(high_ex, best_pair, 'short', cfg['risk']['position_size'], cfg['risk']['leverage'])
                send_telegram(f"[SIM] Arbitrage executed: {best_pair}, Estimated Profit~{net_profit} USDT",
                              cfg['telegram']['token'], cfg['telegram']['chat_id'])
        await asyncio.sleep(5)  # 每5秒扫描一次

if __name__ == "__main__":
    asyncio.run(main())
