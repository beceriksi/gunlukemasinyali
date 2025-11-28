import os
import time
from datetime import datetime, timezone

import requests
import pandas as pd

OKX_BASE = "https://www.okx.com"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")


# --------- Yardımcılar ---------
def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def jget(path, params=None, retries=3, timeout=10):
    url = f"{OKX_BASE}{path}"
    for _ in range(retries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        time.sleep(0.5)
    raise RuntimeError(f"OKX error: {r.status_code} {r.text}")


# --------- OKX Fonksiyonları ---------
def get_usdt_spot_instruments(limit=150):
    data = jget("/api/v5/public/instruments", {"instType": "SPOT"})
    insts = [d["instId"] for d in data["data"] if d["instId"].endswith("-USDT")]
    return insts[:limit]


def get_candles(inst_id, bar="1D", limit=120):
    data = jget("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": limit})
    candles = []
    for c in reversed(data["data"]):  # eski -> yeni
        candles.append(
            {
                "ts": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "vol": float(c[5]),
            }
        )
    return candles


def get_trades(inst_id, limit=200):
    data = jget("/api/v5/market/trades", {"instId": inst_id, "limit": limit})
    trades = []
    for t in data["data"]:
        trades.append(
            {
                "px": float(t["px"]),
                "sz": float(t["sz"]),
                "side": t["side"],  # buy/sell
                "ts": int(t["ts"]),
            }
        )
    return trades


# --------- Göstergeler ---------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_features(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def obv_features(df):
    # df: cols [close, vol]
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["vol"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["vol"].iloc[i])
        else:
            obv.append(obv[-1])
    s = pd.Series(obv)
    # son 10 bara göre trend
    if len(s) < 15:
        return False
    return s.iloc[-1] > s.iloc[-10]


def compute_ema_block(df):
    df["ema10"] = ema(df["close"], 10)
    df["ema20"] = ema(df["close"], 20)
    df["ema30"] = ema(df["close"], 30)
    return df


def ema_cross_freshness(df):
    """
    Son 3 gün içinde EMA10>EMA20>EMA30 ilk defa oluşmuş mu?
    Dönüş: (fresh_days or None, now_bullish: bool)
    """
    cond = (df["ema10"] > df["ema20"]) & (df["ema20"] > df["ema30"])
    if not cond.iloc[-1]:
        return None, False

    last_idx = None
    for i in range(len(df) - 1, 0, -1):
        if cond.iloc[i] and not cond.iloc[i - 1]:
            last_idx = i
            break

    if last_idx is None:
        # çoktandır bull alignment var
        return None, True

    # kaç gün önce?
    fresh_days = len(df) - 1 - last_idx
    return fresh_days, True


# --------- Hacim & Whale ---------
def buy_volume_ratio_1h(inst_id, limit=60):
    """
    Sadece BUY tarafının hacim oranı (1H).
    Yaklaşık hesap: toplam hacim * (buy_notional / (buy+sell)).
    """
    candles = get_candles(inst_id, bar="1H", limit=limit)
    df = pd.DataFrame(candles)

    # son 300 trade ile buy/sell oranı tahmini
    trades = get_trades(inst_id, limit=300)
    if not trades or df.empty:
        return 0.0

    # kabaca son X trade = son X saat gibi farz ediyoruz
    buy_notional = sum(t["px"] * t["sz"] for t in trades if t["side"] == "buy")
    sell_notional = sum(t["px"] * t["sz"] for t in trades if t["side"] == "sell")
    total_notional = buy_notional + sell_notional
    if total_notional == 0:
        return 0.0

    buy_ratio = buy_notional / total_notional

    df["buy_vol_est"] = df["vol"] * buy_ratio

    if len(df) < 25:
        return 0.0

    last3 = df["buy_vol_est"].tail(3).mean()
    prev20 = df["buy_vol_est"].tail(23).head(20).mean()
    if prev20 == 0:
        return 0.0
    return last3 / prev20


def whale_stats(trades):
    whale = 0
    big = 0
    xxl = 0
    buy_notional = 0.0
    sell_notional = 0.0

    for t in trades:
        notional = t["px"] * t["sz"]
        if t["side"] == "buy":
            buy_notional += notional
            if notional >= 300000:
                xxl += 1
            elif notional >= 150000:
                big += 1
            elif notional >= 50000:
                whale += 1
        else:
            sell_notional += notional

    net = buy_notional - sell_notional
    return {
        "whale": whale,
        "big": big,
        "xxl": xxl,
        "net_buy_delta": net,
    }


def trend_4h(inst_id):
    h4 = get_candles(inst_id, bar="4H", limit=40)
    df = pd.DataFrame(h4)
    if len(df) < 15:
        return 0.0
    last = df["close"].iloc[-1]
    base = df["close"].iloc[-10]
    if base <= 0:
        return 0.0
    return (last - base) / base * 100.0


# --------- Piyasa Yönü (BTC & ETH) ---------
def market_direction():
    try:
        symbols = ["BTC-USDT", "ETH-USDT"]
        trends = []
        net_deltas = []
        whale_balance = []

        for s in symbols:
            h4 = get_candles(s, bar="4H", limit=40)
            df = pd.DataFrame(h4)
            if df.empty:
                continue
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            trend_up = df["ema20"].iloc[-1] > df["ema50"].iloc[-1]

            # 4H fiyat değişimi
            if len(df) >= 15:
                t_change = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10] * 100
            else:
                t_change = 0

            macd_line, signal, hist = macd_features(df["close"])
            macd_bull = macd_line.iloc[-1] > signal.iloc[-1] and macd_line.iloc[-1] > 0

            trades = get_trades(s, limit=300)
            ws = whale_stats(trades)
            trends.append((trend_up, t_change, macd_bull))
            net_deltas.append(ws["net_buy_delta"])
            whale_balance.append(ws["whale"] + ws["big"] + ws["xxl"])

        if not trends:
            return "❔", "Belirsiz (veri yok)"

        avg_delta = sum(net_deltas)
        avg_whale = sum(whale_balance)

        up_count = sum(1 for t_up, ch, macd_bull in trends if t_up or ch > 0 or macd_bull)
        down_count = len(trends) - up_count

        if up_count >= 2 and avg_delta > 0 and avg_whale > 0:
            return "🟢", "Boğa (alıcılar güçlü)"
        if down_count >= 2 and avg_delta < 0:
            return "🔴", "Ayı (satıcı baskın)"
        return "🟡", "Yatay / belirsiz"
    except Exception:
        return "❔", "Piyasa yönü okunamadı"


# --------- Skorlama ---------
def score_coin(features):
    score = 0

    # EMA tazeliği
    fresh_days = features["ema_fresh_days"]
    if not features["ema_bull"]:
        ema_pts = 0
    elif fresh_days is None:
        ema_pts = 10
    elif fresh_days <= 1:
        ema_pts = 20
    elif fresh_days <= 3:
        ema_pts = 18
    elif fresh_days <= 5:
        ema_pts = 12
    else:
        ema_pts = 7
    score += ema_pts

    # BUY hacim vRatio
    v = features["vratio_buy"]
    if v > 5:
        score += 20
    elif v > 3:
        score += 15
    elif v > 1.5:
        score += 10
    elif v > 1.0:
        score += 5

    # Whale gücü
    w = features["whale"]
    b = features["big_whale"]
    xxl = features["xxl_whale"]
    net = features["net_buy_delta"]

    score += min(10, w * 3)  # 0-10
    if b > 0:
        score += 5
    if xxl > 0:
        score += 10
    if net > 0:
        score += 5

    # 4H trend
    t = features["trend4h"]
    if t > 5:
        score += 15
    elif t > 2:
        score += 10
    elif t > 1:
        score += 5

    # RSI
    r = features["rsi"]
    if r is not None:
        if r > 60:
            score += 10
        elif r >= 50:
            score += 8
        elif r >= 40:
            score += 5

    # MACD & OBV
    if features["macd_bull"]:
        score += 5
    if features["obv_bull"]:
        score += 5

    return max(0, min(100, int(score)))


# --------- Telegram ---------
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token ya da chat id yok, mesaj:\n", text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=20)
    if r.status_code != 200:
        print("Telegram hata:", r.text)


# --------- Coin Analizi ---------
def analyze_coin(inst_id, market_emoji, market_text):
    daily = get_candles(inst_id, bar="1D", limit=120)
    if len(daily) < 40:
        return None
    dfd = pd.DataFrame(daily)
    dfd = compute_ema_block(dfd)

    fresh_days, ema_bull = ema_cross_freshness(dfd)
    if not ema_bull:
        # EMA yukarı hizalanmamışsa hiç sinyal yok (SELL yok zaten)
        return None

    dfd["rsi"] = rsi(dfd["close"])
    macd_line, signal, hist = macd_features(dfd["close"])
    macd_bull = macd_line.iloc[-1] > signal.iloc[-1] and macd_line.iloc[-1] > 0
    obv_bull = obv_features(dfd[["close", "vol"]])

    rsi_last = float(dfd["rsi"].iloc[-1]) if pd.notna(dfd["rsi"].iloc[-1]) else None
    last_price = float(dfd["close"].iloc[-1])

    vratio = buy_volume_ratio_1h(inst_id, limit=60)
    t4 = trend_4h(inst_id)

    trades = get_trades(inst_id, limit=200)
    ws = whale_stats(trades)

    features = {
        "ema_fresh_days": fresh_days,
        "ema_bull": ema_bull,
        "vratio_buy": vratio,
        "whale": ws["whale"],
        "big_whale": ws["big"],
        "xxl_whale": ws["xxl"],
        "net_buy_delta": ws["net_buy_delta"],
        "trend4h": t4,
        "rsi": rsi_last,
        "macd_bull": macd_bull,
        "obv_bull": obv_bull,
    }

    score = score_coin(features)

    # Uyarı metni (engel yok, sadece info)
    notes = []
    if rsi_last is not None and rsi_last < 45:
        notes.append("RSI zayıf ⚠️")
    if t4 < 0:
        notes.append("4H trend aşağı ⚠️")
    if vratio <= 1.0:
        notes.append("BUY hacim zayıf ⚠️")
    if ws["net_buy_delta"] <= 0:
        notes.append("Net alım zayıf ⚠️")
    if market_emoji == "🔴":
        notes.append("Piyasa AYI modunda ⚠️")

    return {
        "inst": inst_id,
        "price": last_price,
        "score": score,
        "ema_fresh_days": fresh_days,
        "vratio_buy": round(vratio, 2),
        "trend4h": round(t4, 2),
        "rsi": round(rsi_last, 2) if rsi_last is not None else None,
        "whale": ws["whale"],
        "big_whale": ws["big"],
        "xxl_whale": ws["xxl"],
        "net_buy_delta": int(ws["net_buy_delta"]),
        "notes": "; ".join(notes) if notes else "",
    }


# --------- Ana Çalışma ---------
def run():
    print(f"[*] Başladı: {ts()}")

    market_emoji, market_text = market_direction()

    insts = get_usdt_spot_instruments(limit=120)

    results = []
    for inst in insts:
        try:
            info = analyze_coin(inst, market_emoji, market_text)
            if info:
                results.append(info)
        except Exception as e:
            print(f"[!] {inst} hata: {e}")
        time.sleep(0.2)  # API'yı yormamak için

    if not results:
        msg = f"⚡ EMA Premium Screener\n⏱ {ts()}\n\n📌 Piyasa Yönü: {market_emoji} {market_text}\n\nBugün kriterlere uyan alım sinyali bulunamadı."
        send_telegram(msg)
        return

    # skorla sırala, en yüksekten
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:10]

    lines = [
        f"⚡ EMA Premium Screener (Sadece ALIM Sinyali)",
        f"⏱ {ts()}",
        "",
        f"📌 Piyasa Yönü: {market_emoji} {market_text}",
        "",
        f"Top {len(top)} coin (0-100 Güven Puanı):",
    ]

    for i, r in enumerate(top, start=1):
        fresh_txt = "bilinmiyor"
        if r["ema_fresh_days"] is not None:
            fresh_txt = f"{r['ema_fresh_days']}g önce"
            if r["ema_fresh_days"] == 0:
                fresh_txt = "bugün"

        line1 = f"{i}) *{r['inst']}* | Güven: *{r['score']}/100*"
        line2 = (
            f"   Fiyat: {r['price']:.6f} | EMA tazelik: {fresh_txt} | "
            f"BUY vR(1H): {r['vratio_buy']} | 4H Trend: {r['trend4h']}% | RSI(1D): {r['rsi']}"
        )
        line3 = (
            f"   Whale: {r['whale']} / Big: {r['big_whale']} / XXL: {r['xxl_whale']} | "
            f"Net BUY: {r['net_buy_delta']} USDT"
        )
        lines.append(line1)
        lines.append(line2)
        lines.append(line3)
        if r["notes"]:
            lines.append(f"   Not: {r['notes']}")
        lines.append("")

    msg = "\n".join(lines)
    send_telegram(msg)


if __name__ == "__main__":
    run()
