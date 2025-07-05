# Standard library imports
import io
import os
from datetime import datetime, timedelta
from typing import List

# Third-party library imports
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Local project imports
from chat import ChatRequest, chat_logic
from predictor import AdvancedStockPredictor

# Load environment variables
load_dotenv()
naver_client_id = os.getenv("NAVER_CLIENT_ID")
naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

NAME_TO_TICKER = {}
df = pd.read_csv('corp_list.csv', encoding='utf-8')
for index, row in df.iterrows():
    NAME_TO_TICKER[row['회사명']] = row['종목코드']


# --------------------
# FastAPI
# --------------------
app = FastAPI()

templates = Jinja2Templates(directory="template")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --------------------
# index.html
# --------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------
# 사용자가 질문
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    return await chat_logic(req, naver_client_id, naver_client_secret, NAME_TO_TICKER, model)


# --------------------
# 주가 예측 차트 생성하기
# --------------------
@app.get("/plot.png")
async def get_plot(stock_code: str = Query(..., alias="ticker")):
    try:
        ticker = NAME_TO_TICKER.get(stock_code)
        if ticker is None:
            return {"error": "해당 회사의 종목코드를 찾을 수 없습니다."}
        predictor = AdvancedStockPredictor(ticker)
        threeago = datetime.today() - timedelta(days=365 * 3)
        predictor.load_data(start_date=threeago)
        forecasts = predictor.comprehensive_forecast()
        trend_signal, trend_changes, ma_short, ma_long = predictor.detect_regime_changes()
        var_95, max_drawdown, drawdown = predictor.calculate_downside_risk()

        buf = io.BytesIO()
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # 주가 및 이동평균선
        ax1.plot(predictor.df_monthly.index, predictor.df_monthly.values, label='주가', color='black', linewidth=2)
        ax1.plot(ma_short.index, ma_short.values, label='6개월 이동평균선', alpha=0.7, color='blue')
        ax1.plot(ma_long.index, ma_long.values, label='1년 이동평균선', alpha=0.7, color='orange')

        # 예측 구간 음영
        ax1.fill_between(forecasts['dates'], forecasts['monte_carlo'][1], forecasts['monte_carlo'][0],
                         alpha=0.3, color='red', label='예측범위')

        # 스타일 적용
        ax1.set_ylabel('주가(원)', fontsize=18)
        ax1.set_xlabel('날짜', fontsize=18)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.legend(fontsize=18)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("🔥 Plot 오류:", e)
        return {"error": "차트 생성에 실패했습니다."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
