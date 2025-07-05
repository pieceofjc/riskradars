import pandas as pd
import numpy as np
import re
import requests
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta

from pykrx import stock
import yfinance as yf


SYSTEM_PROMPT_TEMPLATE = """
너는 주식과 회사 정보를 중학생도 이해할 수 있게 쉽게 설명해 주는 봇이야.
추가설명 - 중학생 눈높이 이런건 안적어도 돼, 그냥 부가 설명이라고 해, 
진짜 중학생한테 알려주는게 아니라 중학생도 이해할수 있을 정도로 설명하는게 목표야
정보출처는 무조건 다트에서 가져와
주의사항같은거 말하지마
마크다운언어로 예쁘게 작성해되 글머리 기호는 쓰지 말하줘
무조건 코스피 코스닥 기업만 대답하고 아니면 다시 질문해달라고 해줘
예시뉴스 제목에서 길머리 기호는 쓰지마
"""

CHAT_FORMAT_PROMPT = """
아래 형식에 맞춰 한국어로 답변해주세요.

##부도예측 결과
부도 가능성 한 문장

##해당 기업 주요매출 제품
###<주요제품 또는 서비스 1>
두줄정도설명

###<주요제품 또는 서비스 2>
두줄정도설명

##부가설명
그냥 회사에 대한 설립일이랑 개요만 설명해
"""

class ChatRequest(BaseModel):
    """
    챗봇 요청을 위한 Pydantic 모델입니다.
    """
    message: str
    ticker: Optional[str] = None
    stock_name: Optional[str] = None
    stock_info: Optional[dict] = None
    news_items: Optional[List[dict]] = None
    data: Optional[dict] = None
    beta_1y: Optional[float] = None
    beta_3y: Optional[float] = None
    main_products: Optional[str] = None


def get_latest_news_naver(query: str, naver_client_id, naver_client_secret, display: int = 2) -> List[dict]:
    """
    네이버 뉴스 검색 API를 사용하여 최신 뉴스를 가져옵니다.
    """
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret
    }

    # 네이버 API는 최대 100개까지 지원 → 더 많이 긁어서 정제
    params = {
        "query": query + " 주식",
        "display": display * 50,
        "sort": "date"
    }

    try:
        res = requests.get("https://openapi.naver.com/v1/search/news.json",
                           headers=headers, params=params)
        if res.status_code == 200:
            news = []
            for item in res.json().get("items", []):
                # 제목/본문 태그 제거
                title = re.sub("<.*?>", "", item["title"])
                desc = re.sub("<.*?>", "", item["description"])
                link = item["link"]

                # ✅ 기업명이 제목이나 본문에 정확하게 들어간 경우만
                if query in title or query in desc:
                    news.append({"title": title, "link": link})

                # 지정한 개수만큼만 수집
                if len(news) >= display:
                    break
            return news
    except Exception as e:
        print("📰 네이버 뉴스 API 오류:", e)

    return []

def extract_ticker(text: str, NAME_TO_TICKER: dict):
    """
    주어진 텍스트에서 종목 코드를 추출합니다.
    """
    
    for name, ticker in NAME_TO_TICKER.items():
        if name in text:
            return str(ticker).zfill(6), name

    # 숫자 5~6자리 종목코드 직접 입력한 경우
    m = re.search(r"\b\d{5,6}\b", text)
    if m:
        return m.group(0).zfill(6), None

    return None, None

def build_stock_info(ticker: str):
    """
    주어진 종목 코드에 대한 기본 정보를 가져옵니다.
    """
    try:
        name = stock.get_market_ticker_name(ticker)
        return {
            "name": name,
            "summary": "상세 업종 정보는 제공되지 않습니다.",
            "description": "공식 사업 내용은 별도로 확인해 주세요.",
            "products": [],
        }
    except Exception as e:
        print("build_stock_info error", e)
        return None

def fetch_stock_data(ticker: str) -> dict:
    """
    주어진 종목 코드에 대한 주식 데이터를 가져옵니다.
    """
    data = {
        "per": None,
        "roe": None,
        "debt_ratio": None,
        "sales": None,
        "market_cap": None,
        "main_products": None,
        "return_1y": None,
        "return_3y": None,
    }

    try:
        today = datetime.today()
        today_str = today.strftime("%Y%m%d")
        price_df = stock.get_market_ohlcv_by_date(fromdate="20200704", todate=today_str, ticker=ticker)

        if price_df.empty or "종가" not in price_df.columns:
            return data

        # 인덱스가 datetime인지 확인하고 변환
        if not pd.api.types.is_datetime64_any_dtype(price_df.index):
            price_df.index = pd.to_datetime(price_df.index)

        # 현재가
        current = price_df["종가"].dropna().iloc[-1]

        # ✅ 1년 수익률
        date_1y = today - timedelta(days=365)
        future_prices_1y = price_df[price_df.index >= date_1y]["종가"].dropna()
        if not future_prices_1y.empty:
            base_1y_price = future_prices_1y.iloc[0]
            data["return_1y"] = round((current / base_1y_price - 1) * 100, 2)
            print(f"[1Y] 기준일: {date_1y.strftime('%Y-%m-%d')}, 기준가: {base_1y_price}, 현재가: {current}")

        # 3년 수익률
        date_3y = today - timedelta(days=365 * 3)
        future_prices_3y = price_df[price_df.index >= date_3y]["종가"].dropna()

        if not future_prices_3y.empty:
            base_3y_price = future_prices_3y.iloc[0]  # date_3y 이후 가장 가까운 종가
            data["return_3y"] = round((current / base_3y_price - 1) * 100, 2)
            print(f"[3Y] 기준일: {date_3y.strftime('%Y-%m-%d')}, 기준가: {base_3y_price}, 현재가: {current}")

        # ✅ 재무 지표
        finance = stock.get_market_fundamental_by_date(fromdate=today_str, todate=today_str, ticker=ticker)
        if not finance.empty:
            per = finance["PER"].iloc[0]
            roe = finance["ROE"].iloc[0]
            data["per"] = round(per, 2) if pd.notna(per) else None
            data["roe"] = round(roe, 2) if pd.notna(roe) else None

        # ✅ 시가총액
        corp_info = stock.get_market_cap_by_date(fromdate=today_str, todate=today_str, ticker=ticker)
        if not corp_info.empty:
            data["market_cap"] = int(corp_info["시가총액"].iloc[0])

    except Exception as e:
        print("⚠️ fetch_stock_data 오류:", e)

    return data

def get_beta_values(stock_code: str):
    """
    주어진 종목 코드에 대한 베타 값을 계산합니다.
    """
    try:
        print(stock_code)
        stock_code += ".KS"
        beta_1y = get_beta_yf(stock_code, weeks=52)
        beta_3y = get_beta_yf(stock_code, weeks=156)
        return beta_1y, beta_3y
    except Exception as e:
        print(f"⚠️ 베타 계산 오류: {e}")
        return None, None

def get_beta_yf(ticker: str, weeks: int):
    """
    yfinance를 사용하여 베타 값을 계산합니다.
    """
    end = datetime.today()
    start = end - timedelta(weeks=weeks)

    stock_data = yf.download(ticker, start=start, end=end)
    market_data = yf.download("^KS11", start=start, end=end)  # KOSPI 지수

    if stock_data.empty or market_data.empty:
        return None

    stock_returns = stock_data['Close'].pct_change().dropna()
    market_returns = market_data['Close'].pct_change().dropna()

    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ['Stock', 'Market']

    covariance = np.cov(aligned['Stock'], aligned['Market'])[0, 1]
    market_variance = np.var(aligned['Market'])
    beta = covariance / market_variance

    return round(beta, 4)

async def chat_logic(req: ChatRequest, naver_client_id, naver_client_secret, NAME_TO_TICKER, model):
    """
    챗봇의 주요 로직을 처리합니다.
    """
    user_msg = req.message.strip()
    if not user_msg:
        return {"reply": "메시지를 입력해주세요."}

    ticker, stock_name = extract_ticker(user_msg, NAME_TO_TICKER)
    stock_info = build_stock_info(ticker) if ticker else None

    news_items: List[dict] = []

    data = fetch_stock_data(ticker) if ticker else {}

    #베타
    beta_1y, beta_3y = get_beta_values(ticker) if ticker else (None, None)

    main_products = data.get("main_products")

    # ✅ 뉴스 불러오기
    if stock_name:
        news_items = get_latest_news_naver(stock_name, naver_client_id, naver_client_secret)
    else:
        news_items = []

    try:
        context_parts = [SYSTEM_PROMPT_TEMPLATE, CHAT_FORMAT_PROMPT]
        if main_products:
            context_parts.append("주요 제품:\n" + main_products)

        prompt = "\n".join(context_parts) + f"\n사용자 질문: {user_msg}"
        response = model.generate_content(prompt)
        answer = response.text.strip()

        # ✅ 뉴스 마크다운 추가
        if news_items:
            news_markdown = "\n\n## 관련 뉴스\n" + "\n".join(f"[{item['title']}]({item['link']})" for item in news_items)
            answer += news_markdown

    except Exception as e:
        print("🔥 Gemini API 오류:", e)
        answer = "부도예측 결과\n오류로 정보를 불러오지 못했습니다.\n"

    return {
        "reply": answer,
        "name": stock_name,
        "per": data.get("per"),
        "roe": data.get("roe"),
        "debt_ratio": data.get("debt_ratio"),
        "sales": data.get("sales"),
        "market_cap": data.get("market_cap"),
        "main_products": main_products,
        "return_1y": data.get("return_1y"),
        "return_3y": data.get("return_3y"),
        "beta_1y": beta_1y, # 베타  
        "beta_3y": beta_3y, # 베타
        "stock_info": stock_info,
        "news": news_items,
        "stock_code": ticker,
    }
