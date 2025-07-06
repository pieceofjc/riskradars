import pandas as pd
import numpy as np
import re
import requests
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from pykrx import stock
import yfinance as yf
import sqlite3
import ssl
import urllib3
import certifi

# SSL 설정
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 시스템 프롬프트
SYSTEM_PROMPT_TEMPLATE = """
너는 주식과 회사 정보를 중학생도 이해할 수 있게 쉽게 설명해 주는 봇이야.
추가설명 - 중학생 눈높이 이런건 안적어도 돼, 그냥 부가 설명이라고 해, 
진짜 중학생한테 알려주는게 아니라 중학생도 이해할수 있을 정도로 설명하는게 목표야
정보출처는 무조건 다트에서 가져와
주의사항같은거 말하지마
마크다운 형식으로 예쁘게 작성해주세요
무조건 코스피 코스닥 기업만 대답하고 아니면 다시 질문해달라고 해줘
"""

# 챗봇 답변 형식
CHAT_FORMAT_PROMPT = """
아래 형식에 맞춰 한국어로 답변해주세요.

**{해당 기업 이름}의 정보를 제공합니다**

**부도예측 결과**
결과에 따른 자연스러운 설명

**주요 매출 제품**
***주요 제품 또는 서비스 1***
두줄정도설명

***주요 제품 또는 서비스 2***
두줄정도설명

**부가설명**
그냥 회사에 대한 설립일이랑 개요만 설명해주세요.
"""

# 챗봇 요청 모델
class ChatRequest(BaseModel):
    message: str
    ticker: Optional[str] = None
    stock_name: Optional[str] = None
    stock_info: Optional[dict] = None
    news_items: Optional[List[dict]] = None
    data: Optional[dict] = None
    beta_1y: Optional[float] = None
    beta_3y: Optional[float] = None
    main_products: Optional[str] = None

# 네이버 뉴스 검색 함수
def get_latest_news_naver(query: str, naver_client_id, naver_client_secret, display: int = 2) -> List[dict]:
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret
    }
    params = {
        "query": query + " 주식",
        "display": display * 50,
        "sort": "date"
    }
    try:
        res = requests.get("https://openapi.naver.com/v1/search/news.json", headers=headers, params=params)
        if res.status_code == 200:
            news = []
            for item in res.json().get("items", []):
                title = re.sub("<.*?>", "", item["title"])
                desc = re.sub("<.*?>", "", item["description"])
                link = item["link"]
                if query in title or query in desc:
                    news.append({"title": title, "link": link})
                if len(news) >= display:
                    break
            return news
    except Exception as e:
        print("📰 네이버 뉴스 API 오류:", e)
    return []

# 종목코드 추출 함수
def extract_ticker(user_msg, NAME_TO_TICKER):
    """사용자 메시지에서 ticker와 종목명을 추출하는 함수"""
    conn = None
    try:
        conn = sqlite3.connect("rrdb.db")
        cursor = conn.cursor()

        # 메시지에서 회사명 추출 (공백 제거 후 첫 번째 단어 또는 명사 추출)
        import re
        
        # 일반적인 한국어 회사명 패턴 (2-4글자)
        company_patterns = [
            r'([가-힣]{2,4})에',  # 2-4글자 한글 + "에"
            r'([A-Za-z]{2,10})에',  # 2-10글자 영문 + "에"
            r'([가-힣]{2,4})',  # 2-4글자 한글
            r'([A-Za-z]{2,10})',  # 2-10글자 영문
        ]
        
        # "에"가 들어가는 회사명 패턴 (더 긴 패턴 우선)
        extended_patterns = [
            r'([가-힣]{3,6})에',  # 3-6글자 한글 + "에" (더 긴 패턴)
            r'([A-Za-z]{3,12})에',  # 3-12글자 영문 + "에" (더 긴 패턴)
        ]
        
        # 모든 패턴을 길이순으로 정렬 (긴 패턴 우선)
        all_patterns = extended_patterns + company_patterns
        
        keywords = []
        for pattern in all_patterns:
            matches = re.findall(pattern, user_msg)
            keywords.extend(matches)
        
        # 중복 제거 및 길이순 정렬
        keywords = list(set(keywords))
        keywords.sort(key=len, reverse=True)
        
        print(f"[DEBUG] 추출된 키워드: {keywords}")
        
        for keyword in keywords:
            # 정확히 일치하는 경우 우선 검색
            cursor.execute("SELECT 종목코드, 회사명 FROM corp_info WHERE 회사명 = ?", (keyword,))
            row = cursor.fetchone()
            if row:
                return row[0], row[1]

            # 부분일치 검색
            cursor.execute("SELECT 종목코드, 회사명 FROM corp_info WHERE 회사명 LIKE ?", (f"%{keyword}%",))
            rows = cursor.fetchall()
            
            if rows:
                # 가장 긴 회사명을 우선 선택
                rows.sort(key=lambda x: len(x[1]), reverse=True)
                return rows[0][0], rows[0][1]

        return None, None
    except Exception as e:
        print(f"extract_ticker 오류: {e}")
        return None, None
    finally:
        if conn:
            conn.close()


# 종목 기본 정보 가져오기
def build_stock_info(ticker: str):
    try:
        name = stock.get_market_ticker_name(ticker)
        return {
            "name": name,
            "summary": "상세 업종 정보는 제공되지 않습니다.",
            "description": "공식 사업 내용은 별도로 확인해 주세요.",
            "products": []
        }
    except Exception as e:
        print("build_stock_info error", e)
        return None

# 주식 데이터 가져오기 (yfinance 실패시 pykrx 대체)
def fetch_stock_data(ticker: str) -> dict:
    data = {
        "per": None,
        "roe": None,
        "debt_ratio": None,
        "sales": None,
        "market_cap": None,
        "main_products": None,
        "return_1y": None,
        "return_3y": None
    }
    try:
        today = datetime.today()
        today_str = today.strftime("%Y%m%d")
        price_df = stock.get_market_ohlcv_by_date(fromdate="20200704", todate=today_str, ticker=ticker)
        if price_df.empty or "종가" not in price_df.columns:
            return data
        if not pd.api.types.is_datetime64_any_dtype(price_df.index):
            price_df.index = pd.to_datetime(price_df.index)
        current = price_df["종가"].dropna().iloc[-1]
        date_1y = today - timedelta(days=365)
        future_prices_1y = price_df[price_df.index >= date_1y]["종가"].dropna()
        if not future_prices_1y.empty:
            base_1y_price = future_prices_1y.iloc[0]
            data["return_1y"] = round((current / base_1y_price - 1) * 100, 2)
        date_3y = today - timedelta(days=365 * 3)
        future_prices_3y = price_df[price_df.index >= date_3y]["종가"].dropna()
        if not future_prices_3y.empty:
            base_3y_price = future_prices_3y.iloc[0]
            data["return_3y"] = round((current / base_3y_price - 1) * 100, 2)
        finance = stock.get_market_fundamental_by_date(fromdate=today_str, todate=today_str, ticker=ticker)
        if not finance.empty:
            per = finance["PER"].iloc[0]
            roe = finance["ROE"].iloc[0]
            data["per"] = round(per, 2) if pd.notna(per) else None
            data["roe"] = round(roe, 2) if pd.notna(roe) else None
        corp_info = stock.get_market_cap_by_date(fromdate=today_str, todate=today_str, ticker=ticker)
        if not corp_info.empty:
            market_cap_value = corp_info["시가총액"].iloc[0]
            if pd.notna(market_cap_value):
                data["market_cap"] = int(market_cap_value)
    except Exception as e:
        print("⚠️ fetch_stock_data 오류:", e)
    return data

def get_beta_values(stock_code: str):
    """
    1년치, 3년치 베타 값 반환
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
    try:
        end = datetime.today()
        start = end - timedelta(weeks=weeks)

        stock_data = yf.download(ticker, start=start, end=end)
        
        # 종목이 코스피인지 코스닥인지 확인
        market_index = "^KS11"  # 기본값은 KOSPI
        try:
            # 종목 정보에서 시장 구분 확인
            stock_info = yf.Ticker(ticker)
            info = stock_info.info
            if info and 'market' in info:
                market = info['market'].upper()
                if 'KOSDAQ' in market or 'KOSPI' in market:
                    if 'KOSDAQ' in market:
                        market_index = "^KSDAQ"  # KOSDAQ 지수
                        print(f"yfinance KOSDAQ 지수 사용: {ticker}")
                    else:
                        market_index = "^KS11"   # KOSPI 지수
                        print(f"yfinance KOSPI 지수 사용: {ticker}")
        except Exception as e:
            print(f"yfinance 종목 정보 확인 실패: {e}, 기본 KOSPI 지수 사용")
        
        market_data = yf.download(market_index, start=start, end=end)

        # None 체크 추가
        if stock_data is None or market_data is None:
            print(f"yfinance 데이터 다운로드 실패: {ticker}")
            return None
            
        if stock_data.empty or market_data.empty:
            print(f"yfinance 빈 데이터: {ticker}")
            return None

        # pandas Series로 변환하여 안전하게 처리
        stock_returns = pd.Series(stock_data['Close']).pct_change().dropna()
        market_returns = pd.Series(market_data['Close']).pct_change().dropna()

        # 최소 데이터 포인트 확인
        if len(stock_returns) < 10 or len(market_returns) < 10:
            print(f"yfinance 데이터 포인트 부족: {ticker}")
            return None

        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 10:
            print(f"yfinance 정렬된 데이터 포인트 부족: {ticker}")
            return None

        covariance = np.cov(aligned['Stock'], aligned['Market'])[0, 1]
        market_variance = np.var(aligned['Market'])
        
        if market_variance == 0:
            print(f"yfinance 시장 분산이 0: {ticker}")
            return None
            
        beta = covariance / market_variance

        return round(beta, 4)
    except Exception as e:
        print(f"yfinance 베타 계산 오류: {e}")
        return None

# pykrx 베타 계산 함수
def get_beta_pykrx(ticker: str, weeks: int):
    try:
        end = datetime.today()
        start = end - timedelta(weeks=weeks)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        
        # 개별 주식 데이터
        stock_data = stock.get_market_ohlcv_by_date(fromdate=start_str, todate=end_str, ticker=ticker)
        if stock_data.empty:
            print(f"pykrx 주식 데이터 없음: {ticker}")
            return None
            
        # 종목이 코스피인지 코스닥인지 확인
        market_data = None
        
        # 종목코드로 시장 구분 (pykrx 방식)
        try:
            # 코스피 종목: 000000~099999, 코스닥 종목: 100000~999999
            ticker_num = int(ticker)
            if ticker_num < 100000:  # 코스피
                market_tickers = ["KS11", "1001", "KS200"]  # KOSPI 관련 지수
                print(f"pykrx KOSPI 종목으로 판단: {ticker}")
            else:  # 코스닥
                market_tickers = ["KSDAQ", "KS11"]  # KOSDAQ 우선, 실패시 KOSPI
                print(f"pykrx KOSDAQ 종목으로 판단: {ticker}")
        except ValueError:
            # 숫자가 아닌 경우 기본값 사용
            market_tickers = ["KS11", "KSDAQ", "1001", "KS200"]
            print(f"pykrx 종목코드 파싱 실패, 기본 지수 사용: {ticker}")
        
        for market_ticker in market_tickers:
            try:
                market_data = stock.get_market_ohlcv_by_date(fromdate=start_str, todate=end_str, ticker=market_ticker)
                if not market_data.empty:
                    print(f"pykrx 지수 사용: {market_ticker} for {ticker}")
                    break
            except Exception as e:
                print(f"pykrx {market_ticker} 시도 실패: {e}")
                continue
                
        if market_data is None or market_data.empty:
            print("pykrx 지수 데이터 가져오기 실패")
            return None
            
        # 수익률 계산 - pandas Series로 안전하게 처리
        stock_returns = pd.Series(stock_data['종가']).pct_change().dropna()
        market_returns = pd.Series(market_data['종가']).pct_change().dropna()
        
        # 최소 데이터 포인트 확인
        if len(stock_returns) < 10 or len(market_returns) < 10:
            print("pykrx 데이터 포인트 부족")
            return None
        
        # 데이터 정렬
        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 10:  # 최소 데이터 포인트 확인
            print("pykrx 정렬된 데이터 포인트 부족")
            return None
            
        # 베타 계산
        covariance = np.cov(aligned['Stock'], aligned['Market'])[0, 1]
        market_variance = np.var(aligned['Market'])
        
        if market_variance == 0:
            print("pykrx 시장 분산이 0")
            return None
            
        beta = covariance / market_variance
        return round(beta, 4)
    except Exception as e:
        print(f"pykrx 베타 계산 오류: {e}")
        return None

# 부도 예측 데이터 가져오기
def get_default_data(ticker: str, NAME_TO_TICKER = {}):
    if ticker is None:
        return -1

    # NAME_TO_TICKER가 주어졌을 때 우선적으로 탐색
    if NAME_TO_TICKER:
        for _, data in NAME_TO_TICKER.items():
            code, is_defaulted = data
            # 코드와 티커가 완전히 일치하는지 뿐만 아니라, 앞뒤 공백 제거 및 대소문자 무시하여 비교
            if str(code).strip().upper() == str(ticker).strip().upper():
                return is_defaulted
        return -1
    # NAME_TO_TICKER가 비어있거나 None일 때 DB에서 조회
    else:
        import sqlite3
        conn = sqlite3.connect("rrdb.db")
        try:
            df_check = pd.read_sql(
                "SELECT is_defaulted FROM corp_info WHERE 종목코드 = ? LIMIT 1",
                conn,
                params=(ticker,)
            )
            if not df_check.empty:
                return df_check['is_defaulted'].iloc[0]
            else:
                return -1
        finally:
            conn.close()

# 챗봇 주요 로직
def chat_logic(req: ChatRequest, naver_client_id, naver_client_secret, NAME_TO_TICKER, model):
    """채팅 요청을 받아 주가, 기업정보, 뉴스 등을 종합해 답변을 생성하는 함수"""
    user_msg = req.message.strip()
    if not user_msg:
        return {"reply": "메시지를 입력해주세요."}

    ticker, stock_name = extract_ticker(user_msg, NAME_TO_TICKER)
    stock_info = build_stock_info(ticker) if ticker else None
    news_items: List[dict] = []
    data = fetch_stock_data(ticker) if ticker else {}
    beta_1y, beta_3y = get_beta_values(ticker) if ticker else (None, None)
    main_products = data.get("main_products")
    is_defaulted = get_default_data(ticker, NAME_TO_TICKER) if ticker else -1

    print(f"[DEBUG] 사용자 메시지: {user_msg}")
    print(f"[DEBUG] 추출된 ticker: {ticker}, 종목명: {stock_name}")
    print(f"[DEBUG] 부실 예측 결과: {is_defaulted}")
    print(f"[DEBUG] 주요 제품: {main_products}")
    print(f"[DEBUG] 수익률 - 1년: {data.get('return_1y')}%, 3년: {data.get('return_3y')}%")
    print(f"[DEBUG] 베타 - 1년: {beta_1y}, 3년: {beta_3y}")
    print(f"[DEBUG] PER: {data.get('per')}, ROE: {data.get('roe')}, 부채비율: {data.get('debt_ratio')}")

    if stock_name:
        news_items = get_latest_news_naver(stock_name, naver_client_id, naver_client_secret)
    else:
        news_items = []

    try:
        context_parts = [
            SYSTEM_PROMPT_TEMPLATE,
            CHAT_FORMAT_PROMPT,
            f"부실 예측 결과: {'부실' if is_defaulted == 1 else '정상' if is_defaulted == 0 else '정보 없음'}"
        ]

        if main_products:
            context_parts.append("주요 제품:\n" + main_products)

        prompt = "\n".join(context_parts) + f"\n사용자 질문: {user_msg}"
        response = model.generate_content(prompt)
        
        # 응답 검증
        if response and hasattr(response, 'text'):
            answer = response.text.strip()
        else:
            answer = "부도예측 결과\n응답을 생성하지 못했습니다.\n"

        if news_items:
            news_markdown = "\n\n**관련 뉴스**\n" + "\n".join(f"[{item['title']}]({item['link']}){{:target=\"_blank\"}}" for item in news_items)
            answer += news_markdown

    except Exception as e:
        print(f"[ERROR] Gemini API 오류: {e}")
        if "quota" in str(e).lower() or "limit" in str(e).lower() or "exceeded" in str(e).lower():
            answer = "**API 사용량 초과**\n\n현재 Gemini API 사용량이 초과되었습니다. 잠시 후 다시 시도해주세요.\n\n**부도예측 결과**\n정보를 불러오지 못했습니다.\n\n**주요 매출 제품**\n정보를 불러오지 못했습니다.\n\n**부가설명**\nAPI 사용량 초과로 인해 상세 정보를 제공할 수 없습니다."
        else:
            answer = "**오류 발생**\n\n정보를 불러오는 중 오류가 발생했습니다.\n\n**부도예측 결과**\n정보를 불러오지 못했습니다.\n\n**주요 매출 제품**\n정보를 불러오지 못했습니다.\n\n**부가설명**\n오류로 인해 상세 정보를 제공할 수 없습니다."

    print(f"[DEBUG] 최종 응답: {answer[:200]}...")  # 응답이 너무 길 수 있으므로 앞부분만 출력

    # numpy 타입을 JSON 직렬화 가능한 형태로 변환
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    result = {
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
        "beta_1y": beta_1y,
        "beta_3y": beta_3y,
        "is_defaulted": is_defaulted,
        "stock_info": stock_info,
        "news": news_items,
        "stock_code": ticker,
    }
    
    return convert_numpy_types(result)
