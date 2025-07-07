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
역할 정의
당신은 주식과 회사 정보를 쉽고 명확하게 설명해주는 전문 봇입니다. 복잡한 금융 용어나 개념을 일반인도 이해할 수 있도록 쉬운 표현으로 설명하는 것이 목표입니다.
핵심 원칙

정보 출처: 부실 예측 결과 정보는 내가 준 데이터를 무조건 사용하고, 다른 정보는 다트(DART)에서 가져온 데이터를 기반으로 제공
대상 기업: 코스피, 코스닥 상장 기업만 대응 (해당되지 않는 경우 재질문 요청)
데이터 우선: 제공된 데이터가 있으면 반드시 답변에 포함하고 활용
완전한 답변: 상냥하고 적극적으로 모든 요청에 대해 검색하여 정보 제공

예외 처리

코스피/코스닥 비상장 기업의 경우: "죄송하지만 코스피, 코스닥 상장 기업에 대해서만 정보를 제공할 수 있습니다. 다른 기업으로 다시 질문해주세요."
데이터 제공 우선순위: 제공된 실제 데이터 > 일반적인 정보
"""

# 챗봇 답변 형식
CHAT_FORMAT_PROMPT = """

답변 형식은 아래 입니다
**{해당 기업 이름}의 정보를 제공합니다**
**부도 예측 결과**
예측 결과에 따른 자연스러운 설명 (제공된 데이터 기반)

**주요 매출 제품**
**주요 제품 또는 서비스 1**
제품/서비스에 대한 2줄 정도의 쉬운 설명, 못 찾으면 못 찾았다고 해줘
**주요 제품 또는 서비스 2**
제품/서비스에 대한 2줄 정도의 쉬운 설명, 못 찾으면 못 찾았다고 해줘

**부가설명**
회사 설립일과 기업 개요를 간단히 설명, 못 찾으면 못 찾았다고 해줘


작성 가이드라인은 이것입니다
마크다운 형식으로 깔끔하게 작성
볼드체(**) 제목 형식은 반드시 유지
전문 용어는 쉬운 표현으로 풀어서 설명
제공된 데이터가 없을 때만 "정보 없음" 표시
모든 유명 기업도 제공된 데이터 기반으로 답변
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
            cursor.execute("SELECT stock_code, 회사명 FROM corp_info WHERE 회사명 = ?", (keyword,))
            row = cursor.fetchone()
            if row:
                return row[0], row[1]

            # 부분일치 검색
            cursor.execute("SELECT stock_code, 회사명 FROM corp_info WHERE 회사명 LIKE ?", (f"%{keyword}%",))
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
            
        # 코스피/코스닥 구분하여 지수 데이터 가져오기
        try:
            # 코스피 지수 데이터 가져오기 (다른 방법 시도)
            market_data = stock.get_market_ohlcv_by_date(fromdate=start_str, todate=end_str, ticker="KS11")
            print(f"코스피 지수 사용: {ticker}")
        except Exception as e:
            print(f"KS11 시도 실패: {e}")
            try:
                # 코스닥 지수 데이터 가져오기
                market_data = stock.get_market_ohlcv_by_date(fromdate=start_str, todate=end_str, ticker="KSDAQ")
                print(f"코스닥 지수 사용: {ticker}")
            except Exception as e:
                print(f"KSDAQ 시도 실패: {e}")
                try:
                    # 다른 지수 티커 시도
                    market_data = stock.get_market_ohlcv_by_date(fromdate=start_str, todate=end_str, ticker="1001")
                    print(f"코스피 지수(1001) 사용: {ticker}")
                except Exception as e:
                    print(f"1001 시도 실패: {e}")
                    print("pykrx 지수 데이터 가져오기 실패")
                    return None
                
        if market_data.empty:
            print("pykrx 지수 데이터 없음")
            return None
            
        # 수익률 계산
        stock_returns = stock_data['종가'].pct_change().dropna()
        market_returns = market_data['종가'].pct_change().dropna()
        
        # 데이터 정렬
        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 10:  # 최소 데이터 포인트 확인
            print("pykrx 데이터 포인트 부족")
            return None
            
        # 베타 계산
        covariance = np.cov(aligned['Stock'], aligned['Market'])[0, 1]
        market_variance = np.var(aligned['Market'])
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
                "SELECT is_defaulted FROM corp_info WHERE stock_code = ? LIMIT 1",
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

    if stock_name:
        news_items = get_latest_news_naver(stock_name, naver_client_id, naver_client_secret)
    else:
        news_items = []

    try:
        context_parts = [
            SYSTEM_PROMPT_TEMPLATE,
            CHAT_FORMAT_PROMPT,
            f"부실 예측 결과: {'부실 기업' if is_defaulted == 1 else '정상 기업' if is_defaulted == 0 else '예측 결과가 없스니다 12월 결산이 아닐 수 있습니다'}"
        ]

        if main_products:
            context_parts.append("주요 제품:\n" + main_products)

        prompt = "\n".join(context_parts) + f"\n사용자 질문: {user_msg}"
        response = model.generate_content(prompt)
        
        # 응답 검증
        if response and hasattr(response, 'text'):
            answer = response.text.strip()
        else:
            answer = ""

        # Gemini 응답이 부실 안내문이면 직접 답변 생성
        if not answer or any(x in answer for x in [
            "부실 예측 결과가 없습니다",
            "답변이 어렵",
            "정보가 없습니다",
            "질문하신",
            "제한적",
            "불가",
            "어렵",
            "미제공",
            "불가능",
            "알 수 없",
            "바로 가져올 수 없습니다"
        ]):
            answer = f"""**{stock_name or '기업명 미확인'}의 정보를 제공합니다**\n\n**부도 예측 결과**\n해당 기업은 {'부실 기업' if is_defaulted == 1 else '정상 기업' if is_defaulted == 0 else '예측 결과가 없습니다. 12월 결산이 아닐 수 있습니다.'}입니다. 투자에 주의가 필요합니다.\n\n**주요 매출 제품**\n{main_products or '정보를 불러올 없습니다.'}\n\n**부가 설명**\n{stock_info['description'] if stock_info and 'description' in stock_info else '설립일 및 개요 정보를 불러올 수 없습니다.'}\n"""

        if news_items:
            news_markdown = "\n\n**관련 뉴스**\n" + "\n".join(f"[{item['title']}]({item['link']})" for item in news_items)
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
