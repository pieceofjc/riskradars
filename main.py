# Standard library imports
import io
import os
from datetime import datetime, timedelta

# Third-party library imports
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
import matplotlib.pyplot as plt

# Local project imports
from chat import ChatRequest, chat_logic
from predictor import AdvancedStockPredictor

# Load environment variables
load_dotenv()
naver_client_id = os.getenv("NAVER_CLIENT_ID")
naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
google_api_key = os.getenv("GOOGLE_API_KEY")

# 환경 변수 검증
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Configure Google Generative AI API key
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# 종목명과 종목코드 및 부도여부를 매핑하는 딕셔너리
NAME_TO_TICKER = {}
# 예시: CSV에서 불러와 초기화 가능
# df = pd.read_csv('corp_list2.csv', encoding='utf-8')
# for index, row in df.iterrows():
#     NAME_TO_TICKER[row['회사명']] = [row['종목코드'], row['is_defaulted']]

# --------------------
# FastAPI 앱 생성
# --------------------
app = FastAPI()

# 템플릿과 정적 파일 설정
templates = Jinja2Templates(directory="template")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------
# 메인 페이지 라우팅
# --------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

# --------------------
# 사용자의 질문 처리
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    return chat_logic(req, naver_client_id, naver_client_secret, NAME_TO_TICKER, model)

# --------------------
# 주가 예측 차트 이미지 생성
# --------------------
@app.get("/plot.png")
async def get_plot(stock_code: str = Query(..., alias="ticker")):
    try:
        if stock_code is None or stock_code == "default":
            return {"error": "해당 회사의 종목코드를 찾을 수 없습니다."}

        predictor = AdvancedStockPredictor(stock_code)
        three_years_ago = datetime.today() - timedelta(days=365 * 3)

        # 데이터 로드 및 예측 수행
        predictor.load_data(start_date=three_years_ago)
        forecasts = predictor.comprehensive_forecast()
        trend_signal, trend_changes, ma_short, ma_long = predictor.detect_regime_changes()
        var_95, max_drawdown, drawdown = predictor.calculate_downside_risk()

        # 그래프 버퍼 준비
        buf = io.BytesIO()
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # 주가 및 이동평균선 그리기
        ax1.plot(predictor.df_monthly.index, predictor.df_monthly.values, label='주가', color='black', linewidth=2)
        ax1.plot(ma_short.index, ma_short.values, label='6개월 이동평균선', alpha=0.7, color='blue')
        ax1.plot(ma_long.index, ma_long.values, label='1년 이동평균선', alpha=0.7, color='orange')

        # 예측 구간 음영 표시
        ax1.fill_between(forecasts['dates'], forecasts['monte_carlo'][1], forecasts['monte_carlo'][0],
                         alpha=0.3, color='red', label='예측범위')

        # 그래프 스타일 설정
        ax1.set_ylabel('주가(원)', fontsize=18)
        ax1.set_xlabel('날짜', fontsize=18)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.legend(fontsize=18)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # PNG 이미지 스트리밍 응답 반환
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("🔥 Plot 오류:", e)
        # 기본 차트 반환
        try:
            buf = io.BytesIO()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, '차트를 불러올 수 없습니다', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        except:
            return {"error": "차트 생성에 실패했습니다."}

# --------------------
# 앱 실행
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
