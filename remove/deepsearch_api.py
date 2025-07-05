import os
import requests
from typing import List, Optional
from datetime import datetime, timedelta # 뉴스 날짜 범위를 위해 추가

# BASE_URL을 변경하여 도메인만 포함하고 버전은 각 함수에서 명시하도록 합니다.
BASE_URL = "https://api-v2.deepsearch.com"


def _headers():
    key = os.getenv("DEEPSEARCH_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEARCH_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def search_symbol(company_name: str) -> Optional[str]:
    """Return symbol_id for given company name."""
    # 문서 이미지에 따르면 '/v2/markets/symbols'가 심볼 관련 엔드포인트입니다.
    # 하지만 'company_name'으로 직접 검색하는 파라미터는 명확히 나와있지 않습니다.
    # 'query' 파라미터는 이전의 가설적인 엔드포인트에서 사용된 것입니다.
    # 이 부분이 여전히 404를 유발할 수 있습니다.
    url = f"{BASE_URL}/v2/markets/symbols"
    params = {"query": company_name} # 이 'query' 파라미터 이름이 정확한지는 추가 확인 필요
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=10)
        if resp.ok:
            data = resp.json()
            items = data.get("items") # API 응답 구조에 따라 'items' 대신 다른 키일 수 있음
            if items:
                # API 응답 구조에 따라 symbol_id를 추출하는 로직이 달라질 수 있습니다.
                # 현재는 첫 번째 아이템의 'symbol_id'를 가정합니다.
                symbol = items[0].get("symbol_id") # 실제 응답 데이터 구조 확인 필요
                print("DeepSearch search_symbol", company_name, symbol)
                return symbol
            print("DeepSearch search_symbol no items found for", company_name)
        else:
            print(f"DeepSearch search_symbol error: Status Code {resp.status_code}, Response: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"DeepSearch search_symbol request failed: {e}")
    return None


def get_company_overview(symbol_id: str) -> Optional[str]:
    """Fetch overview/description text for the symbol."""
    # 이 엔드포인트에 대한 정보는 여전히 문서에 없습니다.
    # 이전과 동일한 가설적인 엔드포인트를 유지합니다.
    url = f"{BASE_URL}/symbol/{symbol_id}/info"
    try:
        resp = requests.get(url, headers=_headers(), timeout=10)
        if resp.ok:
            data = resp.json()
            overview = data.get("overview")
            print("DeepSearch get_company_overview", symbol_id, bool(overview))
            return overview
        print(f"DeepSearch get_company_overview error: Status Code {resp.status_code}, Response: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"DeepSearch get_company_overview request failed: {e}")
    return None


def get_latest_news(symbol_id: str, limit: int = 2) -> List[dict]:
    """Return latest news items with title and url."""
    # 문서 이미지에 따르면 '/v1/articles'가 뉴스/기사 엔드포인트입니다.
    # 'symbols' 파라미터와 날짜 범위를 사용합니다.
    url = f"{BASE_URL}/v1/articles"
    today = datetime.now()
    date_to_str = today.strftime("%Y-%m-%d")
    # 지난 7일간의 뉴스 (혹은 적절한 기간으로 설정)
    date_from_str = (today - timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "symbols": symbol_id, # 'symbol_id'를 'symbols' 파라미터로 사용 (복수형)
        "date_from": date_from_str,
        "date_to": date_to_str,
        "size": limit # API 문서에 'size' 파라미터가 명시되어 있지 않으면 작동하지 않을 수 있습니다.
                      # 'limit'이라는 이름이 문서에 있다면 'limit'으로 변경 필요
    }
    result = []
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=10)
        if resp.ok:
            data = resp.json()
            # 문서에 따르면 'articles' 엔드포인트는 'items' 키 아래에 리스트를 반환할 것으로 예상됩니다.
            for n in data.get("items", [])[:limit]: # 실제 응답 데이터 구조 확인 필요: 'title', 'link' 키 존재 여부
                result.append({"title": n.get("title"), "link": n.get("link")})
            print("DeepSearch get_latest_news", symbol_id, len(result))
        else:
            print(f"DeepSearch get_latest_news error: Status Code {resp.status_code}, Response: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"DeepSearch get_latest_news request failed: {e}")
    return result


def parse_main_products(text: str, limit: int = 2) -> List[str]:
    """Extract major products from overview text."""
    if not text:
        return []
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(keyword in line for keyword in ["주요", "제품", "서비스"]):
            candidates.append(line)
    if not candidates:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        candidates.extend(sentences)
    return candidates[:limit]