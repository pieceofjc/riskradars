import pandas as pd
import sqlite3

# CSV 파일 읽기
df = pd.read_csv("corp_result_list.csv", dtype={'stock_code':str})

# SQLite DB 연결 (없으면 자동 생성됨)
conn = sqlite3.connect("rrdb.db")

# DataFrame을 SQLite 테이블로 저장
df.to_sql("corp_info", conn, if_exists="replace", index=False)

df_check = pd.read_sql("SELECT * FROM corp_info LIMIT 5", conn)
print(df_check)

conn.close()