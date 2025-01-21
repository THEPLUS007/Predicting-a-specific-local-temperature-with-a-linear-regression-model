import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# CSV 파일 경로
file_path = r'C:\Users\dghlu\OneDrive\바탕 화면\수과탐 인공지능 시뮬\OBS_ASOS_DD_20230521145546.csv'

# CSV 파일 로드
data = pd.read_csv(file_path, encoding='euc-kr')

# '일시' 컬럼을 날짜 형식으로 변환
data['일시'] = pd.to_datetime(data['일시'])

# 입력 데이터(X)와 타깃 데이터(Y)로 분할
X = data['일시'].map(datetime.toordinal)
Y = data['평균기온']

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X.values.reshape(-1, 1), Y)

# 현재 날짜와 시간
now = datetime.now()

# 다음 날짜 계산
next_date = now + timedelta(days=1)

# 다음 날짜의 기온 예측
next_date_ordinal = next_date.toordinal()
prediction = model.predict([[next_date_ordinal]])

next_date_formatted = next_date.strftime('%Y-%m-%d %H:%M:%S').split('.')[0]
print("다음 날짜:", next_date_formatted)
print("예측 기온:", prediction)