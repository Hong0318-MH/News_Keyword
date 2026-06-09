# 뉴스 유사도 검색 엔진

자연어 처리 과목 과제용 프로젝트입니다. 기준으로 삼을 메인 기사 링크와 뉴스 포털/섹션 링크를 입력하면, 포털에서 수집한 기사들을 메인 기사와의 유사도 순으로 정렬합니다. 메인 기사 링크를 직접 지정하지 않는 경우에는 기사 간 평균 유사도가 가장 높은 기사를 자동으로 메인 기사로 선정합니다.

## 기능

- 뉴스 포털/섹션 URL에서 기사 링크 후보 수집
- 기사 본문 추출
- 사용자가 입력한 메인 기사 링크를 기준 문서로 고정
- TF-IDF 코사인 유사도 기반 유사 기사 랭킹
- 자동 모드에서 TF-IDF 평균 유사도 기반 메인 기사 선정
- 형태소 분석 기반 핵심 키워드 추출
- 내장 Skip-gram Word2Vec 기반 키워드 유사도 검증
- Jaccard 유사도 보조 검증

## 설치

```bash
python -m pip install -r requirements.txt
```

## 실행

웹 화면:

```bash
streamlit run app.py
```

현재 PC처럼 패키지를 `vendor` 폴더에 설치한 경우:

```powershell
.\run_app.ps1
```

샘플 데이터 CLI:

```bash
python news_similarity_engine.py --sample
```

뉴스 포털/섹션 링크 자동 분석:

```bash
python news_similarity_engine.py --url "뉴스_포털_또는_섹션_URL" --limit 12
```

메인 기사 링크를 직접 지정하는 분석:

```bash
python news_similarity_engine.py --main-url "기준_메인_기사_URL" --url "뉴스_포털_또는_섹션_URL" --limit 12
```

## 과제 평가 항목 대응

- 유사도 검색 방법론: TF-IDF 코사인 유사도를 사용해 기준 메인 기사와 후보 기사 사이의 유사도를 계산합니다.
- 어떤 유사도 검증용인지: 관련 기사 검색용은 TF-IDF cosine, 토큰 집합 겹침 검증은 Jaccard, 키워드 의미 관계 검증은 Word2Vec cosine입니다.
- Word2Vec 유사도 검증: 핵심 키워드를 기준으로 주변 단어의 벡터 코사인 유사도 예시를 출력합니다.

참고: Python 3.14 환경에서는 `gensim` 빌드가 실패할 수 있어, 이 프로젝트는 과제 시연이 가능하도록 순수 Python Skip-gram Word2Vec 구현을 포함합니다.
