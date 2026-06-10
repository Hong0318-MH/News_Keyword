# Codex Handoff

이 저장소는 자연어 처리 과목 과제용 뉴스 유사도 검색 엔진입니다. 노트북이나 웹 Codex에서 이어서 작업할 때는 이 파일을 먼저 읽으면 됩니다.

## 현재 상태

- GitHub 저장소: https://github.com/Hong0318-MH/News_Keyword
- 기본 브랜치: `master`
- 웹앱: Streamlit `app.py`
- 핵심 분석 모듈: `news_similarity_engine.py`
- 방법론 문서: `docs/methodology.md`
- 샘플 데이터: `sample_articles.json`

## 사용자 요구사항

- 사용자가 `기준 메인 기사 링크`와 `비교할 포털/섹션 링크`를 입력한다.
- 기준 메인 기사를 화면 위에 표시한다.
- 포털/섹션에서 수집한 기사들을 메인 기사와의 TF-IDF cosine 유사도 순으로 보여준다.
- 핵심 키워드를 형태소 분석 기반으로 추출한다.
- Word2Vec 방식으로 핵심 키워드 주변 유사 단어를 보여준다.
- 과제 평가 항목인 유사도 검색 방법론, 유사도 검증 목적, Word2Vec 유사도 검증 예시가 설명되어야 한다.

## 최근 반영한 변경

- 샘플 데이터 실행 체크박스 제거
- 첫 화면 안내 문구 제거
- 기본 수집 기사 수를 12개에서 8개로 줄임
- 최대 수집 기사 수를 30개에서 20개로 줄임
- 기사 본문 수집을 병렬 처리로 변경
- Word2Vec 내장 학습기의 반복 수와 negative sampling 수를 줄여 실행 시간을 단축
- Streamlit 캐시를 추가해 같은 링크를 다시 분석할 때 시간을 줄임

## 실행 방법

일반 환경:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

`streamlit` 명령이 안 먹으면:

```powershell
python -m streamlit run app.py
```

현재 데스크톱처럼 패키지를 `vendor`에 설치한 경우:

```powershell
.\run_app.ps1
```

## 검증 명령

```powershell
python -m unittest discover -s tests
python -m py_compile app.py news_similarity_engine.py
```

## 이어서 작업할 때 주의

- `vendor/`는 `.gitignore`에 포함되어 있으므로 GitHub에 올리지 않는다.
- Python 3.10, 3.11, 3.12 환경이 가장 무난하다.
- Python 3.14에서는 `gensim` 빌드가 실패할 수 있어서, 현재 코드는 순수 Python Skip-gram Word2Vec fallback을 사용한다.
- 실제 뉴스 사이트는 HTML 구조가 달라 본문 추출 품질이 다를 수 있다.
