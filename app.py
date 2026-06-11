from __future__ import annotations

import pandas as pd
import streamlit as st
from urllib.parse import urlparse

from news_similarity_engine import (
    Article,
    analyze_against_main,
    collect_articles,
    fetch_article,
    is_probably_article_text,
)


st.set_page_config(page_title="뉴스 유사도 검색 엔진", page_icon="N", layout="wide")

CACHE_VERSION = "noun-keywords-v1"


@st.cache_data(ttl=600, show_spinner=False)
def cached_fetch_article(url: str, cache_version: str = CACHE_VERSION) -> Article:
    return fetch_article(url)


@st.cache_data(ttl=600, show_spinner=False)
def cached_collect_articles(
    portal_url: str,
    limit: int,
    cache_version: str = CACHE_VERSION,
) -> list[Article]:
    return collect_articles(portal_url, limit=limit)


st.title("뉴스 유사도 검색 엔진")

with st.sidebar:
    with st.form("analysis_form"):
        st.header("입력")
        main_url = st.text_input("기준 메인 기사 링크", placeholder="https://news.example.com/article/...")
        portal_url = st.text_input("비교할 포털/섹션 링크", placeholder="https://news.example.com/section/...")
        limit = st.slider("수집 기사 수", min_value=4, max_value=20, value=8)
        keyword_count = st.slider("핵심 키워드 수", min_value=5, max_value=20, value=10)
        run = st.form_submit_button("분석 실행", type="primary", width="stretch")


def render_article(article: Article, similarity: float | None = None) -> None:
    title = article.title
    if similarity is not None:
        title = f"{title} · 유사도 {similarity:.4f}"
    with st.container(border=True):
        st.subheader(title)
        st.write(f"URL: {article.url}")
        parsed_url = urlparse(article.url)
        if parsed_url.scheme in {"http", "https"} and parsed_url.netloc:
            st.link_button("기사 열기", article.url)
        preview = article.text[:450] + ("..." if len(article.text) > 450 else "")
        st.write(preview)


def run_analysis() -> dict[str, object] | None:
    if not main_url or not portal_url:
        st.warning("기준 메인 기사 링크와 비교할 포털/섹션 링크를 모두 입력하세요.")
        return None
    with st.spinner("기사 수집 중입니다."):
        main_article = cached_fetch_article(main_url)
        if not is_probably_article_text(main_article.text):
            st.error("기준 메인 기사 본문을 제대로 추출하지 못했습니다. 실제 기사 본문 URL인지 확인해 주세요.")
            return None
        articles = cached_collect_articles(portal_url, limit)
    if not articles:
        st.error("비교할 기사를 찾지 못했습니다. 다른 링크를 입력해 주세요.")
        return None
    with st.spinner("유사도 계산 중입니다."):
        result = analyze_against_main(main_article, articles, keyword_count)

    return result


def render_result(result: dict[str, object]) -> None:
    main_article = result["main"]
    ranked = result["ranked"]
    keywords = result["keywords"]
    keyword_doc_counts = result.get("keyword_doc_counts", {})
    word2vec_neighbors = result.get("word2vec_neighbors", {})
    metadata = result["metadata"]

    assert isinstance(main_article, Article)

    st.header("기준 메인 기사")
    render_article(main_article)

    st.header("메인 기사와 유사한 기사 순위")
    rows = [
        {
            "순위": index,
            "제목": item.article.title,
            "종합 유사도": round(item.similarity, 4),
            "URL": item.article.url,
        }
        for index, item in enumerate(ranked, start=1)
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    for item in ranked:
        render_article(item.article, similarity=item.similarity)

    st.header("핵심 키워드")
    keyword_rows = []
    for word, score in keywords[:10]:
        keyword_rows.append(
            {
                "키워드": word,
                "등장 기사 수": keyword_doc_counts.get(word, ""),
                "키워드 점수": round(score, 4),
            }
        )
    st.dataframe(pd.DataFrame(keyword_rows), width="stretch", hide_index=True)

    st.header("Word2Vec 유사 단어")
    neighbor_rows = []
    for keyword, pairs in word2vec_neighbors.items():
        for word, similarity in pairs[:5]:
            neighbor_rows.append(
                {
                    "기준 키워드": keyword,
                    "유사 단어": word,
                    "유사도": round(similarity, 4),
                }
            )
    if neighbor_rows:
        st.dataframe(pd.DataFrame(neighbor_rows), width="stretch", hide_index=True)
    else:
        st.info("Word2Vec으로 표시할 유사 단어가 충분하지 않습니다.")

    with st.expander("유사도 검증용 보조 지표: Jaccard"):
        st.json(metadata.get("jaccard_scores", {}))

    with st.expander("종합 유사도 세부 점수"):
        st.json(
            {
                "weights": metadata.get("similarity_weights", {}),
                "components": metadata.get("score_components", {}),
            }
        )


if run:
    st.session_state.pop("last_result", None)
    try:
        result = run_analysis()
    except Exception as exc:
        st.error(f"분석 중 오류가 발생했습니다: {exc}")
    else:
        if result is not None:
            st.session_state["last_result"] = result

if "last_result" in st.session_state:
    render_result(st.session_state["last_result"])
