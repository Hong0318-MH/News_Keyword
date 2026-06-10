from __future__ import annotations

import pandas as pd
import streamlit as st

from news_similarity_engine import (
    Article,
    analyze_against_main,
    analyze_articles,
    collect_articles,
    fetch_article,
)


st.set_page_config(page_title="뉴스 유사도 검색 엔진", page_icon="N", layout="wide")


@st.cache_data(ttl=600, show_spinner=False)
def cached_fetch_article(url: str) -> Article:
    return fetch_article(url)


@st.cache_data(ttl=600, show_spinner=False)
def cached_collect_articles(portal_url: str, limit: int) -> list[Article]:
    return collect_articles(portal_url, limit=limit)


@st.cache_data(ttl=600, show_spinner=False)
def cached_analyze_against_main(
    main_article: Article,
    articles: list[Article],
    keyword_count: int,
) -> dict[str, object]:
    return analyze_against_main(main_article, articles, keyword_count=keyword_count)


@st.cache_data(ttl=600, show_spinner=False)
def cached_analyze_articles(articles: list[Article], keyword_count: int) -> dict[str, object]:
    return analyze_articles(articles, keyword_count=keyword_count)


st.title("뉴스 유사도 검색 엔진")

with st.sidebar:
    st.header("입력")
    main_mode = st.radio(
        "메인 기사 선택 방식",
        ["메인 링크 직접 지정", "자동 선정"],
        horizontal=False,
    )
    main_url = ""
    if main_mode == "메인 링크 직접 지정":
        main_url = st.text_input("기준 메인 기사 링크", placeholder="https://news.example.com/article/...")
    portal_url = st.text_input("비교할 포털/섹션 링크", placeholder="https://news.example.com/section/...")
    limit = st.slider("수집 기사 수", min_value=4, max_value=20, value=8)
    keyword_count = st.slider("핵심 키워드 수", min_value=5, max_value=20, value=10)
    run = st.button("분석 실행", type="primary")


def render_article(article: Article, similarity: float | None = None) -> None:
    title = article.title
    if similarity is not None:
        title = f"{title} · 유사도 {similarity:.4f}"
    with st.container(border=True):
        st.subheader(title)
        st.write(f"URL: {article.url}")
        preview = article.text[:450] + ("..." if len(article.text) > 450 else "")
        st.write(preview)


def run_analysis() -> None:
    if main_mode == "메인 링크 직접 지정":
        if not main_url or not portal_url:
            st.warning("기준 메인 기사 링크와 비교할 포털/섹션 링크를 모두 입력하세요.")
            return
        with st.spinner("기사 수집 중입니다."):
            main_article = cached_fetch_article(main_url)
            articles = cached_collect_articles(portal_url, limit)
        if not articles:
            st.error("비교할 기사를 찾지 못했습니다. 다른 링크를 입력해 주세요.")
            return
        with st.spinner("유사도 계산 중입니다."):
            result = cached_analyze_against_main(main_article, articles, keyword_count)
    else:
        if not portal_url:
            st.warning("비교할 포털/섹션 링크를 입력하세요.")
            return
        with st.spinner("기사 수집 중입니다."):
            articles = cached_collect_articles(portal_url, limit)
        if len(articles) < 2:
            st.error("분석 가능한 기사가 2개 이상 필요합니다. 다른 링크를 입력해 주세요.")
            return
        with st.spinner("유사도 계산 중입니다."):
            result = cached_analyze_articles(articles, keyword_count)

    main_article = result["main"]
    ranked = result["ranked"]
    keywords = result["keywords"]
    neighbors = result["word2vec_neighbors"]
    metadata = result["metadata"]

    assert isinstance(main_article, Article)

    st.header("기준 메인 기사")
    render_article(main_article)

    st.header("메인 기사와 유사한 기사 순위")
    rows = [
        {
            "순위": index,
            "제목": item.article.title,
            "TF-IDF 코사인 유사도": round(item.similarity, 4),
            "URL": item.article.url,
        }
        for index, item in enumerate(ranked, start=1)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for item in ranked:
        render_article(item.article, similarity=item.similarity)

    left, right = st.columns(2)
    with left:
        st.header("핵심 키워드")
        keyword_df = pd.DataFrame(
            [{"키워드": word, "TF-IDF 점수": round(score, 5)} for word, score in keywords]
        )
        st.dataframe(keyword_df, use_container_width=True, hide_index=True)

    with right:
        st.header("Word2Vec 유사도 검증")
        if neighbors:
            neighbor_rows = []
            for keyword, pairs in neighbors.items():
                for word, score in pairs:
                    neighbor_rows.append(
                        {"기준 키워드": keyword, "유사 단어": word, "Word2Vec cosine": round(score, 4)}
                    )
            st.dataframe(pd.DataFrame(neighbor_rows), use_container_width=True, hide_index=True)
        else:
            st.info("학습 가능한 토큰이 부족하면 Word2Vec 유사 단어가 표시되지 않을 수 있습니다.")

    with st.expander("유사도 검증용 보조 지표: Jaccard"):
        st.json(metadata.get("jaccard_scores", {}))


if run:
    run_analysis()
