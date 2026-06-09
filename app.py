from __future__ import annotations

import pandas as pd
import streamlit as st

from news_similarity_engine import (
    Article,
    analyze_against_main,
    analyze_articles,
    collect_articles,
    fetch_article,
    load_sample_articles,
)


st.set_page_config(page_title="뉴스 유사도 검색 엔진", page_icon="N", layout="wide")

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
    limit = st.slider("수집 기사 수", min_value=4, max_value=30, value=12)
    keyword_count = st.slider("핵심 키워드 수", min_value=5, max_value=30, value=10)
    use_sample = st.checkbox("샘플 데이터로 실행", value=not bool(portal_url or main_url))
    run = st.button("분석 실행", type="primary")

if main_mode == "메인 링크 직접 지정":
    st.caption("입력한 메인 기사 링크를 기준 문서로 고정하고, 포털/섹션에서 수집한 기사들을 유사도 순으로 정렬합니다.")
else:
    st.caption("메인 기사는 기사 간 TF-IDF 코사인 유사도 평균이 가장 높은 문서로 자동 선정합니다.")


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
    if use_sample:
        articles = load_sample_articles()
        if main_mode == "메인 링크 직접 지정":
            result = analyze_against_main(articles[0], articles[1:], keyword_count=keyword_count)
        else:
            result = analyze_articles(articles, keyword_count=keyword_count)
    else:
        if main_mode == "메인 링크 직접 지정":
            if not main_url or not portal_url:
                st.warning("기준 메인 기사 링크와 비교할 포털/섹션 링크를 모두 입력하세요.")
                return
            with st.spinner("기준 메인 기사와 비교 기사들을 수집하는 중입니다."):
                main_article = fetch_article(main_url)
                articles = collect_articles(portal_url, limit=limit)
            if not articles:
                st.error("포털/섹션 링크에서 비교할 기사를 찾지 못했습니다. 다른 링크를 입력해 주세요.")
                return
            with st.spinner("메인 기사 기준 유사도를 계산하는 중입니다."):
                result = analyze_against_main(main_article, articles, keyword_count=keyword_count)
        else:
            if not portal_url:
                st.warning("뉴스 포털/섹션 링크를 입력하거나 샘플 데이터를 선택하세요.")
                return
            with st.spinner("기사 링크를 찾고 본문을 수집하는 중입니다."):
                articles = collect_articles(portal_url, limit=limit)
            if len(articles) < 2:
                st.error("분석 가능한 기사가 2개 이상 필요합니다. 다른 링크를 입력해 주세요.")
                return
            with st.spinner("메인 기사 자동 선정과 유사도 계산을 진행하는 중입니다."):
                result = analyze_articles(articles, keyword_count=keyword_count)

    main_article = result["main"]
    ranked = result["ranked"]
    keywords = result["keywords"]
    neighbors = result["word2vec_neighbors"]
    metadata = result["metadata"]

    assert isinstance(main_article, Article)

    st.header("기준 메인 기사")
    if main_mode == "메인 링크 직접 지정":
        st.success("아래 기사를 기준 문서로 고정했습니다.")
    else:
        st.success("기사 묶음 안에서 평균 유사도가 가장 높은 기사를 메인 기사로 선정했습니다.")
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
else:
    st.info("왼쪽에서 링크를 입력하거나 샘플 데이터로 분석을 실행하세요.")
