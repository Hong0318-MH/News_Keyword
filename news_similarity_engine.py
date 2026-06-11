from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen


KOREAN_STOPWORDS = {
    "그리고", "그러나", "하지만", "또한", "관련", "대한", "위해", "통해", "이번", "지난", "있는",
    "한다", "했다", "된다", "됐다", "에서", "으로", "에게", "까지", "부터", "보다", "처럼", "기자",
    "뉴스", "사진", "제공", "이라며", "다고", "라고", "것으로", "것이다", "있다", "없는", "등을",
    "대하", "대해", "밝히", "밝혔", "밝혔다", "밝히고", "이후", "이전", "통한", "위한", "것", "수", "등",
    "및", "말하", "전하", "설명", "설명했다", "강조", "예정", "계획", "기준", "현재", "최근", "이날",
}

SIMILARITY_WEIGHTS = {
    "content_tfidf": 0.54,
    "title_tfidf": 0.18,
    "main_keyword_coverage": 0.11,
    "token_jaccard": 0.05,
    "word2vec_document": 0.12,
}

MAIN_KEYWORD_OVERLAP_TOP_N = 20
MIN_KEYWORD_LENGTH = 2
MAX_COMPOUND_TOKEN_LENGTH = 12
COMPOUND_KEYWORD_PARTS = (
    "인공지능", "생성형", "반도체", "데이터센터", "데이터", "센터", "프로야구",
    "자연어", "자연어처리", "컴퓨터비전", "클라우드", "서비스", "정책", "교육",
)

KOREAN_ENDING_SUFFIXES = (
    "입니다", "합니다", "했다", "한다", "됐다", "된다", "된다며", "라고", "다고", "였다",
    "했다며", "밝혔다", "밝히고", "설명했다", "강조했다", "전했다", "말했다",
)


@dataclass
class Article:
    title: str
    url: str
    text: str


@dataclass
class RankedArticle:
    article: Article
    similarity: float


class SimpleWord2Vec:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def __contains__(self, word: str) -> bool:
        return word in self.vectors

    def most_similar(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        if word not in self.vectors:
            return []
        scores = []
        for other in self.vectors:
            if other == word:
                continue
            scores.append((other, vector_cosine(self.vectors[word], self.vectors[other])))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:topn]


class NewsHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._current_href: str | None = None
        self._current_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        self._tag_stack.append(tag)
        if tag == "a":
            self._current_href = attrs_dict.get("href")
            self._current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href:
            text = normalize_space(" ".join(self._current_link_text))
            if text:
                self.links.append((self._current_href, text))
            self._current_href = None
            self._current_link_text = []
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = normalize_space(data)
        if not text:
            return
        if self._current_href is not None:
            self._current_link_text.append(text)
        if self._tag_stack and self._tag_stack[-1] == "title":
            self.title_parts.append(text)
        if self._tag_stack and self._tag_stack[-1] in {"p", "article", "h1", "h2", "h3"}:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return normalize_space(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        return normalize_space(" ".join(self.text_parts))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


NOISE_TEXT_PATTERNS = (
    "뉴스 이용 설정",
    "기사 배열은 자동 클러스터링",
    "각 언론사의 가장 많이 본 기사",
    "서비스 정책에 따라",
    "많이 본 뉴스",
    "네이버 AI 뉴스 알고리즘",
    "뉴스 추천 알고리즘",
    "클립 이슈 NOW",
    "언론사에서 직접 선별한 이슈",
    "함께 볼만한",
    "구독",
    "공유",
    "댓글",
    "무단전재",
    "재배포 금지",
    "저작권자",
    "기자 페이지",
    "기사제보",
    "Copyright",
)

KOREAN_PARTICLE_SUFFIXES = (
    "으로부터", "에서", "에게", "까지", "부터", "으로", "라고", "다고",
    "은", "는", "이", "가", "을", "를", "과", "와", "에", "의", "로", "도", "만",
)


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            parsed.query,
            "",
        )
    )


def normalize_article_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host in {"news.naver.com", "n.news.naver.com"}:
        match = re.search(r"/mnews/article/(\d{3})/(\d+)", parsed.path)
        if match:
            return f"https://n.news.naver.com/mnews/article/{match.group(1)}/{match.group(2)}"
        query = parse_qs(parsed.query)
        oid = query.get("oid", [""])[0]
        aid = query.get("aid", [""])[0]
        if parsed.path == "/main/read.naver" and oid and aid:
            return f"https://n.news.naver.com/mnews/article/{oid}/{aid}"
    return canonical_url(url)


def fetch_html(url: str, timeout: int = 6) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 NewsKeywordSimilarityBot/1.0",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        content_type = response.headers.get("Content-Type", "")
    charset_match = re.search(r"charset=([\w-]+)", content_type, re.I)
    charset = charset_match.group(1) if charset_match else "utf-8"
    return raw.decode(charset, errors="replace")


def parse_html(html: str) -> NewsHTMLParser:
    parser = NewsHTMLParser()
    parser.feed(html)
    return parser


def discover_article_urls(portal_url: str, limit: int = 15) -> list[str]:
    parser = parse_html(fetch_html(portal_url))
    urls: list[str] = []
    seen: set[str] = set()
    for href, anchor_text in parser.links:
        absolute = urljoin(portal_url, href)
        normalized = normalize_article_url(absolute)
        if normalized in seen or not looks_like_article_url(absolute, anchor_text):
            continue
        seen.add(normalized)
        urls.append(normalized)
        if len(urls) >= limit:
            break
    return urls


def looks_like_article_url(url: str, anchor_text: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if len(anchor_text) < 8:
        return False
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parse_qs(parsed.query)

    if host in {"news.naver.com", "n.news.naver.com"}:
        if re.search(r"/mnews/article/\d{3}/\d+", path):
            return True
        return path == "/main/read.naver" and bool(query.get("oid")) and bool(query.get("aid"))

    blocked = ("section", "ranking", "cluster", "photo", "video", "weather", "election", "journalist")
    if any(part in path for part in blocked):
        return False

    has_article_word = any(pattern in path for pattern in ("article", "view", "read", "news"))
    has_identifier = bool(re.search(r"\d{4,}", path)) or any(key in query for key in ("aid", "oid", "id", "idx", "no"))
    return has_article_word and has_identifier


def fetch_article(url: str) -> Article:
    html = fetch_html(url)
    return extract_article_from_html(url, html)


def clean_title(title: str) -> str:
    title = normalize_space(title)
    title = re.sub(r"\s*[:|-]\s*네이버 뉴스\s*$", "", title)
    title = re.sub(r"\s*[-|]\s*[^-|]{1,20}\s*$", "", title)
    return normalize_space(title)


def clean_article_text(text: str) -> str:
    lines = []
    for raw_line in re.split(r"[\r\n]+", text):
        line = normalize_space(raw_line)
        if len(line) < 8:
            continue
        if any(pattern in line for pattern in NOISE_TEXT_PATTERNS):
            continue
        lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\[[^\]]{1,20}\]", " ", text)
    text = re.sub(r"\([^)]+기자\)", " ", text)
    return normalize_space(text)


def noise_hit_count(text: str) -> int:
    return sum(1 for pattern in NOISE_TEXT_PATTERNS if pattern in text)


def is_probably_article_text(text: str) -> bool:
    if len(text) < 120:
        return False
    korean_chars = len(re.findall(r"[가-힣]", text))
    if korean_chars < 60:
        return False
    if "뉴스 이용 설정" in text[:120] or "기사 배열은 자동 클러스터링" in text:
        return False
    return noise_hit_count(text) < 3


def extract_article_from_html(url: str, html: str) -> Article:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        parser = parse_html(html)
        return Article(title=clean_title(parser.title or url), url=normalize_article_url(url), text=clean_article_text(parser.text))

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "iframe", "svg"]):
        tag.decompose()

    title = extract_title_from_soup(soup) or url
    candidates: list[tuple[int, str]] = []
    selectors = (
        "#dic_area",
        "#newsct_article",
        ".newsct_article",
        "#articeBody",
        "#articleBody",
        "#articleBodyContents",
        "[itemprop='articleBody']",
        "article",
        ".article_body",
        ".article-body",
        ".article_view",
        ".article-view",
        ".article_content",
        ".article-content",
        ".news_article",
        ".news-content",
        ".view_text",
        ".view_cont",
        ".article_txt",
    )
    for selector in selectors:
        for node in soup.select(selector):
            for unwanted in node.select("script, style, noscript, nav, footer, aside, iframe, button"):
                unwanted.decompose()
            text = clean_article_text(node.get_text("\n", strip=True))
            if text:
                score = len(text) - (noise_hit_count(text) * 500)
                candidates.append((score, text))

    if candidates:
        text = max(candidates, key=lambda item: item[0])[1]
    else:
        description = extract_meta_from_soup(soup, ("og:description", "twitter:description", "description"))
        text = clean_article_text(description or soup.get_text("\n", strip=True))

    if not is_probably_article_text(text):
        description = clean_article_text(extract_meta_from_soup(soup, ("og:description", "twitter:description", "description")))
        if len(description) > len(text):
            text = description

    return Article(title=clean_title(title), url=normalize_article_url(url), text=text)


def extract_title_from_soup(soup) -> str:
    title = extract_meta_from_soup(soup, ("og:title", "twitter:title"))
    if title:
        return clean_title(title)
    heading = soup.select_one("h1, .media_end_head_headline, .article_title, #title_area")
    if heading:
        return clean_title(heading.get_text(" ", strip=True))
    if soup.title:
        return clean_title(soup.title.get_text(" ", strip=True))
    return ""


def extract_meta_from_soup(soup, names: tuple[str, ...]) -> str:
    for name in names:
        node = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if node and node.get("content"):
            return normalize_space(node["content"])
    return ""


def collect_articles(portal_url: str, limit: int = 12) -> list[Article]:
    urls = discover_article_urls(portal_url, limit=limit)
    articles: list[Article] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(urls)))) as executor:
        future_to_url = {executor.submit(fetch_article, url): url for url in urls}
        for future in as_completed(future_to_url):
            try:
                article = future.result()
            except Exception:
                continue
            if is_probably_article_text(article.text):
                articles.append(article)
    return articles


def tokenize(text: str) -> list[str]:
    rough_tokens = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9+.#-]*", text)
    tokens = []
    for token in rough_tokens:
        value = normalize_keyword_token(token)
        if is_keyword_token(value):
            tokens.append(value)
            tokens.extend(expand_compound_token(value))
    return tokens


def normalize_keyword_token(token: str) -> str:
    value = token.strip(" \t\r\n\"'“”‘’()[]{}<>.,!?;:·…")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9+.#-]*", value):
        return value.upper()
    for suffix in sorted(KOREAN_PARTICLE_SUFFIXES + KOREAN_ENDING_SUFFIXES, key=len, reverse=True):
        if len(value) > len(suffix) + 1 and value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def is_keyword_token(token: str) -> bool:
    if len(token) < MIN_KEYWORD_LENGTH:
        return False
    if token in KOREAN_STOPWORDS:
        return False
    if token.isdigit():
        return False
    return bool(re.search(r"[가-힣A-Za-z]", token))


def expand_compound_token(token: str) -> list[str]:
    if not re.fullmatch(r"[가-힣]{4,}", token):
        return []
    if len(token) > MAX_COMPOUND_TOKEN_LENGTH:
        return []

    return [
        part
        for part in COMPOUND_KEYWORD_PARTS
        if part != token and part in token and is_keyword_token(part)
    ]


def compute_tfidf_vectors(articles: list[Article]) -> tuple[list[dict[str, float]], list[list[str]]]:
    docs = [tokenize(article.title + " " + article.text) for article in articles]
    doc_count = len(docs)
    df = Counter()
    for doc in docs:
        df.update(set(doc))

    vectors: list[dict[str, float]] = []
    for doc in docs:
        tf = Counter(doc)
        length = max(1, len(doc))
        vector = {}
        for term, count in tf.items():
            idf = math.log((doc_count + 1) / (df[term] + 1)) + 1
            vector[term] = (count / length) * idf
        vectors.append(vector)
    return vectors, docs


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    common = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def vector_cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def main_keyword_coverage(
    main_vector: dict[str, float],
    candidate_vector: dict[str, float],
    top_n: int = MAIN_KEYWORD_OVERLAP_TOP_N,
) -> float:
    main_terms = sorted(main_vector.items(), key=lambda item: item[1], reverse=True)[:top_n]
    total_weight = sum(score for _, score in main_terms)
    if total_weight == 0:
        return 0.0
    covered_weight = sum(score for term, score in main_terms if term in candidate_vector)
    return covered_weight / total_weight


def blended_similarity_score(
    content_score: float,
    title_score: float,
    keyword_coverage: float,
    token_jaccard: float,
    word2vec_document: float,
) -> float:
    return (
        SIMILARITY_WEIGHTS["content_tfidf"] * content_score
        + SIMILARITY_WEIGHTS["title_tfidf"] * title_score
        + SIMILARITY_WEIGHTS["main_keyword_coverage"] * keyword_coverage
        + SIMILARITY_WEIGHTS["token_jaccard"] * token_jaccard
        + SIMILARITY_WEIGHTS["word2vec_document"] * word2vec_document
    )


def similarity_components(
    main_vector: dict[str, float],
    candidate_vector: dict[str, float],
    main_title_vector: dict[str, float],
    candidate_title_vector: dict[str, float],
    main_tokens: list[str],
    candidate_tokens: list[str],
    main_word2vec_vector: list[float] | None = None,
    candidate_word2vec_vector: list[float] | None = None,
) -> dict[str, float]:
    content_score = cosine_similarity(main_vector, candidate_vector)
    title_score = cosine_similarity(main_title_vector, candidate_title_vector)
    keyword_coverage = main_keyword_coverage(main_vector, candidate_vector)
    token_jaccard = jaccard_similarity(main_tokens, candidate_tokens)
    word2vec_document = (
        vector_cosine(main_word2vec_vector, candidate_word2vec_vector)
        if main_word2vec_vector and candidate_word2vec_vector
        else 0.0
    )
    final_score = blended_similarity_score(
        content_score,
        title_score,
        keyword_coverage,
        token_jaccard,
        word2vec_document,
    )
    return {
        "final_score": final_score,
        "content_tfidf": content_score,
        "title_tfidf": title_score,
        "main_keyword_coverage": keyword_coverage,
        "token_jaccard": token_jaccard,
        "word2vec_document": word2vec_document,
    }


def find_main_article(
    articles: list[Article],
) -> tuple[int, list[list[float]], list[dict[str, float]], list[list[str]], list[dict[str, float]], list[list[float]]]:
    vectors, docs = compute_tfidf_vectors(articles)
    title_vectors, _ = compute_tfidf_vectors(
        [Article(title=article.title, url=article.url, text="") for article in articles]
    )
    word2vec_vectors = document_word2vec_vectors(docs)
    matrix = [[0.0 for _ in articles] for _ in articles]
    for i, left in enumerate(vectors):
        for j, right in enumerate(vectors):
            if i == j:
                matrix[i][j] = 1.0
            else:
                matrix[i][j] = similarity_components(
                    left,
                    right,
                    title_vectors[i],
                    title_vectors[j],
                    docs[i],
                    docs[j],
                    word2vec_vectors[i],
                    word2vec_vectors[j],
                )["final_score"]

    centrality = []
    for i, row in enumerate(matrix):
        others = [score for j, score in enumerate(row) if i != j]
        centrality.append(sum(others) / max(1, len(others)))
    main_index = max(range(len(articles)), key=lambda index: centrality[index])
    return main_index, matrix, vectors, docs, title_vectors, word2vec_vectors


def rank_by_similarity(articles: list[Article]) -> tuple[Article, list[RankedArticle], dict[str, object]]:
    if not articles:
        raise ValueError("분석할 기사가 없습니다.")
    if len(articles) == 1:
        return articles[0], [], {"method": "single_article"}

    main_index, matrix, vectors, docs, title_vectors, word2vec_vectors = find_main_article(articles)
    ranked = []
    score_components = {}
    for index, article in enumerate(articles):
        if index == main_index:
            continue
        components = similarity_components(
            vectors[main_index],
            vectors[index],
            title_vectors[main_index],
            title_vectors[index],
            docs[main_index],
            docs[index],
            word2vec_vectors[main_index],
            word2vec_vectors[index],
        )
        score_components[article.title] = components
        ranked.append(RankedArticle(article=article, similarity=matrix[main_index][index]))
    ranked.sort(key=lambda item: item.similarity, reverse=True)

    jaccard_scores = {
        articles[index].title: jaccard_similarity(docs[main_index], docs[index])
        for index in range(len(articles))
        if index != main_index
    }
    metadata = {
        "method": "blended TF-IDF, title, keyword coverage, Jaccard, and Word2Vec document similarity",
        "similarity_weights": SIMILARITY_WEIGHTS,
        "main_index": main_index,
        "tfidf_matrix": matrix,
        "score_components": score_components,
        "jaccard_scores": jaccard_scores,
        "docs": docs,
        "vectors": vectors,
    }
    return articles[main_index], ranked, metadata


def rank_against_main_article(
    main_article: Article,
    candidate_articles: list[Article],
) -> tuple[Article, list[RankedArticle], dict[str, object]]:
    main_url = canonical_url(main_article.url)
    candidates = [
        article
        for article in candidate_articles
        if canonical_url(article.url) != main_url
    ]
    articles = [main_article, *candidates]
    vectors, docs = compute_tfidf_vectors(articles)
    title_vectors, _ = compute_tfidf_vectors(
        [Article(title=article.title, url=article.url, text="") for article in articles]
    )
    word2vec_vectors = document_word2vec_vectors(docs)
    main_vector = vectors[0]
    ranked = []
    score_components = {}
    for index, article in enumerate(articles[1:], start=1):
        components = similarity_components(
            main_vector,
            vectors[index],
            title_vectors[0],
            title_vectors[index],
            docs[0],
            docs[index],
            word2vec_vectors[0],
            word2vec_vectors[index],
        )
        score_components[article.title] = components
        ranked.append(RankedArticle(article=article, similarity=components["final_score"]))
    ranked.sort(key=lambda item: item.similarity, reverse=True)

    jaccard_scores = {
        article.title: jaccard_similarity(docs[0], docs[index])
        for index, article in enumerate(articles[1:], start=1)
    }
    metadata = {
        "method": "fixed main article blended TF-IDF, title, keyword coverage, Jaccard, and Word2Vec document similarity",
        "similarity_weights": SIMILARITY_WEIGHTS,
        "main_index": 0,
        "score_components": score_components,
        "jaccard_scores": jaccard_scores,
        "docs": docs,
        "vectors": vectors,
    }
    return main_article, ranked, metadata


def extract_keywords(articles: list[Article], top_n: int = 10) -> list[tuple[str, float]]:
    vectors, _ = compute_tfidf_vectors(articles)
    scores: defaultdict[str, float] = defaultdict(float)
    for vector in vectors:
        for term, score in vector.items():
            scores[term] += score
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def extract_repeated_keywords(
    articles: list[Article],
    top_n: int = 10,
    min_doc_count: int = 2,
) -> tuple[list[tuple[str, float]], dict[str, int]]:
    vectors, docs = compute_tfidf_vectors(articles)
    scores: defaultdict[str, float] = defaultdict(float)
    document_counts: Counter[str] = Counter()

    for vector, doc in zip(vectors, docs):
        document_counts.update(set(doc))
        for term, score in vector.items():
            scores[term] += score

    repeated = [
        (term, score)
        for term, score in scores.items()
        if document_counts[term] >= min_doc_count
    ]
    repeated.sort(
        key=lambda item: (document_counts[item[0]], item[1]),
        reverse=True,
    )
    keywords = repeated[:top_n]
    return keywords, {term: document_counts[term] for term, _ in keywords}


def extract_keyword_candidates(
    articles: list[Article],
    min_doc_count: int = 2,
) -> tuple[dict[str, float], dict[str, int], list[list[str]]]:
    vectors, docs = compute_tfidf_vectors(articles)
    scores: defaultdict[str, float] = defaultdict(float)
    document_counts: Counter[str] = Counter()

    for vector, doc in zip(vectors, docs):
        document_counts.update(set(doc))
        for term, score in vector.items():
            scores[term] += score

    candidates = {
        term: score
        for term, score in scores.items()
        if document_counts[term] >= min_doc_count
    }
    if not candidates and min_doc_count > 1:
        candidates = dict(scores)
    return candidates, dict(document_counts), docs


def train_word2vec(tokens_by_doc: list[list[str]]):
    try:
        from gensim.models import Word2Vec

        return Word2Vec(
            sentences=tokens_by_doc,
            vector_size=80,
            window=4,
            min_count=1,
            workers=1,
            sg=1,
            epochs=200,
            seed=42,
        )
    except Exception:
        return train_simple_skipgram(tokens_by_doc)


def train_simple_skipgram(
    tokens_by_doc: list[list[str]],
    vector_size: int = 30,
    window: int = 3,
    epochs: int = 45,
    learning_rate: float = 0.035,
    negative_samples: int = 2,
) -> SimpleWord2Vec | None:
    vocabulary = sorted({token for doc in tokens_by_doc for token in doc})
    if len(vocabulary) < 2:
        return None

    rng = random.Random(42)
    input_vectors = {
        word: [(rng.random() - 0.5) / vector_size for _ in range(vector_size)]
        for word in vocabulary
    }
    output_vectors = {
        word: [(rng.random() - 0.5) / vector_size for _ in range(vector_size)]
        for word in vocabulary
    }
    pairs = []
    for doc in tokens_by_doc:
        for index, center in enumerate(doc):
            start = max(0, index - window)
            end = min(len(doc), index + window + 1)
            for context_index in range(start, end):
                if context_index != index:
                    pairs.append((center, doc[context_index]))
    if not pairs:
        return None

    def update(center: str, context: str, label: int) -> None:
        center_vector = input_vectors[center]
        context_vector = output_vectors[context]
        dot = sum(a * b for a, b in zip(center_vector, context_vector))
        prediction = 1 / (1 + math.exp(-max(-8, min(8, dot))))
        gradient = learning_rate * (label - prediction)
        old_center = center_vector[:]
        for i in range(vector_size):
            center_vector[i] += gradient * context_vector[i]
            context_vector[i] += gradient * old_center[i]

    for _ in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            update(center, context, 1)
            for _ in range(negative_samples):
                negative = rng.choice(vocabulary)
                if negative != context:
                    update(center, negative, 0)

    return SimpleWord2Vec(input_vectors)


def cooccurrence_keyword_neighbors(
    tokens_by_doc: list[list[str]],
    keyword: str,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    related_counts: Counter[str] = Counter()
    keyword_docs = 0
    for tokens in tokens_by_doc:
        if keyword not in tokens:
            continue
        keyword_docs += 1
        related_counts.update(token for token in set(tokens) if token != keyword)
    if keyword_docs == 0:
        related_counts.update(token for tokens in tokens_by_doc for token in set(tokens) if token != keyword)
        keyword_docs = max(1, len(tokens_by_doc))
    return [
        (word, count / keyword_docs)
        for word, count in related_counts.most_common(top_n)
        if is_keyword_token(word)
    ]


def word2vec_keyword_neighbors(tokens_by_doc: list[list[str]], keywords: list[str], top_n: int = 5) -> dict[str, list[tuple[str, float]]]:
    model = train_word2vec(tokens_by_doc)
    results: dict[str, list[tuple[str, float]]] = {}
    for keyword in keywords:
        pairs: list[tuple[str, float]] = []
        if model is not None:
            if hasattr(model, "wv") and keyword in model.wv:
                pairs = [(word, float(score)) for word, score in model.wv.most_similar(keyword, topn=top_n)]
            elif keyword in model:
                pairs = [(word, float(score)) for word, score in model.most_similar(keyword, topn=top_n)]
        if len(pairs) < top_n:
            seen = {word for word, _ in pairs}
            for word, score in cooccurrence_keyword_neighbors(tokens_by_doc, keyword, top_n=top_n):
                if word not in seen:
                    pairs.append((word, score))
                    seen.add(word)
                if len(pairs) >= top_n:
                    break
        if pairs:
            results[keyword] = pairs[:top_n]
    return results


def contains_word(model, word: str) -> bool:
    if hasattr(model, "wv"):
        return word in model.wv
    return word in model


def most_similar_words(model, word: str, top_n: int = 5) -> list[tuple[str, float]]:
    if hasattr(model, "wv"):
        return [(other, float(score)) for other, score in model.wv.most_similar(word, topn=top_n)]
    return [(other, float(score)) for other, score in model.most_similar(word, topn=top_n)]


def word_vector(model, word: str) -> list[float] | None:
    if model is None or not contains_word(model, word):
        return None
    if hasattr(model, "wv"):
        return [float(value) for value in model.wv[word]]
    return [float(value) for value in model.vectors[word]]


def document_word2vec_vectors(tokens_by_doc: list[list[str]]) -> list[list[float]]:
    model = train_word2vec(tokens_by_doc)
    vectors: list[list[float]] = []
    for tokens in tokens_by_doc:
        token_vectors = [
            vector
            for token in tokens
            if (vector := word_vector(model, token)) is not None
        ]
        if not token_vectors:
            vectors.append([])
            continue
        size = len(token_vectors[0])
        vectors.append(
            [
                sum(vector[index] for vector in token_vectors) / len(token_vectors)
                for index in range(size)
            ]
        )
    return vectors


def select_word2vec_keywords(
    articles: list[Article],
    top_n: int = 10,
    min_doc_count: int = 2,
) -> tuple[list[tuple[str, float]], dict[str, int], dict[str, dict[str, object]]]:
    candidates, document_counts, docs = extract_keyword_candidates(
        articles,
        min_doc_count=min_doc_count,
    )
    if not candidates:
        return [], {}, {}

    model = train_word2vec(docs)
    max_tfidf = max(candidates.values()) or 1.0
    max_doc_count = max(document_counts.get(term, 1) for term in candidates) or 1
    validation: dict[str, dict[str, object]] = {}
    ranked: list[tuple[str, float]] = []

    for term, tfidf_score in candidates.items():
        similar_terms: list[tuple[str, float]] = []
        word2vec_score = 0.0
        if model is not None and contains_word(model, term):
            similar_terms = most_similar_words(model, term, top_n=5)
            positive_scores = [max(0.0, score) for _, score in similar_terms]
            if positive_scores:
                word2vec_score = sum(positive_scores) / len(positive_scores)

        tfidf_norm = tfidf_score / max_tfidf
        doc_norm = document_counts.get(term, 1) / max_doc_count
        final_score = (word2vec_score * 0.60) + (tfidf_norm * 0.25) + (doc_norm * 0.15)
        validation[term] = {
            "word2vec_score": word2vec_score,
            "tfidf_score": tfidf_score,
            "similar_terms": similar_terms,
            "final_score": final_score,
        }
        ranked.append((term, final_score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    keywords = ranked[:top_n]
    keyword_doc_counts = {term: document_counts.get(term, 0) for term, _ in keywords}
    validation = {term: validation[term] for term, _ in keywords}
    return keywords, keyword_doc_counts, validation


def select_ranked_keywords(
    articles: list[Article],
    top_n: int = 10,
    min_doc_count: int = 2,
) -> tuple[list[tuple[str, float]], dict[str, int], dict[str, dict[str, float]]]:
    candidates, document_counts, _ = extract_keyword_candidates(
        articles,
        min_doc_count=min_doc_count,
    )
    if not candidates:
        return [], {}, {}

    max_tfidf = max(candidates.values()) or 1.0
    max_doc_count = max(document_counts.get(term, 1) for term in candidates) or 1
    keyword_scores: dict[str, dict[str, float]] = {}
    ranked: list[tuple[str, float]] = []

    for term, tfidf_score in candidates.items():
        tfidf_norm = tfidf_score / max_tfidf
        doc_norm = document_counts.get(term, 1) / max_doc_count
        final_score = (tfidf_norm * 0.75) + (doc_norm * 0.25)
        keyword_scores[term] = {
            "tfidf_score": tfidf_score,
            "document_count": float(document_counts.get(term, 0)),
            "final_score": final_score,
        }
        ranked.append((term, final_score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    keywords = ranked[:top_n]
    keyword_doc_counts = {term: document_counts.get(term, 0) for term, _ in keywords}
    keyword_scores = {term: keyword_scores[term] for term, _ in keywords}
    return keywords, keyword_doc_counts, keyword_scores


def load_sample_articles(path: str | Path = "sample_articles.json") -> list[Article]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Article(title=item["title"], url=item["url"], text=item["text"]) for item in data]


def analyze_articles(articles: list[Article], keyword_count: int = 10) -> dict[str, object]:
    main, ranked, metadata = rank_by_similarity(articles)
    keyword_articles = [item.article for item in ranked]
    if len(keyword_articles) < 2:
        keyword_articles = [main, *keyword_articles]
    keywords, keyword_doc_counts, keyword_scores = select_ranked_keywords(
        keyword_articles,
        top_n=keyword_count,
        min_doc_count=2 if len(keyword_articles) > 1 else 1,
    )
    _, keyword_docs = compute_tfidf_vectors(keyword_articles)
    word2vec_neighbors = word2vec_keyword_neighbors(
        keyword_docs,
        [word for word, _ in keywords],
        top_n=5,
    )
    metadata["keyword_source"] = "tfidf_doc_frequency_similarity_ranked_articles"
    metadata["keyword_doc_counts"] = keyword_doc_counts
    metadata["keyword_scores"] = keyword_scores
    metadata["word2vec_neighbors"] = word2vec_neighbors
    return {
        "main": main,
        "ranked": ranked,
        "keywords": keywords,
        "keyword_doc_counts": keyword_doc_counts,
        "keyword_scores": keyword_scores,
        "word2vec_neighbors": word2vec_neighbors,
        "metadata": metadata,
    }


def analyze_against_main(
    main_article: Article,
    candidate_articles: list[Article],
    keyword_count: int = 10,
) -> dict[str, object]:
    main, ranked, metadata = rank_against_main_article(main_article, candidate_articles)
    keyword_articles = [item.article for item in ranked]
    if len(keyword_articles) < 2:
        keyword_articles = [main, *keyword_articles]
    keywords, keyword_doc_counts, keyword_scores = select_ranked_keywords(
        keyword_articles,
        top_n=keyword_count,
        min_doc_count=2 if len(keyword_articles) > 1 else 1,
    )
    _, keyword_docs = compute_tfidf_vectors(keyword_articles)
    word2vec_neighbors = word2vec_keyword_neighbors(
        keyword_docs,
        [word for word, _ in keywords],
        top_n=5,
    )
    metadata["keyword_source"] = "tfidf_doc_frequency_similarity_ranked_articles"
    metadata["keyword_doc_counts"] = keyword_doc_counts
    metadata["keyword_scores"] = keyword_scores
    metadata["word2vec_neighbors"] = word2vec_neighbors
    return {
        "main": main,
        "ranked": ranked,
        "keywords": keywords,
        "keyword_doc_counts": keyword_doc_counts,
        "keyword_scores": keyword_scores,
        "word2vec_neighbors": word2vec_neighbors,
        "metadata": metadata,
    }


def print_analysis(result: dict[str, object]) -> None:
    main = result["main"]
    ranked = result["ranked"]
    keywords = result["keywords"]
    keyword_doc_counts = result.get("keyword_doc_counts", {})
    keyword_scores = result.get("keyword_scores", {})
    word2vec_neighbors = result.get("word2vec_neighbors", {})
    assert isinstance(main, Article)

    print(f"[메인 기사] {main.title}")
    print(f"URL: {main.url}")
    print()
    print("[유사 기사 순위]")
    for index, item in enumerate(ranked, start=1):
        assert isinstance(item, RankedArticle)
        print(f"{index}. {item.article.title} - score={item.similarity:.4f}")
    print()
    print("[핵심 키워드]")
    for word, score in keywords:
        count = keyword_doc_counts.get(word, "")
        score_data = keyword_scores.get(word, {})
        tfidf_score = score_data.get("tfidf_score", 0.0)
        print(f"- {word}: score={score:.4f}, tfidf={tfidf_score:.4f}, repeated_docs={count}")
    print()
    print("[Word2Vec 유사 단어]")
    for word, pairs in word2vec_neighbors.items():
        formatted = ", ".join(f"{other}({similarity:.3f})" for other, similarity in pairs[:5])
        print(f"- {word}: {formatted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="뉴스 기사 유사도 검색 및 핵심 키워드 분석")
    parser.add_argument("--url", help="뉴스 포털/섹션 URL")
    parser.add_argument("--main-url", help="기준으로 삼을 메인 기사 URL")
    parser.add_argument("--sample", action="store_true", help="샘플 기사로 실행")
    parser.add_argument("--limit", type=int, default=12, help="수집할 기사 수")
    args = parser.parse_args()

    if args.sample or not args.url:
        articles = load_sample_articles()
    elif args.main_url:
        main_article = fetch_article(args.main_url)
        articles = collect_articles(args.url, limit=args.limit)
        result = analyze_against_main(main_article, articles)
        print_analysis(result)
        return
    else:
        articles = collect_articles(args.url, limit=args.limit)
    result = analyze_articles(articles)
    print_analysis(result)


if __name__ == "__main__":
    main()
