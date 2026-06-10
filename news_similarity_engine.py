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
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen


KOREAN_STOPWORDS = {
    "그리고", "그러나", "하지만", "또한", "관련", "대한", "위해", "통해", "이번", "지난", "있는",
    "한다", "했다", "된다", "됐다", "에서", "으로", "에게", "까지", "부터", "보다", "처럼", "기자",
    "뉴스", "사진", "제공", "이라며", "다고", "라고", "것으로", "것이다", "있다", "없는", "등을",
}


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
        if absolute in seen or not looks_like_article_url(absolute, anchor_text):
            continue
        seen.add(absolute)
        urls.append(absolute)
        if len(urls) >= limit:
            break
    return urls


def looks_like_article_url(url: str, anchor_text: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if len(anchor_text) < 8:
        return False
    patterns = ("news", "article", "view", "read", "aid", "sid", "idx", "no=")
    return any(pattern in url.lower() for pattern in patterns)


def fetch_article(url: str) -> Article:
    parser = parse_html(fetch_html(url))
    title = parser.title or url
    text = parser.text
    return Article(title=title, url=url, text=text)


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
            if len(article.text) >= 120:
                articles.append(article)
    return articles


def tokenize(text: str) -> list[str]:
    try:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()
        tokens = []
        for token in kiwi.tokenize(text):
            if token.tag.startswith(("N", "V", "SL")):
                value = token.form.strip()
                if len(value) > 1 and value not in KOREAN_STOPWORDS:
                    tokens.append(value)
        return tokens
    except Exception:
        rough_tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
        return [token for token in rough_tokens if token not in KOREAN_STOPWORDS]


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


def find_main_article(articles: list[Article]) -> tuple[int, list[list[float]], list[dict[str, float]], list[list[str]]]:
    vectors, docs = compute_tfidf_vectors(articles)
    matrix = [[0.0 for _ in articles] for _ in articles]
    for i, left in enumerate(vectors):
        for j, right in enumerate(vectors):
            matrix[i][j] = 1.0 if i == j else cosine_similarity(left, right)

    centrality = []
    for i, row in enumerate(matrix):
        others = [score for j, score in enumerate(row) if i != j]
        centrality.append(sum(others) / max(1, len(others)))
    main_index = max(range(len(articles)), key=lambda index: centrality[index])
    return main_index, matrix, vectors, docs


def rank_by_similarity(articles: list[Article]) -> tuple[Article, list[RankedArticle], dict[str, object]]:
    if not articles:
        raise ValueError("분석할 기사가 없습니다.")
    if len(articles) == 1:
        return articles[0], [], {"method": "single_article"}

    main_index, matrix, vectors, docs = find_main_article(articles)
    ranked = []
    for index, article in enumerate(articles):
        if index == main_index:
            continue
        ranked.append(RankedArticle(article=article, similarity=matrix[main_index][index]))
    ranked.sort(key=lambda item: item.similarity, reverse=True)

    jaccard_scores = {
        articles[index].title: jaccard_similarity(docs[main_index], docs[index])
        for index in range(len(articles))
        if index != main_index
    }
    metadata = {
        "method": "TF-IDF cosine similarity",
        "main_index": main_index,
        "tfidf_matrix": matrix,
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
    main_vector = vectors[0]
    ranked = [
        RankedArticle(article=article, similarity=cosine_similarity(main_vector, vectors[index]))
        for index, article in enumerate(articles[1:], start=1)
    ]
    ranked.sort(key=lambda item: item.similarity, reverse=True)

    jaccard_scores = {
        article.title: jaccard_similarity(docs[0], docs[index])
        for index, article in enumerate(articles[1:], start=1)
    }
    metadata = {
        "method": "fixed main article TF-IDF cosine similarity",
        "main_index": 0,
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


def word2vec_keyword_neighbors(tokens_by_doc: list[list[str]], keywords: list[str], top_n: int = 5) -> dict[str, list[tuple[str, float]]]:
    model = train_word2vec(tokens_by_doc)
    if model is None:
        return {}
    results: dict[str, list[tuple[str, float]]] = {}
    for keyword in keywords:
        if hasattr(model, "wv") and keyword in model.wv:
            results[keyword] = [(word, float(score)) for word, score in model.wv.most_similar(keyword, topn=top_n)]
        elif keyword in model:
            results[keyword] = [(word, float(score)) for word, score in model.most_similar(keyword, topn=top_n)]
    return results


def load_sample_articles(path: str | Path = "sample_articles.json") -> list[Article]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Article(title=item["title"], url=item["url"], text=item["text"]) for item in data]


def analyze_articles(articles: list[Article], keyword_count: int = 10) -> dict[str, object]:
    main, ranked, metadata = rank_by_similarity(articles)
    keywords = extract_keywords(articles, top_n=keyword_count)
    tokens_by_doc = metadata.get("docs")
    if not isinstance(tokens_by_doc, list):
        _, tokens_by_doc = compute_tfidf_vectors(articles)
    neighbors = word2vec_keyword_neighbors(tokens_by_doc, [word for word, _ in keywords[:5]])
    return {
        "main": main,
        "ranked": ranked,
        "keywords": keywords,
        "word2vec_neighbors": neighbors,
        "metadata": metadata,
    }


def analyze_against_main(
    main_article: Article,
    candidate_articles: list[Article],
    keyword_count: int = 10,
) -> dict[str, object]:
    main, ranked, metadata = rank_against_main_article(main_article, candidate_articles)
    articles = [main, *[item.article for item in ranked]]
    keywords = extract_keywords(articles, top_n=keyword_count)
    tokens_by_doc = metadata.get("docs")
    if not isinstance(tokens_by_doc, list):
        _, tokens_by_doc = compute_tfidf_vectors(articles)
    neighbors = word2vec_keyword_neighbors(tokens_by_doc, [word for word, _ in keywords[:5]])
    return {
        "main": main,
        "ranked": ranked,
        "keywords": keywords,
        "word2vec_neighbors": neighbors,
        "metadata": metadata,
    }


def print_analysis(result: dict[str, object]) -> None:
    main = result["main"]
    ranked = result["ranked"]
    keywords = result["keywords"]
    neighbors = result["word2vec_neighbors"]
    assert isinstance(main, Article)

    print(f"[메인 기사] {main.title}")
    print(f"URL: {main.url}")
    print()
    print("[유사 기사 순위]")
    for index, item in enumerate(ranked, start=1):
        assert isinstance(item, RankedArticle)
        print(f"{index}. {item.article.title} - cosine={item.similarity:.4f}")
    print()
    print("[핵심 키워드]")
    for word, score in keywords:
        print(f"- {word}: {score:.4f}")
    print()
    print("[Word2Vec 유사 단어]")
    if not neighbors:
        print("학습 가능한 토큰이 부족합니다.")
    for word, pairs in neighbors.items():
        formatted = ", ".join(f"{other}({score:.3f})" for other, score in pairs)
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
