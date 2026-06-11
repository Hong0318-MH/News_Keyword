import unittest

from news_similarity_engine import (
    analyze_against_main,
    analyze_articles,
    extract_article_from_html,
    load_sample_articles,
    looks_like_article_url,
    normalize_article_url,
    tokenize,
    word2vec_keyword_neighbors,
)


class EngineTest(unittest.TestCase):
    def test_tokenize_keeps_korean_terms(self):
        tokens = tokenize("인공지능 반도체 개발과 생성형 인공지능 서비스")
        self.assertIn("인공지능", tokens)
        self.assertIn("반도체", tokens)

    def test_tokenize_removes_verbs_and_report_words(self):
        tokens = tokenize("정부가 인공지능 반도체 정책에 대해 밝히고 이후 계획을 설명했다")
        self.assertIn("인공지능", tokens)
        self.assertIn("반도체", tokens)
        self.assertNotIn("대해", tokens)
        self.assertNotIn("밝히", tokens)
        self.assertNotIn("이후", tokens)
        self.assertNotIn("계획", tokens)
        self.assertNotIn("정책에", tokens)
        self.assertNotIn("계획을", tokens)
        self.assertFalse(any(token.startswith("밝히") for token in tokens))

    def test_sample_analysis_returns_main_and_keywords(self):
        articles = load_sample_articles("sample_articles.json")
        result = analyze_articles(articles, keyword_count=5)
        self.assertTrue(result["main"].title)
        self.assertEqual(len(result["ranked"]), 3)
        self.assertGreater(len(result["keywords"]), 0)
        self.assertLessEqual(len(result["keywords"]), 5)
        self.assertIn("keyword_scores", result)
        self.assertIn("word2vec_neighbors", result)
        self.assertTrue(
            all(result["keyword_doc_counts"][word] >= 2 for word, _ in result["keywords"])
        )
        self.assertTrue(all(word in result["keyword_scores"] for word, _ in result["keywords"]))
        self.assertIsInstance(result["word2vec_neighbors"], dict)

    def test_tokenize_expands_compound_korean_terms(self):
        tokens = tokenize("인공지능반도체")
        self.assertIn("인공지능반도체", tokens)
        self.assertIn("인공지능", tokens)
        self.assertIn("반도체", tokens)

    def test_word2vec_neighbors_falls_back_to_cooccurrence(self):
        neighbors = word2vec_keyword_neighbors(
            [["인공지능", "반도체"], ["인공지능", "데이터센터"]],
            ["인공지능"],
            top_n=2,
        )
        self.assertIn("인공지능", neighbors)
        self.assertGreaterEqual(len(neighbors["인공지능"]), 1)

    def test_fixed_main_article_ranks_candidates(self):
        articles = load_sample_articles("sample_articles.json")
        result = analyze_against_main(articles[0], articles[1:], keyword_count=5)
        self.assertEqual(result["main"].url, "sample://ai-chip-1")
        self.assertEqual(len(result["ranked"]), 3)
        self.assertGreaterEqual(result["ranked"][0].similarity, result["ranked"][-1].similarity)
        components = result["metadata"]["score_components"][result["ranked"][0].article.title]
        self.assertIn("content_tfidf", components)
        self.assertIn("title_tfidf", components)
        self.assertIn("main_keyword_coverage", components)
        self.assertIn("token_jaccard", components)
        self.assertIn("word2vec_document", components)
        self.assertAlmostEqual(result["ranked"][0].similarity, components["final_score"])
        self.assertIn("keyword_scores", result)
        self.assertIn("word2vec_neighbors", result)
        self.assertTrue(
            all(result["keyword_doc_counts"][word] >= 2 for word, _ in result["keywords"])
        )
        self.assertTrue(all(word in result["keyword_scores"] for word, _ in result["keywords"]))
        self.assertIsInstance(result["word2vec_neighbors"], dict)

    def test_naver_article_url_filter(self):
        article_url = "https://n.news.naver.com/mnews/article/001/0012345678?sid=100"
        section_url = "https://news.naver.com/section/100"
        self.assertTrue(looks_like_article_url(article_url, "충분히 긴 기사 제목입니다"))
        self.assertFalse(looks_like_article_url(section_url, "정치 섹션 뉴스입니다"))
        self.assertEqual(
            normalize_article_url("https://news.naver.com/main/read.naver?oid=001&aid=0012345678"),
            "https://n.news.naver.com/mnews/article/001/0012345678",
        )

    @unittest.skipUnless(__import__("importlib").util.find_spec("bs4"), "beautifulsoup4 is not installed")
    def test_extract_article_body_ignores_naver_noise(self):
        html = """
        <html>
          <head><meta property="og:title" content="테스트 기사 : 네이버 뉴스"></head>
          <body>
            <div id="dic_area">
              <p>인공지능 반도체 시장이 빠르게 성장하면서 국내 기업들이 연구개발 투자를 확대하고 있다.</p>
              <p>정부도 인재 양성과 데이터센터 인프라 지원 정책을 함께 추진한다고 밝혔다.</p>
            </div>
            <div>뉴스 이용 설정을 할 수 있어요 많이 본 뉴스 네이버 AI 뉴스 알고리즘</div>
          </body>
        </html>
        """
        article = extract_article_from_html("https://n.news.naver.com/mnews/article/001/0012345678", html)
        self.assertIn("인공지능 반도체 시장", article.text)
        self.assertNotIn("뉴스 이용 설정", article.text)


if __name__ == "__main__":
    unittest.main()
