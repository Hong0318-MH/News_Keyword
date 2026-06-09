import unittest

from news_similarity_engine import analyze_against_main, analyze_articles, load_sample_articles, tokenize


class EngineTest(unittest.TestCase):
    def test_tokenize_keeps_korean_terms(self):
        tokens = tokenize("인공지능 반도체 개발과 생성형 인공지능 서비스")
        self.assertIn("인공지능", tokens)
        self.assertIn("반도체", tokens)

    def test_sample_analysis_returns_main_and_keywords(self):
        articles = load_sample_articles("sample_articles.json")
        result = analyze_articles(articles, keyword_count=5)
        self.assertTrue(result["main"].title)
        self.assertEqual(len(result["ranked"]), 3)
        self.assertEqual(len(result["keywords"]), 5)

    def test_fixed_main_article_ranks_candidates(self):
        articles = load_sample_articles("sample_articles.json")
        result = analyze_against_main(articles[0], articles[1:], keyword_count=5)
        self.assertEqual(result["main"].url, "sample://ai-chip-1")
        self.assertEqual(len(result["ranked"]), 3)
        self.assertGreaterEqual(result["ranked"][0].similarity, result["ranked"][-1].similarity)


if __name__ == "__main__":
    unittest.main()
