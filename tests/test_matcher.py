"""Unit tests for the V1 dialogue matcher (synthetic transcript data)."""

import json
import tempfile
import unittest
from pathlib import Path

from matcher.core import (
    DialogueMatcher,
    MatchNotFoundError,
    MatcherError,
    TranscriptError,
    generate_windows,
    load_transcript,
    normalize_text,
    text_similarity,
)


def make_words(tokens, start=0.0, step=0.5, prob=0.9):
    """Build a Word list from token strings with synthetic timing."""
    from matcher.core import Word

    return [
        Word(word=t, start=start + i * step, end=start + i * step + 0.4,
             probability=prob)
        for i, t in enumerate(tokens)
    ]


SAMPLE_TOKENS = (
    "my mind rebels at stagnation give me problems give me work".split()
)

def write_transcript(words):
    data = {
        "metadata": {},
        "words": [
            {"word": w.word, "start": w.start, "end": w.end,
             "probability": w.probability}
            for w in words
        ],
    }
    return json.dumps(data)


class TestNormalization(unittest.TestCase):
    def test_case_and_punctuation(self):
        self.assertEqual(
            normalize_text("Hello, World!"),
            "hello world",
        )

    def test_apostrophe_kept_inside_word(self):
        self.assertEqual(normalize_text("Don't stop"), "don't stop")

    def test_trailing_apostrophe_removed(self):
        self.assertEqual(normalize_text("dogs' bones"), "dogs bones")

    def test_curly_quotes_normalized(self):
        self.assertEqual(normalize_text("\u2018Hi\u2019"), "hi")

    def test_whitespace_collapsed(self):
        self.assertEqual(normalize_text("  a   b\tc\n"), "a b c")


class TestWindowGeneration(unittest.TestCase):
    def test_sizes_n_minus_2_to_n_plus_2(self):
        words = make_words([f"w{i}" for i in range(20)])
        windows = generate_windows(words, target_len=5)
        sizes = {len(w) for w in windows}
        self.assertEqual(sizes, {3, 4, 5, 6, 7})

    def test_short_target_clamped_to_one(self):
        words = make_words(["a", "b"])
        windows = generate_windows(words, target_len=1)
        self.assertTrue(all(len(w) >= 1 for w in windows))
        self.assertIn(1, {len(w) for w in windows})

    def test_window_count(self):
        # 3 + 4 + 5 + 6 + 7 windows over 10 words
        words = make_words([f"w{i}" for i in range(10)])
        windows = generate_windows(words, target_len=5)
        expected = sum(10 - s + 1 for s in (3, 4, 5, 6, 7))
        self.assertEqual(len(windows), expected)

    def test_windows_preserve_original_word_objects(self):
        words = make_words(["a", "b", "c", "d"])
        windows = generate_windows(words, target_len=2)
        self.assertIs(windows[0][0], words[0])


class TestSimilarity(unittest.TestCase):
    def test_exact_match_is_one(self):
        self.assertEqual(text_similarity("hello world", "hello world"), 1.0)

    def test_completely_different_is_low(self):
        self.assertLess(text_similarity("hello world", "zzz qqq"), 0.2)

    def test_score_bounded(self):
        score = text_similarity("abc def", "xyz")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.matcher = DialogueMatcher(min_similarity=0.5)
        self.words = make_words(SAMPLE_TOKENS)

    def _match_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split()
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            p.write_text(write_transcript(self.words), encoding="utf-8")
            return self.matcher.find_best_match(" ".join(tokens), str(p))

    def test_exact_match(self):
        result = self._match_tokens("my mind rebels at stagnation")
        self.assertEqual(result.text_similarity, 1.0)
        self.assertEqual(result.matched_text, "my mind rebels at stagnation")

    def test_case_punctuation_difference(self):
        result = self._match_tokens("My mind rebels at stagnation.")
        self.assertAlmostEqual(result.text_similarity, 1.0)

    def test_slight_transcription_difference(self):
        result = self._match_tokens("my mind rebelled at stagnation")
        self.assertGreater(result.text_similarity, 0.7)
        self.assertEqual(result.matched_text, "my mind rebels at stagnation")

    def test_timestamps_from_first_and_last_word(self):
        result = self._match_tokens("my mind rebels at stagnation")
        self.assertAlmostEqual(result.start_time, 0.0)
        # last word index 4: start=4*0.5=2.0, end=2.4
        self.assertAlmostEqual(result.end_time, 2.4)

    def test_confidence_calculated_from_raw_probabilities(self):
        result = self._match_tokens("my mind rebels at stagnation")
        self.assertAlmostEqual(result.average_word_probability, 0.9)
        self.assertAlmostEqual(result.minimum_word_probability, 0.9)

    def test_candidates_retained(self):
        result = self._match_tokens("give me problems")
        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertLessEqual(len(result.candidates), self.matcher.top_n)
        best = max(result.candidates, key=lambda c: c.text_similarity)
        self.assertEqual(best.matched_text, result.matched_text)

    def test_no_reasonable_match_raises(self):
        with self.assertRaises(MatchNotFoundError):
            self._match_tokens(["zebra"] * 5)

    def test_empty_target_raises(self):
        with self.assertRaises(MatcherError):
            self.matcher.find_best_match_in_words("", self.words)
        with self.assertRaises(MatcherError):
            self.matcher.find_best_match_in_words("   ", self.words)


class TestTranscriptValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, content, name="t.json"):
        p = self.dir / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_missing_file(self):
        with self.assertRaises(TranscriptError):
            load_transcript(self.dir / "nope.json")

    def test_invalid_json(self):
        with self.assertRaises(TranscriptError):
            load_transcript(self._write("{not json"))

    def test_missing_words_field(self):
        with self.assertRaises(TranscriptError):
            load_transcript(self._write('{"metadata": {}}'))

    def test_empty_words_list(self):
        with self.assertRaises(TranscriptError):
            load_transcript(self._write('{"words": []}'))

    def test_malformed_word_entry(self):
        bad = '{"words": [{"word": "a", "start": 0.0}]}'
        with self.assertRaises(TranscriptError):
            load_transcript(self._write(bad))

    def test_end_before_start_rejected(self):
        bad = ('{"words": [{"word": "a", "start": 1.0, "end": 0.5,'
               ' "probability": 0.9}]}')
        with self.assertRaises(TranscriptError):
            load_transcript(self._write(bad))


if __name__ == "__main__":
    unittest.main()
