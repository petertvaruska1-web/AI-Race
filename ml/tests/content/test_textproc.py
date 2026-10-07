import re

import numpy as np
import pytest

from airace_content.textproc import (
    MOJIBAKE_CHARS,
    SPAM_MARKERS,
    build_vocab,
    normalize_text,
    quality_score,
    simplicity,
    words,
)
from airace_content.topics import TOPIC_KEYWORDS, tag_topic
from airace_ml.data.corpus import TOPICS
from airace_ml.minipy.interpreter import call_function, run_program
from airace_ml.skills.code import code_train_docs
from airace_ml.skills.facts import fact_prose_docs
from airace_ml.skills.kb import load_kb

CLEAN = "The little dog ran to the park. It played with a red ball. Then it went home to sleep."
VOCAB = build_vocab(
    [CLEAN * 3, "the a an and to of in it is was he she they ball park home sleep"], top_n=5000
)


def test_normalize():
    assert normalize_text("a\r\nb\x07  c\n\n\n\nd  ") == "a\nb c\n\nd"


def test_quality_ordering():
    q_clean = quality_score(CLEAN, VOCAB)
    q_typo = quality_score("Teh littel dgo rna to teh prak. It plyaed wiht a rde blal.", VOCAB)
    q_spam = quality_score("CLICK HERE!!! buy now $$$ http://x.y limited offer act now", VOCAB)
    q_garb = quality_score("ThÃ© littlÃ© dÃ¶g rÃ¤n tÃ¶ thÃ© pÃ¤rk Â¤Â¦Â§", VOCAB)
    assert q_clean > 0.8 and q_clean > q_typo and q_spam < 0.3 and q_garb < 0.3
    assert quality_score("", VOCAB) == 0


def test_simplicity():
    assert simplicity(CLEAN, VOCAB) > simplicity(
        "Notwithstanding epistemological considerations, the hermeneutic paradigm persists.", VOCAB
    )


def test_topics():
    assert TOPICS[tag_topic("The cat and the dog chased a bird near the cow.")] == "animals"
    assert TOPICS[tag_topic("zzz qqq")] == "other"


# ---------------------------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------------------------


def test_normalize_line_endings_and_control_characters():
    assert normalize_text("a\r\nb\rc\nd") == "a\nb\nc\nd"
    assert normalize_text("a\x00b\x1bc\x7fd\x85e\x0bf\x0cg") == "abcdefg"
    assert normalize_text("tab\there\nnext") == "tab\there\nnext"  # tab and newline survive
    assert normalize_text("") == ""
    assert normalize_text("\x07\x08") == ""


def test_normalize_spaces_and_blank_lines():
    assert normalize_text("a    b") == "a b"
    assert normalize_text("a  \nb \n\nc") == "a\nb\n\nc"  # trailing spaces go from every line
    assert normalize_text("a\n\nb") == "a\n\nb"  # two newlines are kept
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"  # three or more become two
    assert normalize_text("a\n  \n\n\nb") == "a\n\nb"  # a line of only spaces is a blank line
    assert normalize_text("a\n  ") == "a\n"
    assert normalize_text("a \t b") == "a \t b"  # only runs of spaces collapse


def test_normalize_keeps_non_ascii_text_apart_from_nfc():
    text = (
        "café 今日はいい天気 \U0001f600 "
        "\U0001f468\u200d\U0001f469\u200d\U0001f467 naïve “quoted” a\u00a0b ½"
    )
    assert normalize_text(text) == text
    assert normalize_text("cafe\u0301") == "caf\u00e9"  # NFC composes
    assert normalize_text("e\x07\u0301") == "\u00e9"  # a removed control exposes a combining mark


def test_normalize_is_idempotent():
    rng = np.random.default_rng(0)
    alphabet = list("ab  \t\n\r\x07\x7fé\u0301\u200d\U0001f600")
    for _ in range(500):
        s = "".join(rng.choice(alphabet, int(rng.integers(0, 40))))
        once = normalize_text(s)
        assert normalize_text(once) == once
        assert "\n\n\n" not in once and "\r" not in once
        # Only a line's leading indentation may hold a run of spaces.
        assert all("  " not in line.lstrip(" \t") for line in once.split("\n"))


# ---------------------------------------------------------------------------------------------
# words, vocab, simplicity
# ---------------------------------------------------------------------------------------------


def test_words_are_lowercase_alphabetic():
    assert words("Don't STOP, it's 3 o'clock! Café") == ["don't", "stop", "it's", "o'clock", "caf"]
    assert words("") == [] and words("123 !!! 今日") == []


def test_build_vocab_takes_the_most_frequent_words_deterministically():
    texts = ["b b b a a c", "d"]
    assert build_vocab(texts, 2) == {"b", "a"}
    assert build_vocab(["c a b"], 2) == {"a", "b"}  # ties break alphabetically, not by arrival
    assert build_vocab(iter(texts), 10) == {"a", "b", "c", "d"}
    assert build_vocab(texts, 0) == set() and build_vocab([], 5) == set()
    with pytest.raises(ValueError):
        build_vocab(texts, -1)


def test_simplicity_values():
    vocab = {"the", "cat", "sat", "on", "mat"}
    assert simplicity("The cat sat on the mat.", vocab) == pytest.approx(1.0)
    assert simplicity("The cat sat on the dog.", vocab) == pytest.approx(5 / 6)
    one_long = " ".join(["cat"] * 36) + "."
    assert simplicity(one_long, {"cat"}) == pytest.approx(0.5)  # 18 / 36 words per sentence
    two_short = " ".join(["cat"] * 18) + ". " + " ".join(["cat"] * 18) + "."
    assert simplicity(two_short, {"cat"}) == pytest.approx(1.0)
    assert simplicity("", vocab) == 0.0
    assert simplicity("!!! 123 ...", vocab) == 0.0


# ---------------------------------------------------------------------------------------------
# quality_score: every factor of the formula, then realistic text
# ---------------------------------------------------------------------------------------------


def test_quality_repeated_lines():
    vocab = {"the", "cat", "sat"}
    # Two lines, one repeats: r = 0.5; 18 of 20 characters are letters, so a = 1.
    assert quality_score("the cat sat.\nthe cat sat.", vocab) == pytest.approx(0.75)
    assert quality_score("the cat sat.\nthe sat cat", vocab) == pytest.approx(1.0)


def test_quality_alpha_ratio():
    vocab = {"the", "cat", "sat"}
    # 9 letters of 13 non-space characters.
    a = (9 / 13 - 0.6) / 0.3
    assert quality_score("the cat sat 1234", vocab) == pytest.approx(0.5 + 0.5 * a)
    assert quality_score("the cat sat 12345678901234567890", vocab) == pytest.approx(0.5)


def test_quality_known_words():
    vocab = {"aaa", "bbb", "ccc", "ddd", "eee", "fff"}
    text = "aaa bbb ccc ddd eee fff ggg hhh"  # 6 of 8 known
    assert quality_score(text, vocab) == pytest.approx((0.75 - 0.5) / 0.4)
    assert quality_score(text + " x y z", vocab) == pytest.approx((0.75 - 0.5) / 0.4)  # 1-letter
    assert quality_score("ggg hhh", vocab) == 0.0  # nothing known
    assert quality_score("1 2 3 !!!", vocab) == 0.0  # no words at all


def test_quality_mojibake_share_of_characters():
    vocab = {"cat"}
    one_in_100 = "cat " * 24 + "cat\u00c3"
    assert len(one_in_100) == 100
    assert quality_score(one_in_100, vocab) == pytest.approx(0.8)  # g = 1 - 20 * 0.01
    two = "cat " * 20 + "cat\u00c3 " * 2 + "cat"
    assert quality_score(two, vocab) == pytest.approx(1 - 20 * 2 / len(two))
    assert quality_score("cat" + "\u00c3" * 5 + " cat" * 20, vocab) == 0.0  # 5% or more: g = 0
    assert "\u00c3" in MOJIBAKE_CHARS and "\ufffd" in MOJIBAKE_CHARS and "a" not in MOJIBAKE_CHARS


def test_quality_spam_markers_count_every_occurrence_case_insensitively():
    vocab = {"cat", "subscribe"}
    base = "cat " * 12
    assert quality_score(base + "subscribe", vocab) == pytest.approx(2 / 3)
    assert quality_score(base + "subscribe SUBSCRIBE", vocab) == pytest.approx(1 / 3)
    assert quality_score(base + "subscribe subscribe Subscribe", vocab) == 0.0
    assert quality_score(base + "subscribe " * 7, vocab) == 0.0  # more than three hits
    assert "click here" in SPAM_MARKERS and "$$$" in SPAM_MARKERS


def test_quality_is_a_float_in_the_unit_interval_for_any_text():
    rng = np.random.default_rng(1)
    alphabet = list("the cat sat\n!$Ãé今\U0001f600 1234\t")
    for _ in range(300):
        text = "".join(rng.choice(alphabet, int(rng.integers(0, 80))))
        q = quality_score(text, VOCAB)
        assert isinstance(q, float) and 0.0 <= q <= 1.0


def test_quality_of_empty_or_symbol_only_text_is_zero():
    for text in ("", " ", "\n\n", "\U0001f600\U0001f600", "!!!", "今日はいい天気"):
        assert quality_score(text, VOCAB) == 0.0


def _fixture_paragraphs() -> list[str]:
    from pathlib import Path

    folder = Path(__file__).parents[1] / "fixtures" / "text"
    paragraphs: list[str] = []
    for name in ("stories.txt", "facts.txt"):
        raw = (folder / name).read_text(encoding="utf-8")
        paragraphs += [p.strip() for p in re.split(r"\n\s*\n", raw) if len(p.split()) >= 25]
    return paragraphs


def test_quality_of_real_prose_is_high_when_its_words_are_known():
    paragraphs = _fixture_paragraphs()
    assert len(paragraphs) >= 20
    vocab = build_vocab(paragraphs, top_n=5000)
    scores = [quality_score(p, vocab) for p in paragraphs]
    assert min(scores) > 0.85


def test_quality_unknown_vocabulary_lowers_the_score():
    paragraphs = _fixture_paragraphs()
    half = len(paragraphs) // 2
    vocab = build_vocab(paragraphs[:half], top_n=200)  # a narrow vocabulary
    narrow = np.mean([quality_score(p, vocab) for p in paragraphs[half:]])
    full = np.mean([quality_score(p, build_vocab(paragraphs, 5000)) for p in paragraphs[half:]])
    assert narrow < full


# ---------------------------------------------------------------------------------------------
# topics
# ---------------------------------------------------------------------------------------------


def test_every_topic_but_other_has_at_least_15_distinct_lowercase_keywords():
    assert set(TOPIC_KEYWORDS) == set(TOPICS) - {"other"}
    for topic, keywords in TOPIC_KEYWORDS.items():
        assert isinstance(keywords, frozenset) and len(keywords) >= 15, topic
        assert all(re.fullmatch(r"[a-z']+", k) for k in keywords), topic


def test_a_keyword_belongs_to_one_topic_only():
    seen: dict[str, str] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            assert keyword not in seen, f"{keyword!r} is in {seen[keyword]!r} and {topic!r}"
            seen[keyword] = topic


def test_keywords_identify_their_topic():
    for topic, keywords in TOPIC_KEYWORDS.items():
        sample = " ".join(sorted(keywords)[:5])
        assert TOPICS[tag_topic(sample)] == topic


def test_topic_is_the_argmax_with_ties_going_to_the_earlier_topic():
    assert TOPICS[tag_topic("apple cat dog bird")] == "animals"  # 3 animals against 1 food
    assert TOPICS[tag_topic("apple banana cat")] == "food"
    assert TOPICS[tag_topic("cat apple")] == "animals"  # tie: animals comes before food
    assert TOPICS[tag_topic("apple cat")] == "animals"


def test_topic_matches_simple_plurals_and_verb_forms():
    assert TOPICS[tag_topic("Dogs and foxes and horses")] == "animals"
    assert TOPICS[tag_topic("berries and cherries")] == "food"
    assert TOPICS[tag_topic("She smiled, then cried and laughed.")] == "feelings"


def test_topic_of_text_without_keywords_or_letters_is_other():
    other = TOPICS.index("other")
    for text in ("", "   ", "zzz qqq", "12345 !!!", "今日はいい天気", "\U0001f600"):
        assert tag_topic(text) == other


def test_topic_agrees_with_the_topic_of_knowledge_base_paragraphs():
    rng = np.random.default_rng(0)
    docs = [d for d in fact_prose_docs(load_kb(), rng, 400) if d.topic != "other"]
    agree = np.mean([TOPICS[tag_topic(d.text)] == d.topic for d in docs])
    assert agree > 0.85


# ---------------------------------------------------------------------------------------------
# fix round 1, R-a: leading indentation survives normalization (code documents)
# ---------------------------------------------------------------------------------------------

NESTED_PROGRAM = (
    "def collatz(n):\n"
    "    steps = 0\n"
    "    while n != 1:\n"
    "        if n % 2 == 0:\n"
    "            n = n // 2\n"
    "        else:\n"
    "            n = 3 * n + 1\n"
    "        steps += 1\n"
    "    return steps\n"
    "\n"
    "for start in [6, 7]:\n"
    "    print(start, collatz(start))\n"
)
MBPP_STYLE = (
    "# Write a function to find the largest sum of two different items.\n"
    "def largest_pair(items):\n"
    "    best = 0\n"
    "    for a in items:\n"
    "        for b in items:\n"
    "            if a != b and a + b > best:\n"
    "                best = a + b\n"
    "    return best"
)


def test_normalize_keeps_each_lines_leading_indentation():
    assert normalize_text("def f():\n    return 1") == "def f():\n    return 1"
    assert normalize_text("if x:\n        y = 1\n    z = 2") == "if x:\n        y = 1\n    z = 2"
    assert normalize_text("  lead  gap") == "  lead gap"  # only the later run collapses
    assert normalize_text("a\n    b    c  \n\t\tx   y") == "a\n    b c\n\t\tx y"
    assert normalize_text(" \t  x") == " \t  x"  # mixed indentation is kept as written
    assert normalize_text("a\n    \nb") == "a\n\nb"  # a line of only spaces is still blank
    assert normalize_text("\r\n    x\r\n") == "\n    x\n"


def test_normalized_minipy_program_is_unchanged_and_still_runs():
    assert normalize_text(NESTED_PROGRAM) == NESTED_PROGRAM
    before, after = run_program(NESTED_PROGRAM), run_program(normalize_text(NESTED_PROGRAM))
    assert before.error is None and before.stdout == "6 8\n7 16\n"
    assert after == before


def test_normalized_mbpp_style_function_is_unchanged_and_still_runs():
    crlf = MBPP_STYLE.replace("\n", "\r\n")  # as it arrives from the dataset
    assert normalize_text(crlf) == MBPP_STYLE
    assert call_function(normalize_text(crlf), "largest_pair", [[1, 5, 3]]) == (8, None)
    assert call_function(MBPP_STYLE, "largest_pair", [[1, 5, 3]]) == (8, None)


def _code_of(text: str) -> str:
    if text.startswith("Program:\n"):
        return text[len("Program:\n") : text.rindex("\nOutput:")]
    return text


def test_generated_code_documents_keep_their_indentation_and_behaviour():
    docs = [d.text for d in code_train_docs(np.random.default_rng(0), 300)]
    assert len(docs) == 300

    def indentation(text: str) -> list[int]:
        return [len(line) - len(line.lstrip(" ")) for line in text.split("\n")]

    assert any("\n    " in text for text in docs)
    for text in docs:
        normalized = normalize_text(text)
        assert indentation(normalized) == indentation(text)
        before, after = run_program(_code_of(text)), run_program(_code_of(normalized))
        assert before.error is None and after.stdout == before.stdout and after.error is None


# ---------------------------------------------------------------------------------------------
# fix round 1, M1/M2: topic keywords through endings, possessives and quotes
# ---------------------------------------------------------------------------------------------


def test_endings_do_not_turn_common_words_into_keywords():
    other = TOPICS.index("other")
    for text in (
        "being",
        "Nobody was being rude.",
        "cared caring cares",
        "rates rated rating",
        "bearing",
        "He was bearing the cost and she was caring for the rated plans.",
    ):
        assert tag_topic(text) == other, text
    assert TOPICS[tag_topic("She cared about her friends being happy.")] in {"family", "feelings"}


def test_endings_still_reach_real_keywords():
    assert TOPICS[tag_topic("foxes and potatoes")] in {"animals", "food"}
    assert TOPICS[tag_topic("Two foxes ran past three more foxes.")] == "animals"
    assert TOPICS[tag_topic("boxes of potatoes, tomatoes and lunches")] == "food"
    assert TOPICS[tag_topic("bears")] == "animals" and TOPICS[tag_topic("cars")] == "technology"
    assert TOPICS[tag_topic("rats and bees")] == "animals"
    assert TOPICS[tag_topic("She was cooking and baking and they cooked.")] == "food"
    assert TOPICS[tag_topic("He smiled, was crying and laughing.")] == "feelings"
    assert TOPICS[tag_topic("The puppies and bunnies")] == "animals"


def test_possessives_and_quoted_words_reach_their_keyword():
    assert TOPICS[tag_topic("the cat's dog's bird's")] == "animals"
    assert TOPICS[tag_topic("the cat\u2019s dog\u2019s bird\u2019s")] == "animals"
    assert TOPICS[tag_topic("the dogs' bowls")] == "animals"
    assert TOPICS[tag_topic("She said 'cat' twice")] == "animals"
    assert TOPICS[tag_topic("She said \u2018cat\u2019 twice")] == "animals"
    assert TOPICS[tag_topic("Mom's and Dad's hugs")] == "family"


# ---------------------------------------------------------------------------------------------
# fix round 1, M3: curly apostrophes and quoted words in words() and quality_score
# ---------------------------------------------------------------------------------------------


def test_words_treat_curly_apostrophes_as_straight_and_trim_quote_marks():
    assert words("isn\u2019t it \u2018hello\u2019 o\u2019clock") == [
        "isn't",
        "it",
        "hello",
        "o'clock",
    ]
    assert words("'hello' and ''quoted'' and dogs'") == ["hello", "and", "quoted", "and", "dogs"]
    assert words("rock 'n' roll") == ["rock", "n", "roll"]
    assert words("' '' ''' \u2019") == []
    assert words("Don't") == ["don't"]


def test_curly_apostrophe_prose_scores_like_straight_apostrophe_prose():
    straight = "I don't think it isn't a good day. They can't wait, and we aren't late."
    curly = straight.replace("'", "\u2019")
    vocab = build_vocab([straight], 100)
    assert build_vocab([curly], 100) == vocab
    assert quality_score(curly, vocab) == quality_score(straight, vocab)
    assert quality_score(curly, vocab) > 0.95


def test_single_quoted_speech_does_not_lower_the_score():
    plain = "She said hello and he said goodbye to the little dog near the old gate."
    vocab = build_vocab([plain], 100)
    base = quality_score(plain, vocab)
    for quoted in (
        "She said 'hello' and he said 'goodbye' to the little dog near the old gate.",
        "She said \u2018hello\u2019 and he said \u2018goodbye\u2019 to the little dog near the old gate.",
    ):
        assert quality_score(quoted, vocab) > base - 0.02  # only the quote marks' letter share
