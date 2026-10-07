import itertools
import re
from collections import Counter

import pytest

from airace_ml.skills import patterns, reasoning
from airace_ml.skills.patterns import pattern_bench_items, pattern_train_docs
from airace_ml.skills.reasoning import reasoning_bench_items, reasoning_train_docs
from airace_ml.skills.types import ExactItem, MCItem, TextDoc, reserved_for_bench, skill_rng


def test_sizes_and_determinism():
    assert len(reasoning_bench_items(skill_rng("reasoning", "bench"))) == 200
    assert len(pattern_bench_items(skill_rng("pattern", "bench"))) == 150
    a = [i.prompt for i in pattern_bench_items(skill_rng("pattern", "bench"))]
    assert a == [i.prompt for i in pattern_bench_items(skill_rng("pattern", "bench"))]


def test_partition_disjoint():
    for it in reasoning_bench_items(skill_rng("reasoning", "bench")) + pattern_bench_items(
        skill_rng("pattern", "bench")
    ):
        assert reserved_for_bench(it.prompt)
    train = "\n".join(
        d.text
        for d in reasoning_train_docs(skill_rng("reasoning", "train"), 2000)
        + pattern_train_docs(skill_rng("pattern", "train"), 2000)
    )
    for it in pattern_bench_items(skill_rng("pattern", "bench")):
        assert it.prompt not in train


def test_answers_correct():
    for it in pattern_bench_items(skill_rng("pattern", "bench")):
        if "fam:step" in it.tags:
            nums = [int(x) for x in it.prompt.removeprefix("Next:").strip(" ,").split(",")]
            assert int(it.answers[0]) == nums[-1] + (nums[1] - nums[0])
    for it in reasoning_bench_items(skill_rng("reasoning", "bench")):
        assert len(set(it.options)) == len(it.options) and 0 <= it.answer_index < len(it.options)
        assert {t.split(":")[0] for t in it.tags} >= {"fam"}


# ---------------------------------------------------------------------------------------------
# Independent solvers: they know the rules of each puzzle, not how the generators build them.
# ---------------------------------------------------------------------------------------------

COMPARATIVES = {
    "taller": ("height", 1),
    "shorter": ("height", -1),
    "older": ("age", 1),
    "younger": ("age", -1),
    "faster": ("speed", 1),
    "slower": ("speed", -1),
    "heavier": ("weight", 1),
    "lighter": ("weight", -1),
}
SUPERLATIVES = {
    "tallest": ("height", 1),
    "shortest": ("height", -1),
    "oldest": ("age", 1),
    "youngest": ("age", -1),
    "fastest": ("speed", 1),
    "slowest": ("speed", -1),
    "heaviest": ("weight", 1),
    "lightest": ("weight", -1),
}


def solve_compare(text: str) -> str:
    facts = re.findall(r"(\w+) is (\w+) than (\w+)\.", text)
    assert len(facts) == 2
    beats: dict[str, set[str]] = {}
    names: set[str] = set()
    for a, adj, b in facts:
        winner, loser = (a, b) if COMPARATIVES[adj][1] > 0 else (b, a)
        beats.setdefault(winner, set()).add(loser)
        names |= {a, b}
    for _ in range(3):
        for x in list(beats):
            for y in list(beats[x]):
                beats[x] |= beats.get(y, set())
    reach = {n: len(beats.get(n, ())) for n in names}
    assert sorted(reach.values()) == [0, 1, 2], "the premises must give one clear order"
    word = re.search(r"Who is the (\w+)\?$", text).group(1)
    assert SUPERLATIVES[word][0] == {COMPARATIVES[adj][0] for _, adj, _ in facts}.pop()
    ranked = sorted(names, key=reach.__getitem__)
    return ranked[-1] if SUPERLATIVES[word][1] > 0 else ranked[0]


def solve_syllogism(text: str) -> str:
    alls = re.findall(r"All (\w+)s are (\w+)s\.", text)
    nos = re.findall(r"No (\w+)s are (\w+)s\.", text)
    members = dict(re.findall(r"(\w+) is a (\w+)\.", text))
    subject, target = re.search(r"Is (?:a )?(\w+) a (\w+)\?$", text).groups()
    known = {members.get(subject, subject)}
    for _ in range(3):
        known |= {b for a, b in alls if a in known}
    excluded = {b for a, b in nos if a in known}
    assert (target in known) != (target in excluded), "the premises must settle the question"
    return "yes" if target in known else "no"


def solve_word_problem(text: str) -> int:
    start, change = (int(x) for x in re.findall(r"\d+", text))
    return start + change if " more" in text else start - change


def solve_count(text: str) -> int:
    target, words = re.fullmatch(r"How many (\w+)s are in this list: (.*)\?", text).groups()
    return words.split(", ").count(target)


def family_of_text(text: str) -> str:
    if "in this list:" in text:
        return "count"
    if "Who is the" in text:
        return "compare"
    if "How many" in text:
        return "word_problem"
    return "syllogism"


def solve(family: str, text: str) -> str:
    return str(
        {
            "compare": solve_compare,
            "syllogism": solve_syllogism,
            "word_problem": solve_word_problem,
            "count": solve_count,
        }[family](text)
    )


def question_text(prompt: str) -> str:
    assert prompt.startswith("Question: ") and prompt.endswith("\nAnswer:")
    return prompt.removeprefix("Question: ").removesuffix("\nAnswer:")


def family_tag(item) -> str:
    (tag,) = [t for t in item.tags if t.startswith("fam:")]
    return tag.removeprefix("fam:")


@pytest.fixture(scope="module")
def reasoning_bench():
    return reasoning_bench_items(skill_rng("reasoning", "bench"))


@pytest.fixture(scope="module")
def pattern_bench():
    return pattern_bench_items(skill_rng("pattern", "bench"))


@pytest.fixture(scope="module")
def reasoning_train():
    return reasoning_train_docs(skill_rng("reasoning", "train"), 2000)


@pytest.fixture(scope="module")
def pattern_train():
    return pattern_train_docs(skill_rng("pattern", "train"), 2000)


def reasoning_blocks(docs: list[TextDoc]) -> list[list[str]]:
    return [d.text.split("\n\n") for d in docs]


# ---------------------------------------------------------------------------------------------
# Reasoning benchmark
# ---------------------------------------------------------------------------------------------


def test_reasoning_bench_shape(reasoning_bench):
    assert [i.id for i in reasoning_bench] == [f"reasoning-{k:04d}" for k in range(200)]
    for it in reasoning_bench:
        assert isinstance(it, MCItem) and it.category == "reasoning"
        assert not it.chat and it.group is None
        assert it.prompt.startswith("Question: ") and it.prompt.endswith("\nAnswer:")
        assert it.prompt.isascii()
    assert len({i.prompt for i in reasoning_bench}) == 200
    families = Counter(family_tag(i) for i in reasoning_bench)
    assert families == {f: 50 for f in reasoning.FAMILIES}
    assert set(reasoning.FAMILIES) == {"compare", "syllogism", "word_problem", "count"}


def test_reasoning_bench_answers_are_right(reasoning_bench):
    for it in reasoning_bench:
        family = family_tag(it)
        text = question_text(it.prompt)
        assert family_of_text(text) == family
        assert it.options[it.answer_index] == solve(family, text), it.prompt
        # exactly one option is right: the others are never the solved answer
        assert [o for o in it.options if o == solve(family, text)] == [it.options[it.answer_index]]
        assert len(set(it.options)) == len(it.options)


def test_compare_options_are_the_three_names_and_a_stranger(reasoning_bench):
    items = [i for i in reasoning_bench if family_tag(i) == "compare"]
    for it in items:
        text = question_text(it.prompt)
        in_premise = set(re.findall(r"[A-Z][a-z]+", text.split(" Who")[0])) - {"Who"}
        assert len(in_premise) == 3 and len(it.options) == 4
        (stranger,) = set(it.options) - in_premise
        assert stranger in reasoning.NAMES and stranger not in it.prompt
        assert set(it.options) - {stranger} == in_premise
    attributes = {
        COMPARATIVES[w][0] for it in items for w in re.findall(r"is (\w+) than", it.prompt)
    }
    assert attributes == {"height", "age", "speed", "weight"}
    asked = {re.search(r"the (\w+)\?", it.prompt).group(1) for it in items}
    assert asked == set(SUPERLATIVES)


def test_syllogisms_use_made_up_words_and_are_balanced(reasoning_bench):
    items = [i for i in reasoning_bench if family_tag(i) == "syllogism"]
    assert all(i.options == ["yes", "no"] for i in items)
    assert Counter(i.options[i.answer_index] for i in items) == {"yes": 25, "no": 25}
    for it in items:
        tokens = set(re.findall(r"[a-z]+", question_text(it.prompt)))
        used = {w for w in reasoning.NONCE_WORDS if w in tokens or w + "s" in tokens}
        assert len(used) >= 2
    # both kinds of puzzle (one step, two steps) occur with both answers
    two_step = Counter(
        (
            question_text(i.prompt).count("All ") + question_text(i.prompt).count("No ") == 2,
            i.options[i.answer_index],
        )
        for i in items
    )
    assert len(two_step) == 4 and min(two_step.values()) >= 5


def test_word_problems_stay_in_range_and_options_are_nearby(reasoning_bench):
    items = [i for i in reasoning_bench if family_tag(i) == "word_problem"]
    kinds = Counter(" more" in question_text(i.prompt) for i in items)
    assert kinds[True] >= 15 and kinds[False] >= 15  # additions and subtractions
    for it in items:
        answer = int(it.options[it.answer_index])
        start, change = (int(x) for x in re.findall(r"\d+", it.prompt))
        assert 0 <= answer <= 20 and start <= 20 and change >= 1
        assert all(o.isdigit() for o in it.options) and len(it.options) == 4
        assert all(
            0 < abs(int(o) - answer) <= 3 for i, o in enumerate(it.options) if i != it.answer_index
        )


def test_count_items(reasoning_bench):
    items = [i for i in reasoning_bench if family_tag(i) == "count"]
    answers = Counter(int(i.options[i.answer_index]) for i in items)
    assert len(answers) >= 4 and 0 in answers
    for it in items:
        assert 4 <= len(question_text(it.prompt).split("list: ")[1].split(", ")) <= 8
        assert all(o.isdigit() for o in it.options) and len(it.options) == 4


def test_answer_positions_are_spread(reasoning_bench):
    positions = Counter(i.answer_index for i in reasoning_bench if i.options != ["yes", "no"])
    assert set(positions) == {0, 1, 2, 3} and min(positions.values()) >= 20


# ---------------------------------------------------------------------------------------------
# Reasoning training text
# ---------------------------------------------------------------------------------------------


def test_reasoning_train_blocks_are_correct_and_well_formed(reasoning_train):
    assert len(reasoning_train) == 2000
    families, explained, total = Counter(), 0, 0
    for doc, blocks in zip(reasoning_train, reasoning_blocks(reasoning_train)):
        assert doc.kind == "plain" and doc.turns is None and doc.topic == "school"
        assert 3 <= len(blocks) <= 8
        for block in blocks:
            lines = block.split("\n")
            assert 2 <= len(lines) <= 3, block
            assert lines[0].startswith("Question: ") and lines[1].startswith("Answer: ")
            text = lines[0].removeprefix("Question: ")
            family = family_of_text(text)
            families[family] += 1
            assert lines[1].removeprefix("Answer: ") == solve(family, text), block
            total += 1
            if len(lines) == 3:  # a one-sentence explanation follows the answer
                explained += 1
                assert lines[2].endswith(".") and "Question" not in lines[2] and lines[2]
    assert set(families) == set(reasoning.FAMILIES)
    assert min(families.values()) > 0.2 * total
    assert 0.2 < explained / total < 0.6


def test_reasoning_train_never_contains_a_benchmark_question(reasoning_train, reasoning_bench):
    extra = reasoning_train_docs(skill_rng("reasoning", "train", "v2"), 1000)
    bench_prompts = {i.prompt for i in reasoning_bench}
    for docs in (reasoning_train, extra):
        for blocks in reasoning_blocks(docs):
            for block in blocks:
                prompt = block.split("\n")[0] + "\nAnswer:"
                assert not reserved_for_bench(prompt)
                assert prompt not in bench_prompts
    text = "\n".join(d.text for d in reasoning_train)
    assert not any(p in text for p in bench_prompts)


def test_reasoning_train_is_deterministic_and_stream_dependent():
    a = [d.text for d in reasoning_train_docs(skill_rng("reasoning", "train"), 30)]
    assert a == [d.text for d in reasoning_train_docs(skill_rng("reasoning", "train"), 30)]
    assert a != [d.text for d in reasoning_train_docs(skill_rng("reasoning", "train", "v2"), 30)]
    assert reasoning_train_docs(skill_rng("reasoning", "train"), 0) == []


def test_reasoning_bench_is_deterministic_and_smaller_sizes_work():
    a = reasoning_bench_items(skill_rng("reasoning", "bench"), 12)
    assert a == reasoning_bench_items(skill_rng("reasoning", "bench"), 12)
    assert [family_tag(i) for i in a] == [f for _ in range(3) for f in reasoning.FAMILIES]
    assert reasoning_bench_items(skill_rng("reasoning", "bench"), 0) == []
    assert a != reasoning_bench_items(skill_rng("reasoning", "bench", "v2"), 12)


def test_reasoning_gives_up_when_nothing_can_be_accepted(monkeypatch):
    calls = []
    monkeypatch.setattr(reasoning, "reserved_for_bench", lambda key: calls.append(key) or False)
    with pytest.raises(RuntimeError, match="reasoning"):
        reasoning_bench_items(skill_rng("reasoning", "bench"), 3)
    assert len(calls) == 50 * 3  # the attempt cap is 50 per requested item
    monkeypatch.setattr(reasoning, "reserved_for_bench", lambda key: True)
    with pytest.raises(RuntimeError, match="reasoning"):
        reasoning_train_docs(skill_rng("reasoning", "train"), 2)


def test_word_pools_are_unambiguous():
    assert len(set(reasoning.NAMES)) == len(reasoning.NAMES) >= 20 and all(
        n.isascii() and n.istitle() for n in reasoning.NAMES
    )
    assert not [(a, b) for a in reasoning.NAMES for b in reasoning.NAMES if a != b and a in b]
    assert len(set(reasoning.NONCE_WORDS)) == len(reasoning.NONCE_WORDS) >= 20
    for word in reasoning.NONCE_WORDS:
        assert word.isascii() and word.islower() and word[0] not in "aeiou"
        assert word[-1] not in "sxz" and not word.endswith(("ch", "sh"))  # "+s" is the plural
        assert word.title() not in reasoning.NAMES


# ---------------------------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------------------------

ALL_CYCLE_WORDS = {w for group in patterns.CYCLE_WORDS.values() for w in group}


def parse_terms(text: str) -> list[str]:
    return text.removeprefix("Next:").strip(" ,").split(", ")


def families_that_continue(shown: list[str]) -> dict[str, str]:
    """Every pattern family whose rule explains the shown terms, with the term it predicts."""
    found: dict[str, str] = {}
    if all(t.isdigit() for t in shown):
        nums = [int(t) for t in shown]
        diffs = {b - a for a, b in itertools.pairwise(nums)}
        if len(diffs) == 1:
            (d,) = diffs
            if 1 <= d <= 9 and 1 <= nums[0] <= 20:
                found["step"] = str(nums[-1] + d)
            if -3 <= d <= -1 and nums[-1] + d >= 0:
                found["countdown"] = str(nums[-1] + d)
        if nums[0] in range(1, 6) and all(b == 2 * a for a, b in itertools.pairwise(nums)):
            found["double"] = str(nums[-1] * 2)
    elif all(len(t) == 1 and "A" <= t <= "Z" for t in shown):
        codes = [ord(t) for t in shown]
        diffs = {b - a for a, b in itertools.pairwise(codes)}
        if len(diffs) == 1:
            (d,) = diffs
            if 1 <= d <= 3 and codes[-1] + d <= ord("Z"):
                found["letters"] = chr(codes[-1] + d)
    elif set(shown) <= ALL_CYCLE_WORDS:
        size = len(set(shown))
        if size in (2, 3) and all(shown[i] == shown[i % size] for i in range(len(shown))):
            found["cycle"] = shown[len(shown) % size]
    return found


def test_pattern_bench_shape(pattern_bench):
    assert [i.id for i in pattern_bench] == [f"pattern-{k:04d}" for k in range(150)]
    for it in pattern_bench:
        assert isinstance(it, ExactItem) and it.category == "pattern"
        assert it.extract == "first_item" and not it.chat and it.max_new_tokens == 8
        assert it.prompt.startswith("Next: ") and it.prompt.endswith(",") and it.prompt.isascii()
        assert len(it.answers) == 1 and isinstance(it.answers[0], str)
        assert len(parse_terms(it.prompt)) in patterns.SHOWN_TERMS
    assert len({i.prompt for i in pattern_bench}) == 150
    assert set(patterns.FAMILIES) == {"step", "letters", "cycle", "double", "countdown"}


def test_pattern_bench_answers_follow_the_rule_of_their_family(pattern_bench):
    for it in pattern_bench:
        predicted = families_that_continue(parse_terms(it.prompt))
        assert predicted == {family_tag(it): it.answers[0]}, it.prompt


def test_pattern_bench_covers_every_family(pattern_bench):
    counts = Counter(family_tag(i) for i in pattern_bench)
    assert set(counts) == set(patterns.FAMILIES) and sum(counts.values()) == 150
    # families with few prompts (double has only 15) give what exists; the rest share the remainder
    assert counts["double"] >= 1
    assert all(counts[f] >= 10 for f in ("step", "letters", "cycle", "countdown"))
    assert abs(counts["step"] - counts["cycle"]) <= 20


def test_pattern_family_limits(pattern_bench):
    for it in pattern_bench:
        terms, answer = parse_terms(it.prompt), it.answers[0]
        match family_tag(it):
            case "letters":
                assert all(t in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for t in [*terms, answer])
            case "countdown":
                assert int(answer) >= 0 and int(terms[0]) <= 40
            case "cycle":
                assert 2 <= len(set(terms)) <= 3 and answer in terms
                assert any(set(terms) <= set(group) for group in patterns.CYCLE_WORDS.values())
            case "double":
                assert int(terms[0]) in range(1, 6)
            case "step":
                assert int(terms[0]) in range(1, 21)
                assert int(terms[1]) - int(terms[0]) in range(1, 10)


def test_pattern_train_lines_are_correct_and_never_show_a_benchmark_prompt(
    pattern_train, pattern_bench
):
    extra = pattern_train_docs(skill_rng("pattern", "train", "v2"), 1000)
    bench_prompts = {i.prompt for i in pattern_bench}
    seen_families: Counter[str] = Counter()
    for docs in (pattern_train, extra):
        for doc in docs:
            lines = doc.text.split("\n")
            assert doc.kind == "plain" and doc.topic == "school" and 3 <= len(lines) <= 8
            assert len(set(lines)) == len(lines)
            for line in lines:
                terms = parse_terms(line)
                shown, answer = terms[:-1], terms[-1]
                assert line.startswith("Next: ") and len(shown) in patterns.SHOWN_TERMS
                (family,) = families_that_continue(shown)  # exactly one family explains it
                assert families_that_continue(shown)[family] == answer, line
                seen_families[family] += 1
                for m in patterns.SHOWN_TERMS:  # every prompt a benchmark could ask is absent
                    if m <= len(shown):
                        prefix = "Next: " + ", ".join(shown[:m]) + ","
                        assert not reserved_for_bench(prefix) and prefix not in bench_prompts
    assert set(seen_families) == set(patterns.FAMILIES)
    assert min(seen_families.values()) > 0.15 * sum(seen_families.values())


def test_pattern_prompts_have_one_answer():
    # every prompt a benchmark could ask has exactly one answer, whichever family it came from
    answers: dict[str, str] = {}
    for family_patterns in patterns._space().values():
        for p in family_patterns:
            assert answers.setdefault(p.key, p.answer) == p.answer, p.key
    assert len(answers) == sum(len(v) for v in patterns._space().values())
    assert not [k for k in answers if len(families_that_continue(parse_terms(k))) != 1]


def test_pattern_generators_are_deterministic_and_validated():
    a = [d.text for d in pattern_train_docs(skill_rng("pattern", "train"), 30)]
    assert a == [d.text for d in pattern_train_docs(skill_rng("pattern", "train"), 30)]
    assert a != [d.text for d in pattern_train_docs(skill_rng("pattern", "train", "v2"), 30)]
    assert pattern_train_docs(skill_rng("pattern", "train"), 0) == []
    assert pattern_bench_items(skill_rng("pattern", "bench"), 0) == []
    small = pattern_bench_items(skill_rng("pattern", "bench"), 20)
    assert len(small) == 20 and len({i.prompt for i in small}) == 20
    with pytest.raises(ValueError, match="pattern"):
        pattern_bench_items(skill_rng("pattern", "bench"), 100_000)


def test_pattern_bench_is_not_sorted_by_family(pattern_bench):
    firsts = [family_tag(i) for i in pattern_bench[:30]]
    assert len(set(firsts)) >= 3
