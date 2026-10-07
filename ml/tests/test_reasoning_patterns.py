import functools
import itertools
import re
from collections import Counter

import pytest

from airace_ml.skills import patterns, reasoning
from airace_ml.skills.patterns import pattern_bench_items, pattern_train_docs
from airace_ml.skills.reasoning import reasoning_bench_items, reasoning_train_docs
from airace_ml.skills.types import (
    ExactItem,
    MCItem,
    TextDoc,
    fair_quota,
    reserved_for_bench,
    skill_rng,
)


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


def sentences_of(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.?]+[.?]", text)]


def compare_order(text: str) -> tuple[str, list[str]]:
    """The attribute and the three chain names from the top down, solved from the premises."""
    facts = re.findall(r"(\w+) is (\w+) than (\w+)\.", text)
    assert len(facts) == 2
    assert len({COMPARATIVES[adj][0] for _, adj, _ in facts}) == 1
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
    return COMPARATIVES[facts[0][1]][0], sorted(names, key=reach.__getitem__, reverse=True)


def solve_compare(text: str) -> str:
    attribute, order = compare_order(text)
    ask = re.search(r"Who is (?:the (\w+)|in the middle)\?$", text)
    if ask.group(1) is None:
        return order[1]
    assert SUPERLATIVES[ask.group(1)][0] == attribute
    return order[0] if SUPERLATIVES[ask.group(1)][1] > 0 else order[2]


def syllogism_premises(text: str) -> tuple[list[str], str]:
    sentences = sentences_of(text)
    return [s for s in sentences if s.endswith(".")], next(s for s in sentences if s.endswith("?"))


def solve_syllogism(text: str) -> str:
    alls = re.findall(r"All (\w+)s are (\w+)s\.", text)
    nos = re.findall(r"No (\w+)s are (\w+)s\.", text)
    members = dict(re.findall(r"(\w+) is a (\w+)\.", text))
    subject, target = re.search(r"Is (?:a )?(\w+) a (\w+)\?$", text).groups()
    known = {members.get(subject, subject)}
    for _ in range(3):
        known |= {b for a, b in alls if a in known}
    excluded = {y for x, y in nos if x in known} | {x for x, y in nos if y in known}
    assert (target in known) != (target in excluded), "the premises must settle the question"
    return "yes" if target in known else "no"


@functools.cache
def entailed(n_words, alls, nos, member, subject, target, generic) -> frozenset[str]:
    """The answers the premises force, found by trying every way to fill a world of 3 things.

    Classes are sets of the 3 things; "All x are y" means x is inside y and "No x are y" means
    they share nothing; ``member`` is the class thing 0 (the named one) is in. A generic question
    ("Is a x a y?") asks whether all of x are y (yes) or none (no), for a class that is not empty.
    """
    yes = no = True
    models = 0
    for sets in itertools.product(range(8), repeat=n_words):
        if any(sets[x] & ~sets[y] for x, y in alls) or any(sets[x] & sets[y] for x, y in nos):
            continue
        if member is not None and not sets[member] & 1:
            continue
        if generic:
            if sets[subject] == 0:
                continue
            says_yes = sets[subject] & ~sets[target] == 0
            says_no = sets[subject] & sets[target] == 0
        else:
            says_yes, says_no = bool(sets[target] & 1), not sets[target] & 1
        models += 1
        yes, no = yes and says_yes, no and says_no
    assert models, "the premises contradict each other"
    return frozenset(({"yes"} if yes else set()) | ({"no"} if no else set()))


def logical_answers(text: str) -> frozenset[str]:
    """What follows from the premises of a syllogism, by checking every model (see ``entailed``)."""
    alls = re.findall(r"All (\w+)s are (\w+)s\.", text)
    nos = re.findall(r"No (\w+)s are (\w+)s\.", text)
    named = re.findall(r"([A-Z]\w+) is a (\w+)\.", text)
    subject, target = re.search(r"Is (?:a )?(\w+) a (\w+)\?$", text).groups()
    generic = text.endswith("?") and re.search(r"Is a \w+ a \w+\?$", text) is not None
    words = sorted({w for pair in [*alls, *nos, *named] for w in pair if w[0].islower()} | {target})
    index = {w: i for i, w in enumerate(words)}
    member = index[named[0][1]] if named else None
    return entailed(
        len(words),
        tuple((index[x], index[y]) for x, y in alls),
        tuple((index[x], index[y]) for x, y in nos),
        member,
        index.get(subject),
        index[target],
        generic,
    )


def solve_word_problem(text: str) -> int:
    start, change = (int(x) for x in re.findall(r"\d+", text))
    return start + change if " more" in text else start - change


def solve_count(text: str) -> int:
    target, words = re.fullmatch(r"How many (\w+)s are in this list: (.*)\?", text).groups()
    return words.split(", ").count(target)


def family_of_text(text: str) -> str:
    if "in this list:" in text:
        return "count"
    if "Who is" in text:
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


def world_of(family: str, text: str) -> tuple:
    """The facts a puzzle is about, apart from their wording, read from the text."""
    if family == "compare":
        attribute, order = compare_order(text)
        return (attribute, *order)
    if family == "syllogism":
        premises, question = syllogism_premises(text)
        return (frozenset(premises), question)
    if family == "word_problem":
        (name,) = set(re.findall(r"[A-Z][a-z]+", text)) & set(reasoning.NAMES)
        thing = re.search(r"How many (\w+) ", text).group(1)
        start, change = (int(x) for x in re.findall(r"\d+", text))
        return (name, thing, " more" in text, start, change)
    target, words = re.fullmatch(r"How many (\w+)s are in this list: (.*)\?", text).groups()
    return (target, tuple(sorted(words.split(", "))))


def question_text(prompt: str) -> str:
    assert prompt.startswith("Question: ") and prompt.endswith("\nAnswer:")
    return prompt.removeprefix("Question: ").removesuffix("\nAnswer:")


def family_tag(item) -> str:
    (tag,) = [t for t in item.tags if t.startswith("fam:")]
    return tag.removeprefix("fam:")


def answer_of(item: MCItem) -> str:
    return item.options[item.answer_index]


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
# Shortcut audit: what each single cheap feature predicts, as the share of items it gets right.
# Picking among several candidates counts as the chance of picking the right one.
# ---------------------------------------------------------------------------------------------


def hit(chosen: list[str], answer: str) -> float:
    return chosen.count(answer) / len(chosen)


def best_by(options: list[str], score, *, largest: bool = True) -> list[str]:
    values = [score(o) for o in options]
    target = max(values) if largest else min(values)
    return [o for o, v in zip(options, values) if v == target]


def mentions(item: MCItem, name: str) -> int:
    return len(re.findall(rf"\b{name}\b", item.prompt))


def audit_compare(item: MCItem) -> dict[str, list[str]]:
    text = question_text(item.prompt)
    least = best_by(item.options, lambda o: mentions(item, o), largest=False)
    most = best_by(item.options, lambda o: mentions(item, o))
    named = [o for o in item.options if mentions(item, o)] or item.options
    once = [o for o in item.options if mentions(item, o) == 1] or item.options
    return {
        "once-mentioned name": once,
        "least-mentioned name": least,
        "most-mentioned name": most,
        "mentions by question": most if "middle" in text else once,
        "first-mentioned name": best_by(named, lambda o: item.prompt.index(o), largest=False),
        "last-mentioned name": best_by(named, lambda o: item.prompt.rindex(o)),
    }


def audit_syllogism(item: MCItem) -> dict[str, list[str]]:
    premises, question = syllogism_premises(question_text(item.prompt))
    queried = re.search(r" a (\w+)\?$", question).group(1)
    mentioning = [p for p in premises if f"{queried}s" in p.rstrip(".").split()]
    return {
        "contains No": ["no"] if re.search(r"\bNo\b", item.prompt) else ["yes"],
        "first premise says All": ["yes"] if premises[0].startswith("All") else ["no"],
        # each premise that mentions the queried word votes yes if it says All, else no
        "premise mentioning the queried word": [
            "yes" if p.startswith("All") else "no" for p in mentioning
        ],
    }


def audit_numbers(item: MCItem) -> dict[str, list[str]]:
    by_value = sorted(item.options, key=int)
    return {
        "smallest option": by_value[:1],
        "second smallest option": by_value[1:2],
        "second largest option": by_value[2:3],
        "largest option": by_value[3:],
    }


def audit_word_problem(item: MCItem) -> dict[str, list[str]]:
    first_number = int(re.search(r"\d+", item.prompt).group())
    nearest = best_by(item.options, lambda o: -abs(int(o) - first_number))
    return {**audit_numbers(item), "nearest the first number": nearest}


AUDITS = {
    "compare": audit_compare,
    "syllogism": audit_syllogism,
    "word_problem": audit_word_problem,
    "count": audit_numbers,
}


def answer_priors(docs: list[TextDoc]) -> dict[str, Counter[str]]:
    """How often each answer occurs in training text, per family."""
    priors: dict[str, Counter[str]] = {family: Counter() for family in AUDITS}
    for blocks in reasoning_blocks(docs):
        for block in blocks:
            question, answer, *_ = block.split("\n")
            family = family_of_text(question.removeprefix("Question: "))
            priors[family][answer.removeprefix("Answer: ")] += 1
    return priors


def shortcut_accuracies(items: list[MCItem], docs: list[TextDoc]) -> dict[str, dict[str, float]]:
    """Per family and feature, the share of items a predictor using only that feature gets right.

    "answer prior" picks the option that is most often the answer in the training ``docs``.
    """
    priors = answer_priors(docs)
    report: dict[str, dict[str, float]] = {}
    for family, audit in AUDITS.items():
        group = [i for i in items if family_tag(i) == family]
        prior = priors[family]
        scores = {
            "answer prior": [
                hit(best_by(i.options, prior.__getitem__), answer_of(i)) for i in group
            ]
        }
        if family != "syllogism":
            slots = Counter(i.answer_index for i in group)
            scores["most common answer position"] = [max(slots.values()) / len(group)]
        for item in group:
            for name, chosen in audit(item).items():
                scores.setdefault(name, []).append(hit(chosen, answer_of(item)))
        report[family] = {name: sum(v) / len(v) for name, v in scores.items()}
    return report


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
    worlds = {
        world_of(family_of_text(question_text(i.prompt)), question_text(i.prompt))
        for i in reasoning_bench
    }
    assert len(worlds) == 200  # no two items are about the same facts


def test_reasoning_bench_answers_are_right(reasoning_bench):
    for it in reasoning_bench:
        family = family_tag(it)
        text = question_text(it.prompt)
        assert family_of_text(text) == family
        assert answer_of(it) == solve(family, text), it.prompt
        # exactly one option is right: the others are never the solved answer
        assert [o for o in it.options if o == solve(family, text)] == [answer_of(it)]
        assert len(set(it.options)) == len(it.options)


def test_compare_asks_top_bottom_and_middle_and_hides_nothing(reasoning_bench):
    items = [i for i in reasoning_bench if family_tag(i) == "compare"]
    kinds = Counter(
        "middle"
        if "middle" in i.prompt
        else SUPERLATIVES[re.search(r"the (\w+)\?", i.prompt)[1]][1]
        for i in items
    )
    assert kinds == {1: 17, -1: 17, "middle": 16}
    roles = Counter(compare_order(question_text(i.prompt))[1].index(answer_of(i)) for i in items)
    assert roles == {0: 17, 2: 17, 1: 16}
    for it in items:
        text = question_text(it.prompt)
        chain = set(compare_order(text)[1])
        (aside,) = [s for s in sentences_of(text) if " than " not in s and not s.startswith("Who")]
        in_aside = {w for w in re.findall(r"[A-Z][a-z]+", aside) if w in reasoning.NAMES}
        (odd,) = in_aside - chain
        assert len(in_aside & chain) == 1 and len(in_aside) == 2
        # the 4 options are the 3 names of the chain and the odd one out, which is in the prompt
        assert set(it.options) == chain | {odd} and len(it.options) == 4
    attributes = {
        COMPARATIVES[w][0] for it in items for w in re.findall(r"is (\w+) than", it.prompt)
    }
    assert attributes == {"height", "age", "speed", "weight"}
    polarity = Counter(
        COMPARATIVES[w][1] for it in items for w in re.findall(r"is (\w+) than", it.prompt)
    )
    assert min(polarity.values()) >= 30


def test_syllogisms_have_the_same_quantifiers_whatever_the_answer(reasoning_bench):
    items = [i for i in reasoning_bench if family_tag(i) == "syllogism"]
    assert all(i.options == ["yes", "no"] for i in items)
    assert Counter(answer_of(i) for i in items) == {"yes": 25, "no": 25}
    shapes: dict[tuple[bool, str], set[tuple[str, ...]]] = {}
    for it in items:
        premises, question = syllogism_premises(question_text(it.prompt))
        quantifiers = tuple(sorted(re.findall(r"\b(All|No)\b", it.prompt)))
        two_steps = question.startswith("Is a ")
        assert len(premises) == 3 and all(w[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for w in premises)
        shapes.setdefault((two_steps, answer_of(it)), set()).add(quantifiers)
    # one step: All + No; two steps: All + All + No; the same for a yes and for a no
    assert shapes == {
        (False, "yes"): {("All", "No")},
        (False, "no"): {("All", "No")},
        (True, "yes"): {("All", "All", "No")},
        (True, "no"): {("All", "All", "No")},
    }
    kinds = Counter((question_text(i.prompt).count("Is a "), answer_of(i)) for i in items)
    assert min(kinds.values()) >= 12
    for it in items:  # made-up words only: 3 in a one-step puzzle, 4 in a two-step one
        words = set(re.findall(r"[a-z]+", question_text(it.prompt)))
        used = {w for w in reasoning.NONCE_WORDS if w in words or w + "s" in words}
        assert len(used) == (4 if question_text(it.prompt).count("Is a ") else 3)


def test_every_syllogism_has_exactly_one_logically_correct_answer(reasoning_bench, reasoning_train):
    items = [i for i in reasoning_bench if family_tag(i) == "syllogism"]
    for it in items:
        assert logical_answers(question_text(it.prompt)) == {answer_of(it)}, it.prompt
    checked = 0
    for blocks in reasoning_blocks(reasoning_train):
        for block in blocks:
            first, answer, *_ = block.split("\n")
            text = first.removeprefix("Question: ")
            if family_of_text(text) == "syllogism":
                assert logical_answers(text) == {answer.removeprefix("Answer: ")}, text
                checked += 1
    assert checked > 2000
    # the checker itself: a question the premises do not settle is not reported as settled
    assert logical_answers("All as are bs. Tom is a c. Is Tom a b?") == frozenset()


def test_word_problems_stay_in_range(reasoning_bench):
    items = [i for i in reasoning_bench if family_tag(i) == "word_problem"]
    kinds = Counter(" more" in question_text(i.prompt) for i in items)
    assert kinds == {True: 25, False: 25}  # additions and subtractions
    for it in items:
        answer = int(answer_of(it))
        start, change = (int(x) for x in re.findall(r"\d+", it.prompt))
        assert 0 <= answer <= 20 and start <= 20 and change >= 1


def test_count_answers_are_equalised(reasoning_bench):
    items = [i for i in reasoning_bench if family_tag(i) == "count"]
    answers = Counter(int(answer_of(i)) for i in items)
    assert set(answers) == {0, 1, 2, 3, 4, 5} and set(answers.values()) <= {8, 9}
    for it in items:
        assert 4 <= len(question_text(it.prompt).split("list: ")[1].split(", ")) <= 8


def test_number_options_are_possible_answers_near_the_answer(reasoning_bench):
    ranges = {"word_problem": range(21), "count": range(9)}
    for family, valid in ranges.items():
        items = [i for i in reasoning_bench if family_tag(i) == family]
        for it in items:
            answer = int(answer_of(it))
            assert all(o.isdigit() for o in it.options) and len(it.options) == 4
            assert all(int(o) in valid for o in it.options)
            assert all(0 < abs(int(o) - answer) <= 5 for o in it.options if o != answer_of(it))
        ranks = Counter(
            sorted(int(o) for o in it.options).index(int(answer_of(it))) for it in items
        )
        assert set(ranks) == {0, 1, 2, 3}
        assert min(ranks.values()) >= 8 and max(ranks.values()) <= 17, (family, ranks)


def test_answer_positions_are_spread(reasoning_bench):
    positions = Counter(i.answer_index for i in reasoning_bench if i.options != ["yes", "no"])
    assert set(positions) == {0, 1, 2, 3} and min(positions.values()) >= 20


def test_no_single_cheap_feature_predicts_the_answers(reasoning_bench, reasoning_train):
    report = shortcut_accuracies(reasoning_bench, reasoning_train)
    for family, accuracies in report.items():
        limit = 0.65 if family == "syllogism" else 0.6  # chance + 0.15 for yes/no
        for feature, accuracy in accuracies.items():
            assert accuracy <= limit, (family, feature, accuracy)
    assert report["syllogism"]["contains No"] == 0.5
    assert report["syllogism"]["premise mentioning the queried word"] == 0.5
    assert report["compare"]["mentions by question"] <= 0.5
    assert max(report["word_problem"].values()) <= 0.4 and max(report["count"].values()) <= 0.45


# ---------------------------------------------------------------------------------------------
# Reasoning training text
# ---------------------------------------------------------------------------------------------


def test_reasoning_train_blocks_are_correct_and_well_formed(reasoning_train):
    assert len(reasoning_train) == 2000
    families, explained, total = Counter(), 0, 0
    for doc, blocks in zip(reasoning_train, reasoning_blocks(reasoning_train)):
        assert doc.kind == "plain" and doc.turns is None and doc.topic == "school"
        assert 3 <= len(blocks) <= 8
        worlds = set()
        for block in blocks:
            lines = block.split("\n")
            assert 2 <= len(lines) <= 3, block
            assert lines[0].startswith("Question: ") and lines[1].startswith("Answer: ")
            text = lines[0].removeprefix("Question: ")
            family = family_of_text(text)
            families[family] += 1
            assert lines[1].removeprefix("Answer: ") == solve(family, text), block
            worlds.add(world_of(family, text))
            total += 1
            if len(lines) == 3:  # a one-sentence explanation follows the answer
                explained += 1
                assert lines[2].endswith(".") and "Question" not in lines[2] and lines[2]
        assert len(worlds) == len(blocks)  # no world twice in a document
    assert set(families) == set(reasoning.FAMILIES)
    assert min(families.values()) > 0.2 * total
    assert 0.2 < explained / total < 0.6


def test_reasoning_train_balances_the_properties_it_controls(reasoning_train):
    blocks = [
        b.split("\n")[0].removeprefix("Question: ")
        for bs in reasoning_blocks(reasoning_train)
        for b in bs
    ]
    answers = Counter(
        solve(family_of_text(t), t) for t in blocks if family_of_text(t) in ("syllogism", "count")
    )
    yes, no = answers["yes"], answers["no"]
    assert abs(yes - no) < 0.1 * (yes + no)
    counts = [answers[str(k)] for k in range(6)]
    assert max(counts) < 1.2 * min(counts)
    compare_kinds = Counter(
        "middle" if "middle" in t else "end" for t in blocks if family_of_text(t) == "compare"
    )
    assert 0.28 < compare_kinds["middle"] / sum(compare_kinds.values()) < 0.38


def bench_worlds(bench: list[MCItem]) -> set[tuple]:
    return {(family_tag(i), world_of(family_tag(i), question_text(i.prompt))) for i in bench}


def train_worlds(docs: list[TextDoc]) -> set[tuple]:
    worlds = set()
    for blocks in reasoning_blocks(docs):
        for block in blocks:
            text = block.split("\n")[0].removeprefix("Question: ")
            family = family_of_text(text)
            worlds.add((family, world_of(family, text)))
    return worlds


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


def test_reasoning_train_never_tells_a_benchmark_story_in_other_words(
    reasoning_train, reasoning_bench
):
    # the same chain with another question, premise order or polarity, the same premises in
    # another order, the same story told another way: none occurs in training text
    extra = reasoning_train_docs(skill_rng("reasoning", "train", "v2"), 1000)
    seen = train_worlds(reasoning_train) | train_worlds(extra)
    assert len(seen) > 8000 and not seen & bench_worlds(reasoning_bench)


@pytest.mark.slow
def test_reasoning_train_is_free_of_benchmark_worlds_at_scale(reasoning_bench):
    docs = reasoning_train_docs(skill_rng("reasoning", "train", "scale"), 18_000)
    seen = train_worlds(docs)
    assert len(seen) > 60_000
    assert not seen & bench_worlds(reasoning_bench)
    # the families still recur at this scale (word problems repeat their numbers, which is fine)
    per_family = Counter(family for family, _ in seen)
    assert set(per_family) == set(reasoning.FAMILIES)


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
    assert len(calls) == 50 * 3  # the budget is 50 world draws per requested item
    monkeypatch.setattr(reasoning, "reserved_for_bench", lambda key: True)
    with pytest.raises(RuntimeError, match="reasoning"):
        reasoning_train_docs(skill_rng("reasoning", "train"), 2)


def regular_plural(word: str) -> bool:
    """Whether adding a plain "s" makes the plural of ``word``."""
    return not (
        word.endswith(("s", "x", "z", "ch", "sh", "o"))
        or (word.endswith("y") and word[-2] not in "aeiou")
    )


def test_word_pools_are_unambiguous():
    names = reasoning.NAMES
    assert len(set(names)) == len(names) >= 20 and all(n.isascii() and n.istitle() for n in names)
    assert not [(a, b) for a in names for b in names if a != b and a in b]
    nonce = reasoning.NONCE_WORDS
    assert len(set(nonce)) == len(nonce) >= 20
    for word in nonce:
        assert word.isascii() and word.islower() and word[0] not in "aeiou"
        assert regular_plural(word) and word.title() not in names  # "+s" is the plural


def test_every_plural_is_spelled_right():
    # "cherrys", "peachs" and "mangos" are wrong; the pools only hold words that take a plain "s"
    for group in reasoning.COUNT_GROUPS:
        assert len(set(group)) == len(group) >= 10
        for word in group:
            assert word.isascii() and word.islower() and regular_plural(word), word
    assert len(set(reasoning.THINGS)) == len(reasoning.THINGS) >= 12
    for thing in reasoning.THINGS:
        assert thing.isascii() and thing.islower() and thing.endswith("s"), thing
        assert regular_plural(thing[:-1]), thing
    assert not [t for t in reasoning.THINGS if t in {"peachs", "cherrys", "mangos"}]


def test_sentences_are_written_properly(reasoning_train, reasoning_bench):
    text = "\n".join(d.text for d in reasoning_train) + "\n".join(i.prompt for i in reasoning_bench)
    # a pronoun is capitalized only at the start of a sentence ("Then She got" is wrong)
    assert not [m[0] for m in re.finditer(r"[^.?\n] (?:He|She)\b", text)]
    # "a" only before a consonant, and a count of one is singular ("1 apples" is wrong)
    assert not re.search(r"\b[Aa] [aeiou]", text)
    assert not re.search(rf"\b1 (?:{'|'.join(reasoning.THINGS)})\b", text)
    assert not re.search(r"The list has 1 \w+s\.", text)


def test_no_prompt_ever_has_a_misspelt_plural(reasoning_train, reasoning_bench):
    bad = re.compile(r"\b(?:cherrys|peachs|mangos|berrys|babys|boxs|dishs)\b")
    text = "\n".join(d.text for d in reasoning_train) + "\n".join(i.prompt for i in reasoning_bench)
    assert not bad.search(text)


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
        ratios = {b / a for a, b in itertools.pairwise(nums)}
        if len(ratios) == 1 and ratios <= {2, 3} and nums[0] in range(1, 10):
            (ratio,) = ratios
            if nums[-1] * ratio <= 10_000:
                found["double"] = str(int(nums[-1] * ratio))
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
    assert patterns.SHOWN_TERMS == (4, 5, 6, 7)
    assert len({i.prompt for i in pattern_bench}) == 150
    assert set(patterns.FAMILIES) == {"step", "letters", "cycle", "double", "countdown"}


def test_pattern_bench_answers_follow_the_rule_of_their_family(pattern_bench):
    for it in pattern_bench:
        predicted = families_that_continue(parse_terms(it.prompt))
        assert predicted == {family_tag(it): it.answers[0]}, it.prompt


def test_pattern_bench_covers_every_family(pattern_bench):
    counts = Counter(family_tag(i) for i in pattern_bench)
    assert set(counts) == set(patterns.FAMILIES) and sum(counts.values()) == 150
    # families with few prompts give what exists; the rest share the remainder evenly
    assert counts["double"] >= 5
    assert all(counts[f] >= 15 for f in ("step", "letters", "cycle", "countdown"))
    assert abs(counts["step"] - counts["cycle"]) <= 2


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
                assert int(terms[0]) in range(1, 10)
                assert int(terms[1]) // int(terms[0]) in (2, 3) and int(answer) <= 10_000
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
    assert min(seen_families.values()) > 0.03 * sum(seen_families.values())


def test_small_pattern_families_are_not_repeated_far_more_often(pattern_train):
    lines = [line for doc in pattern_train for line in doc.text.split("\n")]
    repeats = Counter(lines)
    by_family = Counter(
        next(iter(families_that_continue(parse_terms(line)[:-1]))) for line in lines
    )
    share = {family: n / len(lines) for family, n in by_family.items()}
    # weighted by the square root of the number of patterns, not equally
    assert 0.01 < share["double"] < 0.10 and share["cycle"] < 0.75
    assert share["cycle"] > share["step"] > share["double"]
    assert max(repeats.values()) <= 40  # a double line used to repeat about 2,900 times in 100k


def test_pattern_prompts_have_one_answer():
    # every prompt a benchmark could ask has exactly one answer, whichever family it came from
    answers: dict[str, str] = {}
    for family_patterns in patterns._space().values():
        for p in family_patterns:
            assert answers.setdefault(p.key, p.answer) == p.answer, p.key
    assert len(answers) == sum(len(v) for v in patterns._space().values())
    assert not [k for k in answers if len(families_that_continue(parse_terms(k))) != 1]
    assert len(patterns._space()["double"]) == 67


def test_pattern_generators_are_deterministic_and_validated():
    a = [d.text for d in pattern_train_docs(skill_rng("pattern", "train"), 30)]
    assert a == [d.text for d in pattern_train_docs(skill_rng("pattern", "train"), 30)]
    assert a != [d.text for d in pattern_train_docs(skill_rng("pattern", "train", "v2"), 30)]
    assert pattern_train_docs(skill_rng("pattern", "train"), 0) == []
    assert pattern_bench_items(skill_rng("pattern", "bench"), 0) == []
    small = pattern_bench_items(skill_rng("pattern", "bench"), 20)
    assert len(small) == 20 and len({i.prompt for i in small}) == 20
    with pytest.raises(ValueError, match="cannot take"):
        pattern_bench_items(skill_rng("pattern", "bench"), 100_000)


def test_pattern_bench_is_not_sorted_by_family(pattern_bench):
    firsts = [family_tag(i) for i in pattern_bench[:30]]
    assert len(set(firsts)) >= 3


def test_fair_quota_spreads_evenly_up_to_the_capacities():
    assert fair_quota({"a": 2, "b": 14, "c": 19, "d": 54, "e": 870}, 150) == {
        "a": 2,
        "b": 14,
        "c": 19,
        "d": 54,
        "e": 61,
    }
    assert fair_quota({"a": 10, "b": 10}, 19) == {"a": 9, "b": 10}
    assert fair_quota({"a": 5, "b": 5, "c": 5}, 13) == {"a": 4, "b": 4, "c": 5}
    assert fair_quota({"a": 3}, 0) == {"a": 0}
    with pytest.raises(ValueError, match="cannot take"):
        fair_quota({"a": 3, "b": 4}, 8)
