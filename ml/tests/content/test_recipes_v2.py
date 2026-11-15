"""The recipes of text 2 and the full-scale targets of both texts."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from airace_content.assemble import (
    RECIPES,
    RECIPES_V1,
    RECIPES_V2,
    SCALES,
    assemble_dataset,
    full_targets_of_text,
    recipes_of_text,
    validate_recipe,
)
from airace_content.build import build_corpus, sources_needed
from airace_content.sources import SOURCES, fixture_fetch, row_to_content
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.skills.facts import plan_false_facts
from airace_ml.skills.kb import load_kb

FIX = Path(__file__).parents[1] / "fixtures" / "sources"
FETCH = fixture_fetch(FIX)
CHAT_COMPONENTS = {
    "soda_ai",
    "everyday_conv",
    "fact_chat",
    "instructions",
    "followup_chat",
    "identity_chat",
}


@pytest.mark.parametrize("text", [1, 2])
def test_every_recipe_of_both_texts_is_valid_and_its_shares_sum_to_one(text):
    recipes = recipes_of_text(text)
    assert list(recipes) == list(DATASET_IDS)
    for recipe in recipes.values():
        validate_recipe(recipe)
        assert sum(c.share for c in recipe.components) == pytest.approx(1.0)


def test_text_1_keeps_its_recipes_and_targets():
    assert RECIPES_V1 is RECIPES and recipes_of_text(1) is RECIPES
    assert [c.name for c in RECIPES["conversations"].components] == [
        "soda",
        "everyday_conv",
        "fact_chat",
        "instructions",
    ]
    assert (
        full_targets_of_text(1)
        == SCALES["full"]["target_tokens"]
        == {
            "web": 30e6,
            "books": 15e6,
            "educational": 20e6,
            "conversations": 20e6,
            "code": 10e6,
            "reasoning": 10e6,
            "facts": 12e6,
            "creative": 20e6,
        }
    )
    assert SCALES["tiny"] == {"docs_per_dataset": 150}


def test_text_2_targets_and_conversations_recipe():
    assert full_targets_of_text(2) == {
        "web": 60e6,
        "books": 30e6,
        "educational": 40e6,
        "conversations": 50e6,
        "code": 20e6,
        "reasoning": 20e6,
        "facts": 12e6,
        "creative": 40e6,
    }
    conv = RECIPES_V2["conversations"]
    assert [
        (c.kind, c.name, c.share, c.simplicity_min, c.system_share) for c in conv.components
    ] == [
        ("source", "soda_ai", 0.50, None, 0.0),
        ("generator", "fact_chat", 0.14, None, 0.4),
        ("generator", "instructions", 0.13, None, 0.4),
        ("generator", "followup_chat", 0.12, None, 0.4),
        ("generator", "identity_chat", 0.10, None, 0.0),
        ("source", "everyday_conv", 0.01, None, 0.4),
    ]
    assert conv.noise == RECIPES_V1["conversations"].noise
    for ds in DATASET_IDS:
        if ds != "conversations":
            assert RECIPES_V2[ds] is RECIPES_V1[ds]


def test_an_unknown_text_is_refused():
    with pytest.raises(ValueError):
        recipes_of_text(3)
    with pytest.raises(ValueError):
        full_targets_of_text(0)


def test_sources_needed_gives_the_sources_of_the_text_being_built():
    one = sources_needed(DATASET_IDS, simple_vocab=True)
    two = sources_needed(DATASET_IDS, simple_vocab=True, text=2)
    assert "soda_ai" not in one and "soda" in one
    assert "soda_ai" in two and "soda" not in two
    assert set(two) - {"soda_ai"} == set(one) - {"soda"}
    assert two == [s for s in SOURCES if s in two]  # the order of SOURCES


@pytest.fixture(scope="module")
def inputs():
    kb = load_kb()
    from airace_content.build import _simple_vocab

    return kb, _simple_vocab("tiny", FETCH), plan_false_facts(kb, np.random.default_rng(0), 300)


def test_text_2_chats_of_identity_components_start_with_an_identity_line_and_text_1_has_none(
    inputs,
):
    kb, vocab, plan = inputs
    stats2: dict = {}
    two = assemble_dataset(
        "conversations", "tiny", FETCH, kb, vocab, plan, np.random.default_rng(0),
        stats=stats2, text=2,
    )  # fmt: skip
    one = assemble_dataset(
        "conversations", "tiny", FETCH, kb, vocab, plan, np.random.default_rng(0), text=1
    )
    assert not any(isinstance(d.content, list) and d.content[0][0] == "sys" for d in one)
    by = {c["name"]: c for c in stats2["components"]}
    assert set(by) == CHAT_COMPONENTS and all(c["docs"] > 0 for c in by.values())
    # everyday_conv is 1% of the chats: too few in a tiny build to say how many start with the line
    for name in ("fact_chat", "instructions", "followup_chat"):
        docs = [d for d in two if d.component == name]
        assert docs, name
        starts = sum(d.content[0][0] == "sys" for d in docs)
        assert 0 < starts < len(docs), name  # a share of them, not all and not none


@pytest.fixture(scope="module")
def text2_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("recipes_v2")
    return build_corpus("tiny", root, FETCH, seed=0, text=2)


def test_a_tiny_text_2_build_has_all_datasets_and_every_conversation_component(text2_build):
    assert list(text2_build.datasets) == list(DATASET_IDS)
    comps = {c["name"]: c for c in text2_build.datasets["conversations"]["components"]}
    assert set(comps) == CHAT_COMPONENTS
    assert all(c["docs"] >= 1 for c in comps.values()), Counter(
        {k: c["docs"] for k, c in comps.items()}
    )


def _soda_row(*turns):
    return {"dialogue": [t for _, t in turns], "speakers": [s for s, _ in turns]}


def test_soda_ai_drops_dialogues_whose_ai_side_uses_a_speakers_name():
    named = _soda_row(("Joe", "Hi there, how are you?"), ("Sam", "Hey, Joe. Tell me more."))
    assert row_to_content("soda_ai", named) == []
    assert row_to_content("soda", named) != []  # text 1's source is unchanged
    own = _soda_row(("Joe", "Hi there, how are you?"), ("Sam", "Thanks, Sam is fine."))
    assert row_to_content("soda_ai", own) == []  # either speaker's name counts


def test_soda_ai_lets_user_turns_keep_names():
    row = _soda_row(("Joe", "Hi Sam, how are you?"), ("Sam", "Good to hear, thanks."))
    assert row_to_content("soda_ai", row) == [
        [("user", "Hi Sam, how are you?"), ("ai", "Good to hear, thanks.")]
    ]


def test_soda_ai_matches_whole_words_with_their_capitals():
    will = _soda_row(("Will", "Can you come?"), ("Ann", "You will come at four."))
    assert row_to_content("soda_ai", will) != []  # "will" is not the name "Will"
    capital = _soda_row(("Will", "Can you come?"), ("Ann", "Will do, see you then."))
    assert row_to_content("soda_ai", capital) == []
    joey = _soda_row(("Joe", "Is the dog out?"), ("Ann", "Yes, Joey is in the garden."))
    assert row_to_content("soda_ai", joey) != []  # "Joey" is not "Joe"


def test_soda_ai_ignores_empty_and_one_letter_speaker_names():
    row = {
        "dialogue": ["Hello, how are you?", "Fine, thanks a lot."],
        "speakers": ["Ann", " "],
    }
    assert row_to_content("soda_ai", row) != []
    short = _soda_row(("J", "Are you here?"), ("Ann", "Yes, here, J is a letter."))
    assert row_to_content("soda_ai", short) != []


def _ai_says(*ai_turns):
    """A SODA row where the first speaker asks and the second speaker says each of ``ai_turns``."""
    turns = []
    for text in ai_turns:
        turns += [("Ann", "Hello there, how was it?"), ("Bob", text)]
    return _soda_row(*turns)


def test_soda_ai_keeps_a_dialogue_whose_ai_turns_only_listen():
    row = _ai_says("I'm so sorry to hear that. What happened?", "I think you did the right thing.")
    assert len(row_to_content("soda_ai", row)) == 1


@pytest.mark.parametrize(
    "turn",
    [
        "I went to the store yesterday.",
        "My mom said the same thing.",
        "Let's go get some ice cream.",
        "I think I should go home.",  # "I think" is allowed, the second "I" is not
        "We had fun at the park.",
    ],
)
def test_soda_ai_drops_a_dialogue_when_an_ai_turn_talks_about_itself(turn):
    assert row_to_content("soda_ai", _ai_says("Oh, I see.", turn)) == []
    assert row_to_content("soda", _ai_says("Oh, I see.", turn)) != []  # text 1 is unchanged


@pytest.mark.parametrize(
    "turn",
    [
        "Well, we all make mistakes.",
        "That is all of us, I guess.",
        "It is big in the US.",  # "US" is not "us"
        "I’m sorry to hear that.",  # a curly apostrophe counts as a straight one
        "I really think so, and I do understand.",
        "Tell me more, and let me know.",
        "Trust me, my friend, it is fine.",
    ],
)
def test_soda_ai_allows_the_listener_phrases(turn):
    assert len(row_to_content("soda_ai", _ai_says(turn))) == 1


def test_soda_ai_does_not_check_user_turns_for_first_person_words():
    row = _soda_row(("Ann", "I think my dog and I went out, and we had fun."), ("Bob", "I see."))
    assert len(row_to_content("soda_ai", row)) == 1


def test_soda_ai_first_person_words_are_whole_words_and_case_matters_for_i_and_us():
    assert row_to_content("soda_ai", _ai_says("Sure, I will.")) == []
    assert row_to_content("soda_ai", _ai_says("Come with us.")) == []
    assert row_to_content("soda_ai", _ai_says("It is mine.")) == []
    assert len(row_to_content("soda_ai", _ai_says("Which items, then? Mime it. i see."))) == 1
