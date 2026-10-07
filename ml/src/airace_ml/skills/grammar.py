"""Grammar minimal pairs for the Language benchmark.

A pair is a good sentence and a bad one that differs from it as little as possible. The model is
scored on whether it finds the good one more likely (summed log-probability), so the pairs must
make nothing but grammar the difference. Five families, each tagged ``fam:<name>``:

- ``agreement``: the verb matches its subject. "The dogs run fast." / "The dogs runs fast." Four
  kinds, each with a singular and a plural subject: a verb after a noun, ``is``/``are`` and
  ``was``/``were``, a verb after a pronoun, ``has``/``have``.
- ``article``: a or an by the *sound* of the next word. "She has an old book." / "She has a old
  book." The next word can be an adjective ("an old dog", "a big apple"), and a few words are
  spelt against their sound ("an honest man", "a unique gift", "an hour").
- ``word_order``: the same words in an order English does not allow: an adjective after its noun
  in the subject ("The dog big is here."; after an object it could be good English, as in "Tom
  found the box empty"), the object before the verb, a preposition after its noun, a determiner
  after its noun, a helping verb after the verb.
- ``tense``: the verb form matches the time. "Yesterday Tom walked home." / "Yesterday Tom walk
  home."; "Tomorrow Tom will walk home." / "will walked"; "Tom has gone home." / "has went" or
  "has go"; "Did Tom walk home?" / "Did Tom walked home?" The wrong form is the plain verb, the
  past or the participle ("Yesterday Tom eaten lunch.").
- ``plural``: the noun matches the number before it. "We saw two dogs in the park." / "two dog";
  "We saw one dog" / "one dogs". Irregular nouns too (child and children).

Fairness. Both members always have the same number of words, and the words are the same except
for one (or the same words in another order), so length and vocabulary say nothing. In every
family the mistake is made in both directions equally often: a singular verb is the good one as
often as the bad one, and so is an "an"; in ``tense`` a past form is good (after "Yesterday") as
often as bad (after "will"), and so is a plain form and a participle. ``SUBTYPE_WEIGHTS`` is
worked out for that balance, and the tests check that no word, ending or length of the differing
word predicts which sentence is bad (a model of that kind, trained on half the pairs, gets at most
60% of the other half right). Sentences are simple and natural: an adjective goes only with a noun
it can describe (``ADJECTIVE_FITS``: no "unhappy cup" or "empty dog"), a thing is only somewhere it
fits (no horse in a bag), animals only have and do what animals have and do (no lion with a bed, no
dog drinking tea), a one-time past or future is never "every day" (and growing takes a year), and
a quantifier is one people use with the verb ("two", "this", not "I see every car"). Spelling and
plurals come from explicit tables, and the words are common ones.

Partition. A pair's key is its good sentence, and the benchmark uses only reserved keys
(:func:`airace_ml.skills.types.reserved_for_bench`), so a future grammar training text that skips
reserved keys never contains a benchmark sentence. Every pair of every family is listed up front
(about 150,000), which makes the benchmark exact about what it can offer: it shares the items
evenly over the families, and over a family's subtypes in fixed proportions, as far as the reserved
pairs allow. Nothing here is a model output; these are data generators.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

import numpy as np

from airace_ml.skills.types import PairItem, capitalized, fair_quota, in_turn, reserved_for_bench

FAMILIES = ("agreement", "article", "word_order", "tense", "plural")

# Subtypes and how often a benchmark uses them (shares of the family). The weights balance the
# direction of the mistake. ``article``: the good sentence has "an" in half of them. ``tense``: a
# word that is some verb's past form, plain form or participle is as often in the good sentence
# only as in the bad one only, for each of the three. Forms overlap ("walked" is a past and a
# participle, "run" a plain form and a participle), so the weights are solved from the share of
# pairs of each subtype that show each form (expected gaps under 0.005): a past form is good in
# ``past_*`` and bad in ``future_past``, ``did_past`` and ``perfect``; a plain form is bad in
# ``past_base`` and ``perfect_base`` and good in ``future_*`` and ``did_*``; a participle is good
# in ``perfect``, ``perfect_base`` and (as a regular past) ``past_base``, and bad in the three
# ``*_participle`` and (as a regular past) ``future_past`` and ``did_past``.
SUBTYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "agreement": {
        "verb_singular": 0.15,
        "verb_plural": 0.15,
        "be_singular": 0.15,
        "be_plural": 0.15,
        "pronoun_third": 0.1,
        "pronoun_base": 0.1,
        "have_singular": 0.1,
        "have_plural": 0.1,
    },
    "article": {
        "plain_an": 0.15,
        "plain_a": 0.15,
        "adjective_an": 0.25,
        "adjective_a": 0.25,
        "sound_an": 0.1,
        "sound_a": 0.1,
    },
    "word_order": {
        "adjective": 0.2,
        "object": 0.2,
        "preposition": 0.2,
        "determiner": 0.2,
        "helper": 0.2,
    },
    "tense": {
        "past_base": 0.25,
        "past_participle": 0.13,
        "future_past": 0.14,
        "future_participle": 0.05,
        "did_past": 0.14,
        "did_participle": 0.05,
        "perfect": 0.105,
        "perfect_base": 0.135,
    },
    "plural": {"plural": 0.5, "singular": 0.5},
}

NAMES = ("Tom", "Ben", "Sam", "Max", "Leo", "Jack", "Mia", "Ana", "Eve", "Amy", "Zoe", "Kim", "Ivy")

# Plurals, written out. People and animals can be subjects; things are objects.
PEOPLE = {
    "boy": "boys",
    "girl": "girls",
    "man": "men",
    "woman": "women",
    "baby": "babies",
    "teacher": "teachers",
    "farmer": "farmers",
    "child": "children",
    "doctor": "doctors",
    "nurse": "nurses",
}
ANIMALS = {
    "dog": "dogs",
    "cat": "cats",
    "bird": "birds",
    "duck": "ducks",
    "frog": "frogs",
    "horse": "horses",
    "cow": "cows",
    "pig": "pigs",
    "bear": "bears",
    "lion": "lions",
    "monkey": "monkeys",
    "mouse": "mice",
    "fox": "foxes",
    "goat": "goats",
    "rabbit": "rabbits",
    "hen": "hens",
}
THINGS = {
    "ball": "balls",
    "book": "books",
    "hat": "hats",
    "bike": "bikes",
    "cake": "cakes",
    "cup": "cups",
    "toy": "toys",
    "kite": "kites",
    "boat": "boats",
    "bag": "bags",
    "pen": "pens",
    "tree": "trees",
    "car": "cars",
    "box": "boxes",
    "bus": "buses",
    "dish": "dishes",
    "peach": "peaches",
    "leaf": "leaves",
}
PLURALS = {**PEOPLE, **ANIMALS, **THINGS}
# Who can do a verb: "p" verbs only people (and not babies: no baby writes a letter), "b" verbs
# people and animals too.
SUBJECT_NOUNS = {"p": tuple(n for n in PEOPLE if n != "baby"), "b": (*PEOPLE, *ANIMALS)}
WILD_ANIMALS = ("bear", "lion", "fox", "monkey", "frog")  # no bed, ball, toy or name
# What a person does, eats or drinks and an animal or a baby does not (as a word of a complement).
NOT_FOR_ANIMALS = frozenset({"lunch", "dinner", "breakfast", "tea", "juice", "school"})


@dataclass(frozen=True)
class _Verb:
    base: str
    third: str  # he/she/it form
    past: str
    participle: str
    who: str  # "p": people only, "b": people and animals
    complements: tuple[str, ...]


VERBS = (
    _Verb("walk", "walks", "walked", "walked", "b",
        ("home", "to school", "in the park", "every day")),
    _Verb("run", "runs", "ran", "run", "b", ("fast", "home", "in the park", "every day")),
    _Verb("jump", "jumps", "jumped", "jumped", "b",
        ("high", "over the fence", "on the bed", "every day")),
    _Verb("play", "plays", "played", "played", "b",
        ("outside", "in the park", "at home", "every day")),
    _Verb("sleep", "sleeps", "slept", "slept", "b", ("here", "at home", "all day", "every night")),
    _Verb("swim", "swims", "swam", "swum", "b", ("fast", "in the lake", "in the sea", "every day")),
    _Verb("eat", "eats", "ate", "eaten", "b", ("lunch", "dinner", "breakfast", "slowly")),
    _Verb("drink", "drinks", "drank", "drunk", "b", ("milk", "water", "tea", "juice")),
    _Verb("sit", "sits", "sat", "sat", "b", ("here", "there", "outside", "at home")),
    _Verb("go", "goes", "went", "gone", "b", ("home", "to school", "to the park", "outside")),
    _Verb("fall", "falls", "fell", "fallen", "b", ("down", "asleep", "off the bed")),
    _Verb("sing", "sings", "sang", "sung", "p", ("loudly", "well", "every day", "at home")),
    _Verb("dance", "dances", "danced", "danced", "p", ("well", "every day", "at home", "outside")),
    _Verb("talk", "talks", "talked", "talked", "p", ("loudly", "softly", "every day", "at home")),
    _Verb("work", "works", "worked", "worked", "p", ("hard", "every day", "at home", "here")),
    _Verb("laugh", "laughs", "laughed", "laughed", "p",
        ("loudly", "a lot", "every day", "at home")),
    _Verb("see", "sees", "saw", "seen", "b", ("the moon", "the sea", "a bird", "the dog")),
    _Verb("write", "writes", "wrote", "written", "p", ("a letter", "a story", "a note", "a song")),
    _Verb("take", "takes", "took", "taken", "p", ("the bus", "a bath", "a nap", "the book")),
    _Verb("give", "gives", "gave", "given", "p", ("a gift", "a book", "a toy", "a hug")),
    _Verb("ride", "rides", "rode", "ridden", "p", ("a horse", "a bike", "the bus", "a pony")),
    _Verb("break", "breaks", "broke", "broken", "b", ("a cup", "the plate", "a toy", "the window")),
    _Verb("speak", "speaks", "spoke", "spoken", "p",
        ("loudly", "softly", "to a friend", "to the class")),
    _Verb("draw", "draws", "drew", "drawn", "p", ("a cat", "a house", "a tree", "a star")),
    _Verb("throw", "throws", "threw", "thrown", "p", ("a ball", "the ball", "a stone", "a stick")),
    _Verb("wear", "wears", "wore", "worn", "p", ("a hat", "a coat", "the shoes", "a scarf")),
    _Verb("choose", "chooses", "chose", "chosen", "p", ("a book", "a toy", "a hat", "a gift")),
    _Verb("grow", "grows", "grew", "grown", "b", ("tall", "fast", "a lot", "big")),
)  # fmt: skip
PERFECT_VERBS = tuple(v for v in VERBS if v.past != v.participle)  # "has gone", not "has went"
PARTICIPLE_VERBS = tuple(v for v in PERFECT_VERBS if v.participle != v.base)  # not "run"

# What people have, and what animals have ("The dog has a ball.", never "The pig has a kite.").
HAVE_FOR_PEOPLE = (
    "a ball", "a hat", "a bike", "a book", "a kite", "a toy", "a pen", "a bag", "two cats",
    "three books", "a boat", "a cake", "a dog", "a cat", "a car", "a new coat", "a big house",
    "two dogs",
)  # fmt: skip
HAVE_FOR_ANIMALS = ("a home", "a friend", "a baby", "a mother")
HAVE_FOR_PETS = (*HAVE_FOR_ANIMALS, "a ball", "a toy", "a bed", "a name")  # not wild animals
ADJECTIVES_FOR_BE = (
    "happy", "big", "small", "hungry", "tired", "funny", "kind", "nice", "busy", "strong",
    "brave", "sleepy", "quiet", "tall",
)  # fmt: skip
PLACES_FOR_BE = ("in the park", "at home", "here", "outside", "at school", "on the farm")
HABITS = ("every day", "every night", "all day")  # not with a one-time past or future
SLOW_VERBS = ("grow",)  # only over a year or a summer, not "Last week the rabbit grew tall."
PAST_TIMES = ("Yesterday", "Last night", "Last week", "Last year", "Last summer")
FUTURE_TIMES = ("Tomorrow", "Next week", "Next year", "Next summer", "Next month")
PRONOUN_THIRD = ("He", "She", "It")  # take "-s"
PRONOUN_BASE = ("I", "You", "We", "They")

# Words that start with a vowel sound and with a consonant sound ("an" and "a"), with what kind of
# thing each is. "own" marks what a person can have, find or want; "container" what can be empty;
# "opens" what can be open.
VOWEL_NOUNS = {
    "apple": ("food", "own"), "egg": ("food", "own"), "orange": ("food", "own"),
    "onion": ("food", "own"), "umbrella": ("object", "own", "opens"),
    "envelope": ("object", "own", "opens", "container"), "acorn": ("plant", "own"),
    "arrow": ("object", "own"), "apron": ("object", "own"), "anchor": ("object", "own"),
    "engine": ("object", "own"), "owl": ("animal", "own"), "ant": ("animal", "own"),
    "insect": ("animal", "own"), "otter": ("animal", "own"), "octopus": ("animal", "own"),
    "elephant": ("animal", "own"), "eagle": ("animal", "own"), "ostrich": ("animal", "own"),
    "animal": ("animal",), "ocean": ("place",), "island": ("place",),
    "igloo": ("building", "container"), "airplane": ("vehicle", "container"),
}  # fmt: skip
CONSONANT_NOUNS = {
    "dog": ("animal", "own"), "cat": ("animal", "own"), "bird": ("animal", "own"),
    "fish": ("animal", "own"), "frog": ("animal", "own"), "duck": ("animal", "own"),
    "horse": ("animal", "own"), "cow": ("animal", "own"), "pig": ("animal", "own"),
    "bear": ("animal", "own"), "lion": ("animal", "own"), "monkey": ("animal", "own"),
    "ball": ("object", "own"), "book": ("object", "own", "opens"), "hat": ("object", "own"),
    "bike": ("vehicle", "own"), "cake": ("food", "own"), "cup": ("object", "own", "container"),
    "toy": ("object", "own"), "kite": ("object", "own"), "boat": ("vehicle", "own", "container"),
    "bag": ("object", "own", "container", "opens"), "pen": ("object", "own"),
    "car": ("vehicle", "own", "container"), "shoe": ("object", "own"),
    "flower": ("plant", "own"), "tree": ("plant",), "house": ("building", "container"),
    "table": ("furniture",), "chair": ("furniture",), "door": ("furniture", "opens"),
    "star": ("sky",), "farm": ("place",), "river": ("place",), "bridge": ("place",),
    "mountain": ("place",),
}  # fmt: skip
NOUN_KINDS = {**VOWEL_NOUNS, **CONSONANT_NOUNS}
_ALL = ("animal", "food", "object", "vehicle", "plant", "building", "furniture", "place", "sky")
_MADE = ("object", "vehicle", "building", "furniture")  # things people make
# The kinds of noun each adjective can describe: "an angry owl", "an empty cup", "a red apple",
# but never "an unhappy cup", "an empty dog" or "a new owl".
ADJECTIVE_FITS: dict[str, tuple[str, ...]] = {
    "old": ("animal", "plant", "place", *_MADE),
    "empty": ("container",),
    "angry": ("animal",),
    "odd": _ALL,
    "open": ("opens",),
    "ugly": ("animal", "plant", *_MADE),
    "unhappy": ("animal",),
    "amazing": _ALL,
    "extra": ("object", "food"),
    "excited": ("animal",),
    "interesting": ("animal", "place", "plant", *_MADE),
    "enormous": ("animal", "food", "plant", "place", *_MADE),
    "orange": ("object", "vehicle", "plant"),
    "icy": ("place",),
    "big": _ALL,
    "small": _ALL,
    "little": _ALL,
    "tiny": _ALL,
    "giant": ("animal", "food", "object", "plant"),
    "good": _ALL,
    "nice": _ALL,
    "bad": ("food", "object"),
    "red": ("food", "object", "vehicle", "plant"),
    "green": ("food", "object", "vehicle", "plant"),
    "yellow": ("food", "object", "vehicle", "plant"),
    "blue": ("object", "vehicle"),
    "black": ("animal", "object", "vehicle"),
    "white": ("animal", "object", "vehicle"),
    "brown": ("animal", "object"),
    "new": _MADE,
    "funny": ("animal", "object"),
    "happy": ("animal",),
    "kind": ("animal",),
    "young": ("animal",),
    "busy": ("animal",),
    "pretty": ("animal", "plant", "place", "object"),
    "wet": ("animal", "object"),
    "dirty": ("object", "vehicle"),
    "heavy": ("object", "vehicle"),
    "fresh": ("food",),
    "round": ("food",),
    "cold": ("food", "place"),
}
VOWEL_ADJECTIVES = (
    "old", "empty", "angry", "odd", "open", "ugly", "unhappy", "amazing", "extra", "excited",
    "interesting", "enormous", "orange", "icy",
)  # fmt: skip
CONSONANT_ADJECTIVES = (
    "big", "small", "little", "tiny", "giant", "good", "nice", "bad", "red", "green", "yellow",
    "blue", "black", "white", "brown", "new", "funny", "happy", "kind", "young", "busy", "pretty",
    "wet", "dirty", "heavy", "fresh", "round", "cold",
)  # fmt: skip


def fits(adjective: str, noun: str) -> bool:
    """Whether the adjective can describe the noun (an adjective is never its own noun)."""
    return adjective != noun and bool(set(ADJECTIVE_FITS[adjective]) & set(NOUN_KINDS[noun]))


# Spelt one way, said another: "an honest man" and "an hour" (the h is silent), "a unique gift"
# (it starts like "you"). The silent-h words take "an", the "you" words "a".
SILENT_H_NOUNS = (
    "man", "boy", "girl", "teacher", "farmer", "friend", "doctor", "child", "woman",
)  # fmt: skip
SILENT_H_PHRASES = (  # (the phrase, the sentences it goes in)
    ("hour", ("We waited for {x}.", "It took {x}.", "She slept for {x}.", "He ran for {x}.",
              "They played for {x}.", "I read for {x}.")),
    ("honor", ("It is {x}.", "It was {x}.")),
    ("honest mistake", ("It was {x}.", "She made {x}.")),
    ("honest answer", ("He gave {x}.", "That was {x}.")),
)  # fmt: skip
HONEST_FRAMES = (
    "This is {x}.",
    "She has {x}.",
    "There is {x} here.",
    "He was {x}.",
    "We need {x}.",
    "I know {x}.",
)
YOU_SOUND_WORDS = ("unicorn", "university", "uniform")
YOU_SOUND_ADJECTIVES = {  # each with what it can describe: "a used car", "a unique gift"
    "useful": ("tool", "book", "gift", "map", "bag", "box"),
    "used": ("book", "car", "bike", "toy", "bag"),
    "unique": ("gift", "place", "cat", "dog", "game", "house", "toy"),
}
ARTICLE_FRAMES = (  # sentences for any noun
    "I see {x}.",
    "There is {x} here.",
    "We saw {x} today.",
    "This is {x}.",
    "It was {x}.",
    "Mia drew {x}.",
)
OWN_FRAMES = ("She has {x}.", "He found {x}.", "Tom wants {x}.")  # for what a person can have

TRANSITIVE_PAST = (
    "chased", "saw", "found", "liked", "helped", "followed", "heard", "hugged", "pushed",
    "pulled", "called", "watched",
)  # fmt: skip
PLACE_PHRASES = (
    ("table", "on"), ("table", "under"), ("box", "in"), ("bed", "on"), ("bed", "under"),
    ("tree", "under"), ("tree", "near"), ("house", "near"), ("house", "behind"), ("chair", "near"),
    ("chair", "behind"), ("car", "behind"), ("car", "near"), ("bag", "in"), ("door", "behind"),
    ("fence", "behind"), ("fence", "near"), ("lake", "near"), ("park", "in"), ("barn", "in"),
)  # fmt: skip
# Who and what fits where: in a box or a bag only something small, on or under a table, bed or
# chair nothing as big as a horse or a grown-up.
TINY = ("ball", "book", "hat", "toy", "cup", "mouse", "frog", "cat")
SMALL = (
    *TINY, "bag", "dog", "duck", "hen", "bird", "rabbit", "fox", "monkey", "baby", "boy", "girl",
    "child",
)  # fmt: skip
PLACE_SIZES = {"box": TINY, "bag": TINY, "table": SMALL, "bed": SMALL, "chair": SMALL}
LOCATIONS = ("in the park", "at home", "by the lake", "on the farm", "at school")
PLURAL_FRAMES = (("We", "saw"), ("I", "see"), ("They", "found"), ("She", "has"), ("Tom", "drew"))
SINGULAR_QUANTIFIERS = ("one", "this", "that")  # not "I see every car", "Tom drew each man"
PLURAL_QUANTIFIERS = ("two", "three", "four", "five", "six", "many", "these", "those")
HAS_QUANTIFIERS = ("one", "two", "three", "four", "five", "six", "many")  # what "She has" takes
HELPERS = ("can", "will", "must", "should")
OBJECT_FRAMES = (("I", "see"), ("She", "has"), ("He", "likes"), ("We", "saw"), ("Tom", "found"))
ORDER_WORDS = (
    "big", "small", "little", "red", "blue", "green", "new", "old", "funny", "happy", "black",
    "white",
)  # fmt: skip
# What a subject with an adjective does in an adjective-order pair, for an animal and for a thing.
ORDER_PREDICATES = {
    "animal": ("is here", "is over there", "sleeps here", "ran away", "is at home"),
    "thing": ("is here", "is over there", "was here", "was over there"),
}


@dataclass(frozen=True)
class _Pair:
    family: str
    subtype: str
    good: str
    bad: str


def _subjects(who: str, *, plural: bool) -> list[str]:
    """Subject phrases (starting in lower case, except names) that can do a verb of ``who``."""
    nouns = SUBJECT_NOUNS[who]
    if plural:
        return [f"the {PLURALS[n]}" for n in nouns]
    return [f"the {n}" for n in nouns] + list(NAMES)


_NOUN_OF = {form: noun for noun, plural in PLURALS.items() for form in (noun, plural)}


def _says(subject: str, rest: str) -> bool:
    """Whether a subject phrase goes with what follows its verb.

    An animal or a baby never eats lunch, drinks tea or goes to school; a wild animal has nothing
    to do with a bed, a ball, a toy or a name, and no animal bigger than a dog jumps on a bed; and a
    subject is not in its own complement ("The dog sees the dog.").
    """
    noun = _NOUN_OF.get(subject.split()[-1])
    if noun is None:  # a name
        return True
    words = set(rest.split())
    if (noun in ANIMALS or noun == "baby") and NOT_FOR_ANIMALS & words:
        return False
    if noun in WILD_ANIMALS and words & {"bed", "ball", "toy", "name"}:
        return False
    if noun in ANIMALS and "bed" in words and noun not in SMALL:
        return False
    return not {noun, PLURALS[noun]} & words


# --- agreement ------------------------------------------------------------------------------------


def _agreement_verb(plural: bool) -> Iterator[_Pair]:
    subtype = "verb_plural" if plural else "verb_singular"
    for verb in VERBS:
        good, bad = (verb.base, verb.third) if plural else (verb.third, verb.base)
        for subject in _subjects(verb.who, plural=plural):
            for comp in verb.complements:
                if _says(subject, comp):
                    yield _Pair(
                        "agreement",
                        subtype,
                        f"{capitalized(subject)} {good} {comp}.",
                        f"{capitalized(subject)} {bad} {comp}.",
                    )


def _agreement_be(plural: bool) -> Iterator[_Pair]:
    subtype = "be_plural" if plural else "be_singular"
    for one, many in (("is", "are"), ("was", "were")):
        good, bad = (many, one) if plural else (one, many)
        for subject in _subjects("b", plural=plural):
            for rest in (*ADJECTIVES_FOR_BE, *PLACES_FOR_BE):
                if _says(subject, rest):
                    yield _Pair(
                        "agreement",
                        subtype,
                        f"{capitalized(subject)} {good} {rest}.",
                        f"{capitalized(subject)} {bad} {rest}.",
                    )


def _agreement_pronoun(third: bool) -> Iterator[_Pair]:
    subtype = "pronoun_third" if third else "pronoun_base"
    for pronoun in PRONOUN_THIRD if third else PRONOUN_BASE:
        for verb in VERBS:
            if pronoun == "It" and verb.who == "p":
                continue
            good, bad = (verb.third, verb.base) if third else (verb.base, verb.third)
            for comp in verb.complements:
                yield _Pair(
                    "agreement",
                    subtype,
                    f"{pronoun} {good} {comp}.",
                    f"{pronoun} {bad} {comp}.",
                )


def _agreement_have(plural: bool) -> Iterator[_Pair]:
    subtype = "have_plural" if plural else "have_singular"
    good, bad = ("have", "has") if plural else ("has", "have")
    for subject in _subjects("b", plural=plural):
        noun = _NOUN_OF.get(subject.split()[-1])
        if noun in ANIMALS:
            objects = HAVE_FOR_ANIMALS if noun in WILD_ANIMALS else HAVE_FOR_PETS
        else:
            objects = HAVE_FOR_PEOPLE
        for obj in objects:
            yield _Pair(
                "agreement",
                subtype,
                f"{capitalized(subject)} {good} {obj}.",
                f"{capitalized(subject)} {bad} {obj}.",
            )


# --- article --------------------------------------------------------------------------------------


def _article_pairs(subtype: str, an_is_good: bool, phrases: list[str], frames: tuple[str, ...]):
    for phrase in phrases:
        for frame in frames:
            good_article, bad_article = ("an", "a") if an_is_good else ("a", "an")
            yield _Pair(
                "article",
                subtype,
                frame.format(x=f"{good_article} {phrase}"),
                frame.format(x=f"{bad_article} {phrase}"),
            )


def _frames(noun: str) -> tuple[str, ...]:
    """The article sentences a noun can go in: all of them for what a person can have."""
    return ARTICLE_FRAMES + OWN_FRAMES if "own" in NOUN_KINDS[noun] else ARTICLE_FRAMES


def _article_plain(an_is_good: bool) -> Iterator[_Pair]:
    nouns = VOWEL_NOUNS if an_is_good else CONSONANT_NOUNS
    for noun in nouns:
        subtype = "plain_an" if an_is_good else "plain_a"
        yield from _article_pairs(subtype, an_is_good, [noun], _frames(noun))


def _article_adjective(an_is_good: bool) -> Iterator[_Pair]:
    # the adjective decides: "an old dog", "a big apple" (the noun is of the other sound)
    adjectives = VOWEL_ADJECTIVES if an_is_good else CONSONANT_ADJECTIVES
    nouns = CONSONANT_NOUNS if an_is_good else VOWEL_NOUNS
    subtype = "adjective_an" if an_is_good else "adjective_a"
    for adjective in adjectives:
        for noun in nouns:
            if fits(adjective, noun):
                phrase = [f"{adjective} {noun}"]
                yield from _article_pairs(subtype, an_is_good, phrase, _frames(noun))


def _article_sound(an_is_good: bool) -> Iterator[_Pair]:
    if an_is_good:  # a silent h
        phrases = [f"honest {noun}" for noun in SILENT_H_NOUNS]
        yield from _article_pairs("sound_an", True, phrases, HONEST_FRAMES)
        for phrase, frames in SILENT_H_PHRASES:
            yield from _article_pairs("sound_an", True, [phrase], frames)
    else:  # a letter u that sounds like "you"
        phrases = list(YOU_SOUND_WORDS)
        phrases += [
            f"{adj} {noun}" for adj, nouns in YOU_SOUND_ADJECTIVES.items() for noun in nouns
        ]
        yield from _article_pairs("sound_a", False, phrases, ARTICLE_FRAMES + OWN_FRAMES)


# --- word order -----------------------------------------------------------------------------------


def _order_adjective() -> Iterator[_Pair]:
    # In the subject, before the verb: "The big dog is here." / "The dog big is here." After an
    # object an adjective can be good English ("Tom found the box empty"), so it is never there.
    nouns = [n for n, kinds in NOUN_KINDS.items() if "own" in kinds]
    for adjective in ORDER_WORDS:
        for noun in nouns:
            if not fits(adjective, noun):
                continue
            kind = "animal" if "animal" in NOUN_KINDS[noun] else "thing"
            for rest in ORDER_PREDICATES[kind]:
                yield _Pair(
                    "word_order",
                    "adjective",
                    f"The {adjective} {noun} {rest}.",
                    f"The {noun} {adjective} {rest}.",
                )


def _order_object() -> Iterator[_Pair]:
    names = [*PEOPLE, *ANIMALS]
    for first in names:
        for verb in TRANSITIVE_PAST:
            for second in names:
                if first != second:
                    yield _Pair(
                        "word_order",
                        "object",
                        f"The {first} {verb} the {second}.",
                        f"The {first} the {second} {verb}.",
                    )


def _order_preposition() -> Iterator[_Pair]:
    for thing in (*ANIMALS, *PEOPLE, "ball", "book", "hat", "toy", "cup", "bag"):
        for place, prep in PLACE_PHRASES:
            if place != thing and thing in PLACE_SIZES.get(place, (thing,)):
                yield _Pair(
                    "word_order",
                    "preposition",
                    f"The {thing} is {prep} the {place}.",
                    f"The {thing} is the {place} {prep}.",
                )


def _order_determiner() -> Iterator[_Pair]:
    for subject, verb in OBJECT_FRAMES:
        nouns = [n for n, kinds in NOUN_KINDS.items() if "own" in kinds]
        people = () if verb == "has" else tuple(PEOPLE)  # not "She has the teacher at school."
        for noun in dict.fromkeys((*ANIMALS, *people, *nouns)):
            for location in LOCATIONS:
                yield _Pair(
                    "word_order",
                    "determiner",
                    f"{subject} {verb} the {noun} {location}.",
                    f"{subject} {verb} {noun} the {location}.",
                )


def _order_helper() -> Iterator[_Pair]:
    for verb in VERBS:
        for helper in HELPERS:
            for subject in [*_subjects(verb.who, plural=False), *_subjects(verb.who, plural=True)]:
                for comp in verb.complements[:2]:
                    if _says(subject, comp):
                        yield _Pair(
                            "word_order",
                            "helper",
                            f"{capitalized(subject)} {helper} {verb.base} {comp}.",
                            f"{capitalized(subject)} {verb.base} {helper} {comp}.",
                        )


# --- tense ----------------------------------------------------------------------------------------

# subtype -> (how the sentence is built, good verb form, bad verb form, the verbs it can use)
TENSE_KINDS = {
    "past_base": ("past", "past", "base", VERBS),
    "past_participle": ("past", "past", "participle", PARTICIPLE_VERBS),
    "future_past": ("future", "base", "past", VERBS),
    "future_participle": ("future", "base", "participle", PARTICIPLE_VERBS),
    "did_past": ("did", "base", "past", VERBS),
    "did_participle": ("did", "base", "participle", PARTICIPLE_VERBS),
    "perfect": ("perfect", "participle", "past", PERFECT_VERBS),
    "perfect_base": ("perfect", "participle", "base", PARTICIPLE_VERBS),  # "has eat"
}


def _times(times: tuple[str, ...], verb: _Verb) -> tuple[str, ...]:
    """The time phrases a verb's sentence can start with: long ones only for a slow change."""
    if verb.base in SLOW_VERBS:
        return tuple(t for t in times if t.endswith(("year", "summer")))
    return times


def _tense(subtype: str) -> Iterator[_Pair]:
    frame, good_form, bad_form, verbs = TENSE_KINDS[subtype]
    for verb in verbs:
        good, bad = getattr(verb, good_form), getattr(verb, bad_form)
        for plural in (False, True):
            for subject in _subjects(verb.who, plural=plural):
                for comp in verb.complements:
                    if not _says(subject, comp):
                        continue
                    if frame in ("past", "future") and comp in HABITS:
                        continue  # not "Yesterday Tom walked every day."
                    match frame:
                        case "past":
                            times = _times(PAST_TIMES, verb)
                            texts = [f"{t} {subject} {{}} {comp}." for t in times]
                        case "future":
                            times = _times(FUTURE_TIMES, verb)
                            texts = [f"{t} {subject} will {{}} {comp}." for t in times]
                        case "did":
                            texts = [f"Did {subject} {{}} {comp}?"]
                        case _:
                            helper = "have" if plural else "has"
                            texts = [f"{capitalized(subject)} {helper} {{}} {comp}."]
                    for text in texts:
                        yield _Pair("tense", subtype, text.format(good), text.format(bad))


# --- plural ---------------------------------------------------------------------------------------


def _plural_pairs(plural_is_good: bool) -> Iterator[_Pair]:
    every = PLURAL_QUANTIFIERS if plural_is_good else SINGULAR_QUANTIFIERS
    for subject, verb in PLURAL_FRAMES:
        nouns, quantifiers = PLURALS, every
        if verb == "has":  # "She has two cats", never "She has these men" or "every duck"
            nouns = {**ANIMALS, **THINGS}
            quantifiers = tuple(q for q in every if q in HAS_QUANTIFIERS)
        for quantifier in quantifiers:
            for noun, plural in nouns.items():
                good, bad = (plural, noun) if plural_is_good else (noun, plural)
                for location in LOCATIONS:
                    yield _Pair(
                        "plural",
                        "plural" if plural_is_good else "singular",
                        f"{subject} {verb} {quantifier} {good} {location}.",
                        f"{subject} {verb} {quantifier} {bad} {location}.",
                    )


def _generators() -> Iterator[Iterator[_Pair]]:
    for plural in (False, True):
        yield _agreement_verb(plural)
        yield _agreement_be(plural)
        yield _agreement_have(plural)
    for third in (True, False):
        yield _agreement_pronoun(third)
    for an_is_good in (True, False):
        yield _article_plain(an_is_good)
        yield _article_adjective(an_is_good)
        yield _article_sound(an_is_good)
    yield _order_adjective()
    yield _order_object()
    yield _order_preposition()
    yield _order_determiner()
    yield _order_helper()
    for subtype in TENSE_KINDS:
        yield _tense(subtype)
    yield _plural_pairs(plural_is_good=True)
    yield _plural_pairs(plural_is_good=False)


def _pairs() -> Iterator[_Pair]:
    """Every pair of every subtype, in a fixed order (about 150,000).

    A good sentence occurs once in a subtype; two subtypes may share one (the past sentence
    "Yesterday Tom ate lunch." has a plain-form bad one and a participle bad one).
    """
    for generator in _generators():  # each subtype comes from one generator
        seen: set[str] = set()
        for pair in generator:
            if pair.good != pair.bad and pair.good not in seen:
                seen.add(pair.good)
                yield pair


@cache
def _bench_pools() -> dict[tuple[str, str], tuple[_Pair, ...]]:
    """The pairs the benchmark may use, by (family, subtype): those with a reserved good one.

    A good sentence two families share ("The dog is in the park." is an agreement and a word-order
    one) is in the first family's pools only, so a family can always use all of its own.
    """
    pools: dict[tuple[str, str], list[_Pair]] = {}
    owner: dict[str, str] = {}
    for pair in _pairs():
        if (
            reserved_for_bench(pair.good)
            and owner.setdefault(pair.good, pair.family) == pair.family
        ):
            pools.setdefault((pair.family, pair.subtype), []).append(pair)
    return {key: tuple(pairs) for key, pairs in pools.items()}


def _allocate(total: int, weights: dict[str, float], capacity: dict[str, int]) -> dict[str, int]:
    """Split ``total`` over the keys in proportion to ``weights``, never over a key's capacity.

    Largest remainders first; what a full key cannot take goes to the others. ``ValueError`` if
    ``total`` is more than the capacities add up to.
    """
    if total > sum(capacity.values()):
        raise ValueError(f"cannot take {total} from {sum(capacity.values())} available")
    share = {key: 0 for key in weights}
    left = total
    while left > 0:
        open_keys = [k for k in weights if share[k] < capacity[k]]
        scale = sum(weights[k] for k in open_keys)
        ideal = {k: left * weights[k] / scale for k in open_keys}
        grant = {k: min(int(ideal[k]), capacity[k] - share[k]) for k in open_keys}
        if not any(grant.values()):  # fewer left than keys: the largest remainder takes one
            best = max(open_keys, key=lambda k: (ideal[k], weights[k]))
            grant[best] = 1
        for key, amount in grant.items():
            share[key] += amount
            left -= amount
    return share


def grammar_pairs(rng: np.random.Generator, n: int = 200) -> list[PairItem]:
    """``n`` minimal pairs (category ``"language"``, ``language-0000``...), tagged ``fam:<family>``.

    The five families share the pairs evenly and take turns, so any prefix is as balanced as it
    can be; within a family the subtypes (see ``SUBTYPE_WEIGHTS``) share them in fixed proportions
    that keep the mistake's direction balanced. All good sentences are different, and each is
    reserved for the benchmark. ``ValueError`` if ``n`` is more than there are.
    """
    pools = _bench_pools()
    capacity = {
        family: sum(len(pools[(family, sub)]) for sub in SUBTYPE_WEIGHTS[family])
        for family in FAMILIES
    }
    quota = fair_quota(capacity, n)
    per_family: list[list[_Pair]] = []
    used: set[str] = set()
    for family in FAMILIES:
        weights = SUBTYPE_WEIGHTS[family]
        counts = _allocate(
            quota[family], weights, {sub: len(pools[(family, sub)]) for sub in weights}
        )
        chosen: list[_Pair] = []
        for sub, count in counts.items():
            pool, taken = pools[(family, sub)], 0
            for i in rng.permutation(len(pool)):
                if taken == count:
                    break
                if pool[int(i)].good not in used:  # subtypes can share a good sentence
                    used.add(pool[int(i)].good)
                    chosen.append(pool[int(i)])
                    taken += 1
            if taken < count:
                raise RuntimeError(f"grammar: could not draw enough {family} {sub} pairs")
        per_family.append([chosen[int(i)] for i in rng.permutation(len(chosen))])
    ordered = in_turn(per_family)
    return [
        PairItem(f"language-{i:04d}", "language", p.good, p.bad, (f"fam:{p.family}",))
        for i, p in enumerate(ordered)
    ]
