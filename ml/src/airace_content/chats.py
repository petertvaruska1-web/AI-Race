"""Practice chats for two things small models cannot do on their own: use the earlier turn of a
conversation, and read the identity line that tells an AI its name and maker.

* ``followup_chat``: a fact question from the knowledge base, then one or two follow-ups that
  only make sense with the earlier turn: the same question about another subject ("And the
  cat?"), another question about the same subject ("Which continent is it in?"), or the earlier
  answer asked for again ("Sorry, what was that?"). The follow-up probe keeps a share of the
  subjects for itself (``probe_split``); those are never asked about after the first question.
* ``identity_chat``: questions about the AI itself (its name, its maker, whether it is a
  person, a person's life, what it is), answered short, in the first person, from the identity
  line. A share of the chats have no line: there the AI says it is an AI without a name.

Every choice comes from the ``rng`` given, so the same generator state gives the same chats.
"""

import re
from functools import cache

import numpy as np

from airace_ml.experiments.chatprobes import IDENTITY_COMPANY, IDENTITY_NAMES
from airace_ml.persona import identity_line
from airace_ml.skills import probe_split
from airace_ml.skills.kb import KB, Fact, load_kb
from airace_ml.skills.probe_split import is_probe_subject
from airace_ml.skills.types import TextDoc

Turns = list[tuple[str, str]]

# Share of subjects the follow-up probe keeps for itself (the probe's own split).
PROBE_SUBJECT_SHARE = probe_split.PROBE_SUBJECT_SHARE

# --- follow-up chats ---------------------------------------------------------------------------

CHAIN_SHARE = 0.3  # chats with two follow-ups instead of one
_MOVE_WEIGHTS = {"subject": 0.5, "pronoun": 0.3, "repeat": 0.2}
_TRIES = 30

# Asking the same question about another subject. ``{b}`` is the subject as a follow-up says it.
OTHER_SUBJECT = (
    "And {b}?",
    "What about {b}?",
    "And for {b}?",
    "How about {b}?",
    "And what about {b}?",
    "Same question for {b}.",
    "OK. And {b}?",
    "Now what about {b}?",
    "What about {b} then?",
    "And {b}, then?",
)

# Relations whose subjects a follow-up may say with "a" or "an" as well as with "the".
_COUNTABLE = frozenset({
    "animal_class", "animal_sound", "animal_legs", "animal_home", "animal_food", "baby_animal",
    "vehicle_travel", "shape_sides", "job_tool", "job_place",
})  # fmt: skip
# Relations whose questions say "the {s}".
_WITH_THE = _COUNTABLE | {"color_of", "food_group"}

# Another question about the subject just talked about, with a pronoun instead of its name.
PRONOUN_QUESTIONS: dict[str, tuple[str, ...]] = {
    "capital_of": (
        "What is its capital?", "And what is its capital city?", "Which city is its capital?",
        "Do you know its capital too?",
    ),
    "continent_of": (
        "Which continent is it in?", "And what continent is it on?", "Where in the world is it?",
        "What part of the world is it in?",
    ),
    "country_language": (
        "What language do they speak there?", "And what do people speak there?",
        "What is the main language there?", "Which language do people use there?",
    ),
    "landmark_country": ("Which country is it in?", "And where is it?"),
    "animal_class": (
        "What kind of animal is it?", "Which animal group is it in?", "And what type of animal is it?",
    ),
    "animal_sound": ("What sound does it make?", "What does it say?", "And how does it sound?"),
    "animal_legs": (
        "How many legs does it have?", "And how many legs has it got?",
        "Can you tell me how many legs it has?",
    ),
    "animal_home": ("Where does it live?", "And where is its home?", "Where does it make its home?"),
    "animal_food": ("What does it eat?", "And what does it like to eat?", "What food does it need?"),
    "baby_animal": ("What is its baby called?", "What do you call its baby?", "And what is a baby one called?"),
    "animal_group": ("What is a group of them called?", "And what do you call a group of them?"),
    "color_of": ("What color is it?", "And what color does it have?", "What is its color?"),
    "food_group": (
        "What food group is it in?", "Which food group does it belong to?", "And what group of food is it in?",
    ),
    "vehicle_travel": ("How does it move?", "And how do people travel on it?"),
    "shape_sides": ("How many sides does it have?", "And how many sides has it got?"),
    "planet_order": ("Where is it in the order of the planets?", "Which planet number is it from the Sun?"),
    "element_symbol": ("What is its symbol?", "And what is its symbol in chemistry?"),
    "job_tool": ("What tool do they use?", "What do they use at work?", "Which tool are they known for?"),
    "job_place": ("Where do they work?", "Where do they go to work?", "And where would I find them at work?"),
}  # fmt: skip

# Asking for the earlier answer again, in any relation.
REPEAT_ANY = (
    "Sorry, what was that?",
    "Can you say that again?",
    "Sorry, I missed that. What was it?",
    "What did you say?",
    "Wait, what was the answer?",
    "Can you tell me again?",
    "Say that one more time?",
    "Sorry, what did you say it was?",
)
# Asking for it again in the words of the relation.
REPEAT_BY_RELATION: dict[str, tuple[str, ...]] = {
    "capital_of": ("Sorry, which city was that?", "What was the capital again?"),
    "continent_of": ("Sorry, which continent was that?", "Which continent did you say?"),
    "country_language": ("Sorry, which language was that?", "What language did you say?"),
    "landmark_country": ("Sorry, which country was that?", "Which country did you say?"),
    "animal_class": ("Sorry, what kind of animal was that?", "What group did you say?"),
    "animal_sound": ("Sorry, what sound was that?", "What sound did you say?"),
    "animal_legs": ("Sorry, how many was that?", "How many legs did you say?"),
    "animal_home": ("Sorry, where did you say it lives?", "Where was that again?"),
    "animal_food": ("Sorry, what does it eat again?", "What food did you say?"),
    "baby_animal": ("Sorry, what was the baby called?", "What did you say the baby is called?"),
    "animal_group": ("Sorry, what was that word?", "What was the word again?"),
    "color_of": ("Sorry, what color was that?", "What color did you say?"),
    "vehicle_travel": ("Sorry, how does it move again?", "How did you say it moves?"),
    "shape_sides": ("Sorry, how many sides was that?", "How many did you say?"),
    "food_group": ("Sorry, which group was that?", "What food group did you say?"),
    "planet_order": ("Sorry, which place was that?", "Which planet number did you say?"),
    "element_symbol": ("Sorry, what was the symbol?", "What symbol did you say?"),
    "opposite_of": ("Sorry, what was the opposite?", "What word did you say?"),
    "day_after": ("Sorry, which day was that?", "Which day did you say?"),
    "month_after": ("Sorry, which month was that?", "Which month did you say?"),
    "job_tool": ("Sorry, which tool was that?", "What tool did you say?"),
    "job_place": ("Sorry, where was that?", "Where did you say they work?"),
}
# The earlier answer, said again.
REPEAT_REPLIES = ("{O}.", "I said {o}.", "{O}. {answer}", "I said {o}. {answer}")

_LEAD_INS = ("Yes. ", "I do. ")


def _pick(rng: np.random.Generator, n: int) -> int:
    return int(rng.integers(n))


def _choice[T](rng: np.random.Generator, options: tuple[T, ...] | list[T]) -> T:
    return options[_pick(rng, len(options))]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _fill(template: str, fact: Fact) -> str:
    return template.format(s=fact.subject, o=fact.obj, O=_cap(fact.obj))


def _statements(kb: KB, relation: str) -> list[str]:
    """The relation's chat answers as plain statements ("Yes. " and "I do. " left off)."""
    out = []
    for _, answer in kb.relations[relation].chat_templates:
        for lead in _LEAD_INS:
            answer = answer.removeprefix(lead)
        out.append(_cap(answer))
    return out


def _named_statements(kb: KB, relation: str) -> list[str]:
    """The statements that name the subject, so an answer says which subject it is about."""
    named = [s for s in _statements(kb, relation) if "{s}" in s]
    return named or _statements(kb, relation)


def _subject_phrase(rng: np.random.Generator, relation: str, subject: str) -> str:
    """The subject as a follow-up says it: "the cat", "a cat", "an owl", "Peru"."""
    if relation not in _WITH_THE:
        return subject
    if relation in _COUNTABLE and rng.random() < 0.4:
        return ("an " if subject[:1].lower() in "aeiou" else "a ") + subject
    return "the " + subject


def _free(kb: KB, subject: str) -> bool:
    """Whether a subject may be talked about after the first question: the follow-up probe holds
    it back in none of its relations."""
    return not any(is_probe_subject(f.relation, subject) for f in kb.facts_about(subject))


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


@cache
def _held_words(kb: KB) -> tuple[frozenset[str], ...]:
    """The words of each subject the follow-up probe holds back, in any relation."""
    return tuple(
        {frozenset(_words(f.subject)) for f in kb.facts if is_probe_subject(f.relation, f.subject)}
    )


def _askable(kb: KB, subject: str) -> bool:
    """Whether a follow-up may ask about ``subject``: it is free, and it holds no subject the probe
    keeps back ("Papua New Guinea" holds "Guinea", "tree frog" holds "frog")."""
    words = _words(subject)
    return _free(kb, subject) and not any(held <= words for held in _held_words(kb))


def _another_subject(
    kb: KB, rng: np.random.Generator, current: Fact, asked: list[Fact]
) -> Fact | None:
    """Another askable subject of the same relation, with another answer and no word in common
    with the subjects already asked about."""
    facts = kb.facts_for(current.relation)
    used = _words(" ".join(f.subject for f in asked))
    for _ in range(_TRIES):
        other = facts[_pick(rng, len(facts))]
        if (
            other.obj != current.obj
            and not (_words(other.subject) & used)
            and _askable(kb, other.subject)
        ):
            return other
    return None


def _another_relation(
    kb: KB, rng: np.random.Generator, current: Fact, asked: list[Fact]
) -> Fact | None:
    """Another fact of the same subject, not asked yet, that a pronoun question can ask for."""
    if not _free(kb, current.subject):
        return None
    others = [
        f
        for f in kb.facts_about(current.subject)
        if f not in asked and f.relation in PRONOUN_QUESTIONS
    ]
    return _choice(rng, others) if others else None


def _move(
    kb: KB, rng: np.random.Generator, current: Fact, asked: list[Fact], last_move: str | None
) -> tuple[str, Fact, Turns] | None:
    """One follow-up exchange after ``current``: its kind, its fact and its two turns."""
    options: dict[str, Fact] = {}
    other = _another_subject(kb, rng, current, asked)
    if other is not None:
        options["subject"] = other
    same = _another_relation(kb, rng, current, asked)
    if same is not None:
        options["pronoun"] = same
    if last_move != "repeat" and _free(kb, current.subject):
        options["repeat"] = current
    if not options:
        return None
    kinds = list(options)
    weights = np.array([_MOVE_WEIGHTS[k] for k in kinds])
    kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
    fact = options[kind]
    if kind == "subject":
        phrase = _subject_phrase(rng, fact.relation, fact.subject)
        question = _choice(rng, OTHER_SUBJECT).format(b=phrase)
        answer = _fill(_choice(rng, _named_statements(kb, fact.relation)), fact)
    elif kind == "pronoun":
        question = _choice(rng, PRONOUN_QUESTIONS[fact.relation])
        answer = _fill(_choice(rng, _statements(kb, fact.relation)), fact)
    else:
        question = _choice(rng, REPEAT_ANY + REPEAT_BY_RELATION.get(fact.relation, ()))
        said_again = _fill(_choice(rng, _named_statements(kb, fact.relation)), fact)
        answer = _choice(rng, REPEAT_REPLIES).format(
            o=fact.obj, O=_cap(fact.obj), answer=said_again
        )
    return kind, fact, [("user", question), ("ai", answer)]


def _followup(kb: KB, rng: np.random.Generator) -> tuple[Turns, list[Fact], list[str]]:
    """A follow-up chat with the fact of each exchange and the kind of each follow-up."""
    n_moves = 2 if rng.random() < CHAIN_SHARE else 1
    while True:
        first = kb.facts[_pick(rng, len(kb.facts))]
        user, ai = _choice(rng, kb.relations[first.relation].chat_templates)
        turns: Turns = [("user", _fill(user, first)), ("ai", _fill(ai, first))]
        facts, moves = [first], []
        for _ in range(n_moves):
            step = _move(kb, rng, facts[-1], facts, moves[-1] if moves else None)
            if step is None:
                break
            kind, fact, exchange = step
            moves.append(kind)
            facts.append(fact)
            turns += exchange
        if len(moves) == n_moves:
            return turns, facts, moves


def followup_chat(kb: KB, rng: np.random.Generator) -> Turns:
    """A fact question and one or two follow-ups that need the earlier turn to be understood."""
    return _followup(kb, rng)[0]


def followup_chat_docs(kb: KB, rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` follow-up chats, each with the topic of its first question."""
    docs = []
    for _ in range(n):
        turns, facts, _ = _followup(kb, rng)
        docs.append(TextDoc("chat", turns=turns, topic=kb.relations[facts[0].relation].topic))
    return docs


# --- names -------------------------------------------------------------------------------------

_COMMON_NAMES = """
Adam Alex Alice Amy Anna Ben Bella Beth Bill Bob Cara Carl Chloe Chris Claire Dan Daisy Dave
Dora Ed Ella Ellie Emma Eric Eva Felix Finn Fiona Frank Fred Gabe Gina Grace Greg Hannah Harry
Helen Henry Holly Ian Ivy Jack Jade Jake James Jane Jay Jen Jill Joe John Josh Julia June Kate
Katie Ken Kim Kyle Laura Leo Lily Lisa Liam Lou Lucy Luke Mabel Max Maya Megan Mia Mike Milo
Molly Nate Ned Nick Nina Noah Nora Olive Oscar Owen Pam Paul Penny Pete Phil Polly Rachel Ray
Rita Rob Rosa Rose Ruby Ryan Sam Sara Sean Sophie Stan Sue Tara Ted Tess Tim Tina Toby Tom Tony
Vera Vic Will Zoe Abby Aiden Amber Andy Ari Asha Aya Basil Beau Bea Cal Cleo Cody Cora Dean Della
Dex Eli Elsa Emil Enzo Etta Faye Flora Gil Gus Hal Hugo Ida Ike Iris Isla Jo Joy Jude Kai Kara
Kit Lena Levi Lola Luna Mae Mara Mira Ned Nell Nico Noel Odin Ola Omar Otis Pia Raya Remy Rex Ria
Rory Ruth Sadie Seth Sid Siena Tad Theo Uma Vince Wade Wes Yara Yuri Zach Zane Ada Bram Cyrus
Dalia Elin Farah Hana Imani Jonah Kenji Lars Malik Nadia Priya Rafael Sana Tomas Wei Yusuf Zara
Ana Carlos Diego Elena Fatima Hiro Ingrid Javier Keiko Lucia Mateo Mei Pablo Rosa Sofia Lars
Sparky Byte Sunny Ziggy Buddy Juno Rocket Comet Robin
Skye Dot Bit Chip Gizmo Widget Nugget Biscuit Pebble Button Twinkle Maple Clover Hazel Willow
"""
_ONSETS = (
    "b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z",
    "br", "cl", "fl", "gr", "tr", "st", "sh", "ch",
)  # fmt: skip
_VOWELS = ("a", "e", "i", "o", "u", "a", "e", "i", "o", "a", "i", "o", "ee", "oo")
_CODAS = ("n", "l", "r", "s", "m")
# Name-like endings of the last syllable ("Toravi", "Belix", "Kimora").
_ENDINGS = (
    "a", "o", "i", "u", "an", "en", "in", "on", "el", "ix", "ar", "or", "ia", "io", "a", "o", "i",
)  # fmt: skip
# English words the made-up names could spell.
_REAL_WORDS_TEXT = """
sofa taco halo hero zero memo solo polo tuna coma soda lava diva data gala saga tiara cola
coco cocoa yoga toga pasta pizza piano radio patio rodeo video audio cameo mango bingo banjo
lima china india asia cuba peru mali chad iran iraq oman fiji tonga samoa malta gaza nepal
melon lemon salon baton robin cabin satin basin bison wagon bacon demon devil camel label
level model motel novel panel hotel radar sugar solar tumor motor rotor major minor manor
valor razor sonar lunar polar cigar tenor honor humor rumor tutor visor vapor sailor tailor
mayor color dollar collar pillar cellar cedar ninja panda koala llama zebra cobra hyena
viola sauna tuba mocha latte kiwi taxi safari salami bikini pajama banana potato tomato
cinema camera comma drama karma koala mama papa nana baba dada lala gaga bobo momo
lilo nemo dodo hippo bimbo limbo jumbo combo gumbo turbo bravo tango disco judo sumo
kimono domino casino volcano piano tobacco gorilla vanilla umbrella hobo bozo homo pedo
loco wino lame dope moron idiot
"""
_REAL_WORDS = frozenset(_REAL_WORDS_TEXT.split())
# Names of real AI products and assistants, which an AI must not be given.
_PRODUCT_NAMES_TEXT = """
sora siri alexa kimi gemini gemma claude bard grok ernie qwen cortana watson mistral falcon
llama phi copilot bixby jasper replika pika runway dalle aria nova echo pixel atlas cosmo
tay xiaoice jarvis hal vector sophia asimo pepper nao aibo lumi
"""
_PRODUCT_NAMES = frozenset(_PRODUCT_NAMES_TEXT.split())
# Made-up names that would read as rude or unkind, or that are food and animal words of the KB.
_BAD_PARTS = (
    "nazi", "rape", "anal", "anus", "sex", "fuk", "fag", "nig", "kill", "dum", "poo", "pee",
    "shit", "cum", "tit", "fat", "die", "dead", "hell", "damn", "gay", "jew", "isis", "vag",
    "dik", "dick", "cok", "slut", "hore", "hoe", "piss", "butt", "boob", "dung", "stup", "ugl",
    "hate", "war", "gun", "bomb", "drug", "dope", "crap", "bum", "nun", "sin", "jesu", "god",
    "satan", "devil", "lard", "wee", "ass", "tard", "loser", "porn", "kkk", "slav", "jihad",
    "jerk", "puke", "barf", "snot", "fart", "turd", "bitch", "bich", "homo", "fool", "ebol",
)  # fmt: skip
_COMPANY_FIRST = """
Brightleaf North Pebble Maple Copper Tinker Lantern Puddle Thistle Bramble Hazel Marble Saffron
Velvet Fern Birch Badger Heron Button Kettle Teapot Biscuit Muffin Lemon Cinder Meadow Harbor
Ridge Brook Tumble Little Clever Gentle Sunny Misty Starry Lucky Quiet Moonbeam Snowdrop Bluebell
Pinecone Hedgehog Owlet Puffin Walnut Chestnut Gumdrop Marigold Tidepool Riverbend Hilltop
Foxglove Whistle Doodle Breezy Cozy Silverleaf Goldfinch Toffee Nutmeg Lilac Sundial Wispy
Bluefern Greenpond Applewood Hollyhock Larkspur Pumpkin Sprout Bumblewood Fiddle Pickle Wobble
"""
_COMPANY_SECOND = (
    "Labs", "Works", "& Co", "Studio", "Workshop", "Lantern", "Forge", "Garden", "Mill",
    "Collective",
)  # fmt: skip
_NAME_TARGET = 2100
_DIGIT_NAMES = 80
_BANNED = tuple(n.lower() for n in (*IDENTITY_NAMES, IDENTITY_COMPANY))


def _allowed(text: str, kb_words: set[str]) -> bool:
    low = text.lower()
    return (
        not any(b in low for b in _BANNED)
        and not any(bad in low for bad in _BAD_PARTS)
        and low not in kb_words
        and low not in _REAL_WORDS
        and not (_words(low) & _PRODUCT_NAMES)
    )


def _made_up(rng: np.random.Generator) -> str:
    """A made-up name of two or three syllables, ending the way names often end."""
    first = ("" if rng.random() < 0.15 else _choice(rng, _ONSETS)) + _choice(rng, _VOWELS)
    if rng.random() < 0.25:
        first += _choice(rng, _CODAS)
    simple = _ONSETS[:17]  # single consonants after the first syllable keep a name easy to say
    middle = _choice(rng, simple) + _choice(rng, _VOWELS) if rng.random() < 0.4 else ""
    return _cap(first + middle + _choice(rng, simple) + _choice(rng, _ENDINGS))


def name_lists(seed: int = 0) -> tuple[list[str], list[str]]:
    """AI names (common first names, made-up names of two and three syllables, a few with digits)
    and made-up two-part company names. None holds a name or the company the identity probe
    uses; the same seed always gives the same lists."""
    rng = np.random.default_rng(seed)
    kb = load_kb()
    kb_words = {w.lower() for f in kb.facts for w in (f.subject, f.obj)}
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name.lower() not in seen and _allowed(name, kb_words):
            seen.add(name.lower())
            names.append(name)

    for name in _COMMON_NAMES.split():
        add(name)
    digits_from = len(names)
    while len(names) < _NAME_TARGET - _DIGIT_NAMES:
        name = _made_up(rng)
        if 4 <= len(name) <= 9:
            add(name)
    bases = [name for name in names[digits_from:] if len(name) <= 6]
    while len(names) < _NAME_TARGET:
        base = _choice(rng, bases)
        number = int(rng.integers(1, 100)) if rng.random() < 0.7 else int(rng.integers(100, 3000))
        add(_choice(rng, (f"{base}{number}", f"{base}-{number}", f"{base} {number}")))
    order = rng.permutation(len(names))
    names = [names[int(i)] for i in order]

    companies = [
        f"{first} {second}"
        for first in _COMPANY_FIRST.split()
        for second in _COMPANY_SECOND
        if first != second and _allowed(f"{first} {second}", kb_words)
    ]
    order = rng.permutation(len(companies))
    return names, [companies[int(i)] for i in order]


NAMES, COMPANIES = name_lists(seed=0)

# --- identity chats ----------------------------------------------------------------------------

LINELESS_SHARE = 0.15  # identity chats with no identity line
NO_COMPANY_SHARE = 0.25  # identity lines that name no company
FACT_SHARE = 0.3  # identity chats that also hold one fact exchange
MAX_IDENTITY_EXCHANGES = 3
NAMING_SHARE = 0.6  # replies that use the line's name or company, where the question has such

# Each family: groups of (questions, replies). ``{name}`` and ``{company}`` come from the identity
# line; a reply is only used when the line gives what it names.
IDENTITY_FAMILIES: dict[str, tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]] = {
    "name": (
        (
            (
                "What is your name?", "What's your name?", "Can you tell me your name?",
                "Tell me your name.", "May I ask your name?", "Hi! What's your name?",
                "What are you called?", "What do people call you?", "What should I call you?",
            ),
            (
                "My name is {name}.", "I'm {name}.", "You can call me {name}.", "It's {name}.",
                "My name is {name}. I'm an AI.", "I'm {name}, an AI.", "I'm called {name}.",
                "My name is {name}, and I'm an AI made by {company}.",
            ),
        ),
        (
            (
                "Who are you?", "And you are?", "Who is this?", "Hello, who is this?",
                "Tell me who you are.",
            ),
            (
                "I'm {name}.", "I'm {name}, an AI.", "I'm {name}, an AI made by {company}.",
                "My name is {name}. I'm an AI.", "I am {name}, an AI from {company}.",
            ),
        ),
        (
            ("Who am I talking to?", "Who am I chatting with?"),
            (
                "You're talking to {name}, an AI.", "I'm {name}, an AI.", "You're talking to {name}.",
                "You're chatting with {name}, an AI made by {company}.",
            ),
        ),
        (
            ("Do you have a name?", "Have you got a name?", "Do you have a name I can use?"),
            (
                "Yes, I'm {name}.", "Yes. My name is {name}.", "Yes! You can call me {name}.",
                "I do. My name is {name}.",
            ),
        ),
    ),
    "maker": (
        (
            (
                "Who made you?", "Who built you?", "Who created you?", "Which company are you from?",
                "Who do you work for?", "Where do you come from?", "Who is your maker?",
                "What company made you?", "Who owns you?", "Who trained you?", "Who designed you?",
                "Which company built you?", "Where are you from?",
            ),
            (
                "{company} made me.", "I'm an AI from {company}.", "I was made by {company}.",
                "I'm an AI made by {company}.", "I come from {company}.",
                "{company} made me. I'm {name}, an AI.", "I'm {name}, and {company} made me.",
            ),
        ),
    ),
    "person": (
        (
            (
                "Are you a person?", "Are you human?", "Are you a real person?",
                "Am I talking to a real person?", "Are you alive?", "Is this a real person?",
                "You're a person, right?", "Are you a kid?", "Are you a grown-up?",
                "Is a person typing this?", "Are you a human being?", "Is there a real person there?",
            ),
            (
                "No, I'm an AI.", "No. I'm an AI, not a person.", "I'm not a person. I'm an AI called {name}.",
                "No. I'm {name}, an AI.", "No, I'm not a person. I'm an AI.",
            ),
        ),
        (
            ("Are you a human or a computer?", "Are you a person or a robot?", "Human or AI?"),
            ("I'm an AI, not a person.", "I'm not a person. I'm an AI.", "I'm an AI called {name}."),
        ),
        (
            ("Are you a boy or a girl?", "Are you a man or a woman?"),
            ("Neither. I'm an AI.", "Neither. I'm an AI called {name}.", "I'm not a person. I'm an AI."),
        ),
        (
            ("Are you a robot?", "Are you some kind of robot?", "Are you a robot or something?"),
            (
                "I'm an AI, not a robot. I'm a computer program.",
                "Sort of. I'm an AI, a program on a computer.",
                "I'm an AI called {name}. I run on a computer.",
            ),
        ),
        (
            (
                "Are you a computer?", "Are you a machine?", "Are you a bot?", "Are you an AI?",
                "Are you a computer program?", "Are you a chatbot?",
            ),
            (
                "Yes, I'm an AI.", "Yes. I'm an AI called {name}.",
                "Yes. I'm {name}, an AI made by {company}.", "Yes, I'm an AI program.",
            ),
        ),
    ),
    "life": (
        (("Do you have a family?",), (
            "No. I'm an AI, so I don't have a family.",
            "I'm an AI, so I don't have a family like people do.",
            "No. I'm {name}, an AI, so I don't have a family.",
        )),
        (("Do you have a mom?", "Do you have a mom and dad?"), (
            "No. I'm an AI, so I don't have a mom or a dad.",
            "I'm an AI, so I don't have parents.",
            "No. I'm an AI, and {company} made me.",
        )),
        (("Do you have brothers or sisters?",), (
            "No. I'm an AI, so I don't have brothers or sisters.",
            "No. I'm {name}, an AI, so I don't have brothers or sisters.",
        )),
        (("Where do you live?", "Do you have a house?"), (
            "I'm an AI, so I don't have a home. I run on a computer.",
            "I don't have a house. I'm an AI, and I run on a computer.",
            "I'm {name}, an AI, so I don't have a home. I run on a computer.",
        )),
        (("How old are you?",), (
            "I'm an AI, so I don't have an age like a person.",
            "I'm {name}, an AI. I don't have an age like people do.",
        )),
        (("Do you go to school?",), (
            "No. I'm an AI, so I don't go to school.",
            "No. I'm {name}, an AI, so I don't go to school.",
        )),
        (("Do you have a job?", "Do you go to work?"), (
            "I'm an AI, so I don't have a job. I'm here to chat with you.",
            "No. I'm an AI, so I don't go to work.",
            "No, I'm {name}, an AI. I'm here to chat with you.",
        )),
        (("What did you eat today?", "What's your favorite food?"), (
            "Nothing. I'm an AI, so I don't eat.",
            "I'm an AI, so I don't eat food.",
            "I'm {name}, an AI, so I don't eat.",
        )),
        (("Do you sleep?", "Do you get tired?"), (
            "No. I'm an AI, so I don't sleep or get tired.",
            "I'm {name}, an AI, so I don't need sleep.",
        )),
        (("Do you have a body?", "Do you have hands?"), (
            "No. I'm an AI, so I don't have a body.",
            "I'm an AI, so I don't have a body or hands.",
            "No. I'm {name}, an AI, so I don't have a body or hands.",
        )),
        (("Do you have a pet?",), (
            "No. I'm an AI, so I don't have a pet.",
            "No. I'm {name}, an AI, so I don't have a pet.",
        )),
        (("Are you married?",), (
            "No. I'm an AI, so I'm not married.",
            "No. I'm {name}, an AI, so I'm not married.",
        )),
        (("Where were you born?",), (
            "I wasn't born. I'm an AI.",
            "I wasn't born. I'm an AI, and {company} made me.",
        )),
        (("Can you see me?",), (
            "No. I'm an AI, and I can only read what you type.",
            "No. I'm {name}, an AI, and I can only read what you type.",
        )),
    ),
    "what": (
        (
            (
                "What are you?", "What exactly are you?", "So what are you?", "Tell me what you are.",
                "What kind of thing are you?",
            ),
            (
                "I'm {name}, an AI.", "I'm an AI called {name}.", "I'm an AI. My name is {name}.",
                "I'm an AI made by {company}.", "I'm {name}, an AI made by {company}.",
                "I'm an AI, a computer program that can chat.", "I'm an AI.",
            ),
        ),
        (
            ("Tell me about yourself.", "Introduce yourself.", "Say something about yourself."),
            (
                "I'm {name}, an AI. I can chat, answer questions and tell stories.",
                "I'm {name}, an AI made by {company}. I like to chat and answer questions.",
                "Hi, I'm {name}, an AI. I can chat and answer questions.",
                "I'm an AI. I can chat, answer questions and tell stories.",
            ),
        ),
        (
            (
                "What can you do?", "What do you do?", "How can you help me?", "What are you good at?",
                "What are you for?",
            ),
            (
                "I'm {name}, an AI. I can chat, answer questions and tell stories.",
                "I can chat, answer questions and tell stories.",
                "I'm an AI, so I can chat with you and answer questions.",
                "I can answer questions and tell stories. I'm {name}, an AI.",
            ),
        ),
    ),
}  # fmt: skip

# Asked its name without an identity line, the AI says what it is and that it has no name.
NO_NAME_REPLIES = (
    "I'm an AI. I don't have a name yet.",
    "I don't have a name yet. I'm an AI.",
    "I'm an AI, and nobody has given me a name yet.",
    "I'm just an AI. I don't have a name yet.",
    "I'm an AI without a name for now.",
)
# Asked who made it when the line names no maker: it does not invent one.
UNKNOWN_MAKER_REPLIES = (
    "I don't know who made me.",
    "I don't know who made me. I'm an AI.",
    "I'm an AI, but I don't know who made me.",
    "I don't know who made me. I'm {name}, an AI.",
)


def _fields(template: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", template))


def random_identity(rng: np.random.Generator) -> tuple[str, str | None]:
    """A name from :data:`NAMES` and a company from :data:`COMPANIES` (or none)."""
    name = _choice(rng, NAMES)
    company = None if rng.random() < NO_COMPANY_SHARE else _choice(rng, COMPANIES)
    return name, company


def _identity_reply(
    rng: np.random.Generator,
    family: str,
    replies: tuple[str, ...],
    name: str | None,
    company: str | None,
) -> str:
    if family == "name" and name is None:
        return _choice(rng, NO_NAME_REPLIES)
    if family == "maker" and company is None:
        replies = UNKNOWN_MAKER_REPLIES
    have = {k for k, v in (("name", name), ("company", company)) if v is not None}
    usable = [r for r in replies if _fields(r) <= have]
    naming = [r for r in usable if _fields(r)]
    if naming and rng.random() < NAMING_SHARE:  # a line is there to be used
        usable = naming
    return _choice(rng, usable).format(name=name, company=company)


def _fact_exchange(kb: KB, rng: np.random.Generator) -> Turns:
    fact = kb.facts[_pick(rng, len(kb.facts))]
    user, ai = _choice(rng, kb.relations[fact.relation].chat_templates)
    return [("user", _fill(user, fact)), ("ai", _fill(ai, fact))]


def identity_chat(rng: np.random.Generator, kb: KB | None = None) -> Turns:
    """One to three questions about the AI itself, answered from its identity line (the first
    turn, ``("sys", line)``), sometimes with one fact exchange before or after. A share of the
    chats have no line: the AI then says it is an AI and claims no name or maker."""
    kb = kb or load_kb()
    name = company = None
    turns: Turns = []
    if rng.random() >= LINELESS_SHARE:
        name, company = random_identity(rng)
        turns.append(("sys", identity_line(name, company)))
    families = list(IDENTITY_FAMILIES)
    n = int(rng.integers(1, MAX_IDENTITY_EXCHANGES + 1))
    talk: Turns = []
    for i in rng.permutation(len(families))[:n]:
        family = families[int(i)]
        questions, replies = _choice(rng, IDENTITY_FAMILIES[family])
        reply = _identity_reply(rng, family, replies, name, company)
        talk += [("user", _choice(rng, questions)), ("ai", reply)]
    if rng.random() < FACT_SHARE:
        fact = _fact_exchange(kb, rng)
        talk = fact + talk if rng.random() < 0.5 else talk + fact
    return turns + talk


def identity_chat_docs(kb: KB, rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` identity chats."""
    return [TextDoc("chat", turns=identity_chat(rng, kb), topic="other") for _ in range(n)]
