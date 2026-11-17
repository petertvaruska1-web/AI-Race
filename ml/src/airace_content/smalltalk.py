"""Practice conversations between a person and a small, friendly AI (``assistant_chat``).

No simple, permissively licensed assistant conversations exist at scale, so the game writes its
own: one to four exchanges of plain small talk, in which every turn follows from the one before.
The person greets, says how they feel and why, tells something about themselves, asks for a joke,
a fact, an idea or a bit of advice, asks about the AI, asks what it cannot know, reacts, and says
goodbye. The AI answers in a few short, warm sentences, as an AI: it has no body, family, home or
day of its own, it is honest about what it cannot see or
do, and about two in five of its turns end with a simple question that fits what was just said.

Two in five of the chats start with an identity line (``("sys", "You are Zula, an AI made by
Pengu.")``): there the AI says its name, and its maker now and then, when it greets the person and
when it is asked its name, and nowhere else; in the others it has no name and no maker. Half of the
chats give the person and their pets made-up names and not names from a short list: a name just
read can only be copied, and copying is what the practice is for.

The talk is written as data: a :class:`Step` is what the person may say and the AI lines that fit
it, and an AI :class:`Line` that asks something carries the steps that answer it (``then``) or the
name of a handler that does (``ask``). Slots in braces (``{pet}``, ``{food}``) are drawn once per
chat, so a later turn says the same pet or food as an earlier one. Facts come only from the
knowledge base, through its own training sentences, and no turn ever names a subject the follow-up
probe holds back.

Every choice comes from the ``rng`` given, so the same generator state gives the same chats.
"""

import string
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from airace_content.chats import (
    NO_NAME_REPLIES,
    _cap,
    _says_held,
    _usable_later,
    fresh_name,
    random_identity,
)
from airace_content.held import is_held_small_talk
from airace_ml.experiments.chatprobes import IDENTITY_COMPANY, IDENTITY_NAMES
from airace_ml.persona import identity_line
from airace_ml.skills.kb import KB, Fact, load_kb
from airace_ml.skills.types import TextDoc

Turns = list[tuple[str, str]]

# The eight kinds of exchange the brief names, in its order.
KINDS = (
    "opener", "feeling", "self", "request", "about_ai", "cannot_know", "reaction", "closer",
)  # fmt: skip
EXCHANGE_SHARES = (0.35, 0.30, 0.25, 0.10)  # chats of one, two, three and four exchanges
MAX_AI_CHARS = 220
MAX_USER_CHARS = 120
ANSWER_SHARE = 0.9  # the person answers what the AI just asked, most of the time
_TRIES = 200


# --- the shape of the talk ---------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """What the person may say (any of ``users``) and the AI lines that fit it."""

    users: tuple[str, ...]
    replies: tuple["Line", ...]


@dataclass(frozen=True)
class Line:
    """One AI line. ``then`` holds what the person may answer to it; ``ask`` names a handler for
    the answer instead; ``ctx`` says what kind of thing the line is, for the reactions that fit
    it; ``why`` is the reason the AI gives when asked "Why?". A line whose text starts with "@"
    is an action the generator carries out (``@joke``, ``@fact``, ``@idea``, ``@topic``)."""

    text: str
    then: tuple[Step, ...] = ()
    ask: str | None = None
    ctx: str = "chat"
    why: str | None = None


def L(
    text: str, *then: Step, ask: str | None = None, ctx: str = "chat", why: str | None = None
) -> Line:
    """An AI line (see :class:`Line`)."""
    return Line(text, tuple(then), ask, ctx, why)


def S(users: str, *replies: "str | Line | tuple[Line, ...]") -> Step:
    """A step: the person's phrasings, separated by " | ", and the AI lines that fit them (a
    tuple of lines, as :func:`C` gives, counts as its lines)."""
    lines: list[Line] = []
    for r in replies:
        if isinstance(r, tuple):
            lines.extend(r)
        else:
            lines.append(r if isinstance(r, Line) else Line(r))
    return Step(tuple(u.strip() for u in users.split("|") if u.strip()), tuple(lines))


def _alts(text: str) -> tuple[str, ...]:
    """The phrasings in ``text``, separated by " | "."""
    return tuple(t.strip() for t in text.split("|") if t.strip())


def C(*replies: "str | Line", ctx: str) -> tuple[Line, ...]:
    """Replies that all have the context ``ctx``."""
    return tuple(
        Line(r.text, r.then, r.ask, ctx, r.why) if isinstance(r, Line) else Line(r, ctx=ctx)
        for r in replies
    )


# --- slots -------------------------------------------------------------------------------------

# Words the person uses about themselves. A slot is drawn the first time a turn needs it and kept
# for the rest of the chat. Options that are dicts set several fields at once.
_PETS = (
    "cat", "kitten", "puppy", "rabbit", "bunny", "goldfish", "parrot", "turtle", "tortoise",
    "lizard", "gecko", "budgie", "mouse", "fish", "hermit crab", "rat", "ferret",
    "chinchilla", "canary", "cockatiel",
)  # fmt: skip
_PET_NAMES = (
    "Biscuit", "Max", "Luna", "Bella", "Milo", "Coco", "Pepper", "Oreo", "Daisy", "Teddy",
    "Shadow", "Ginger", "Nugget", "Peanut", "Mochi", "Ziggy", "Bubbles", "Sunny", "Pickles",
    "Rocky", "Charlie", "Lucky", "Socks", "Muffin", "Noodle", "Waffles", "Pumpkin", "Snowy",
    "Toby", "Rex", "Bean", "Olive", "Pip", "Cookie", "Fluffy", "Smokey", "Honey", "Maple",
    "Sprinkles", "Button", "Crumble", "Jelly", "Pebbles", "Ollie", "Zelda", "Loki", "Simba",
    "Nala", "Gizmo", "Patch", "Whiskers", "Tofu", "Sushi", "Marshmallow", "Cupcake", "Dot",
    "Bubba", "Frankie", "Lulu", "Mango", "Kiki", "Popcorn", "Captain", "Squeaky", "Bandit",
    "Taco", "Buttons", "Rosie", "Duke", "Penny",
)  # fmt: skip
_FOODS = tuple(
    {
        "food": f,
        "fis": "are" if pl else "is",
        "fit": "them" if pl else "it",
        "fs": "" if pl else "s",
    }
    for f, pl in (
        ("pizza", 0),
        ("pasta", 0),
        ("spaghetti", 0),
        ("pancakes", 1),
        ("waffles", 1),
        ("ice cream", 0),
        ("chocolate", 0),
        ("cookies", 1),
        ("apples", 1),
        ("bananas", 1),
        ("grapes", 1),
        ("carrots", 1),
        ("rice", 0),
        ("noodles", 1),
        ("soup", 0),
        ("tacos", 1),
        ("sushi", 0),
        ("burgers", 1),
        ("french fries", 1),
        ("popcorn", 0),
        ("cheese", 0),
        ("sandwiches", 1),
        ("dumplings", 1),
        ("mangoes", 1),
        ("watermelon", 0),
        ("cereal", 0),
        ("yogurt", 0),
        ("eggs", 1),
        ("broccoli", 0),
        ("blueberries", 1),
        ("cherries", 1),
        ("peaches", 1),
        ("muffins", 1),
        ("cupcakes", 1),
        ("cake", 0),
        ("apple pie", 0),
        ("mac and cheese", 0),
        ("mashed potatoes", 1),
        ("peas", 1),
        ("cucumbers", 1),
        ("nachos", 1),
        ("pretzels", 1),
        ("hot chocolate", 0),
        ("smoothies", 1),
        ("lemonade", 0),
        ("oranges", 1),
        ("pears", 1),
        ("salad", 0),
        ("raspberries", 1),
        ("tomato soup", 0),
        ("fried rice", 0),
        ("grilled cheese", 0),
        ("burritos", 1),
        ("bagels", 1),
        ("granola", 0),
        ("kiwis", 1),
        ("plums", 1),
        ("meatballs", 1),
    )
)
_COLORS = (
    {"color": "red", "assoc": "ripe cherries and bright fire trucks"},
    {"color": "red", "assoc": "ladybugs and juicy watermelon"},
    {"color": "blue", "assoc": "the sky on a sunny day"},
    {"color": "blue", "assoc": "the deep, calm sea"},
    {"color": "green", "assoc": "grass, leaves and tall trees"},
    {"color": "green", "assoc": "a forest in the spring"},
    {"color": "yellow", "assoc": "sunshine and sunflowers"},
    {"color": "yellow", "assoc": "bananas and little ducklings"},
    {"color": "purple", "assoc": "grapes and lavender flowers"},
    {"color": "purple", "assoc": "the sky just after sunset"},
    {"color": "pink", "assoc": "cherry blossoms and bubble gum"},
    {"color": "pink", "assoc": "flamingos and pretty sunsets"},
    {"color": "orange", "assoc": "pumpkins and autumn leaves"},
    {"color": "orange", "assoc": "oranges and warm sunsets"},
    {"color": "black", "assoc": "the night sky full of stars"},
    {"color": "white", "assoc": "new snow and fluffy clouds"},
    {"color": "gold", "assoc": "treasure and shiny medals"},
    {"color": "silver", "assoc": "the moon and twinkling stars"},
    {"color": "teal", "assoc": "a tropical sea"},
    {"color": "brown", "assoc": "warm cocoa and cozy wooden cabins"},
)
_ANIMALS = (
    {"animal": "elephants", "remark": "They are the biggest animals that live on land."},
    {"animal": "giraffes", "remark": "They are the tallest animals in the world."},
    {"animal": "penguins", "remark": "They are amazing swimmers."},
    {"animal": "whales", "remark": "The blue whale is the biggest animal that has ever lived."},
    {"animal": "sharks", "remark": "They have lived in the sea for a very long time."},
    {"animal": "koalas", "remark": "They sleep for most of the day."},
    {"animal": "foxes", "remark": "They have big, bushy tails."},
    {"animal": "wolves", "remark": "They live and hunt together in groups called packs."},
    {"animal": "otters", "remark": "They are playful and great swimmers."},
    {"animal": "owls", "remark": "They can turn their heads a very long way."},
    {"animal": "dolphins", "remark": "They are very clever and like to play."},
    {"animal": "cats", "remark": "They spend a lot of the day napping."},
    {"animal": "dogs", "remark": "They have a wonderful sense of smell."},
    {"animal": "horses", "remark": "They can sleep standing up."},
    {"animal": "rabbits", "remark": "Their long ears help them hear very well."},
    {"animal": "turtles", "remark": "They carry their shells with them everywhere."},
    {"animal": "octopuses", "remark": "They have eight arms and are very clever."},
    {"animal": "bees", "remark": "They make honey and help flowers grow."},
    {"animal": "ladybugs", "remark": "They have little spots on their backs."},
    {"animal": "parrots", "remark": "Some of them can copy the sounds they hear."},
    {"animal": "hedgehogs", "remark": "They roll into a ball when they feel scared."},
    {"animal": "pandas", "remark": "They eat lots and lots of bamboo."},
    {"animal": "lions", "remark": "They live in family groups called prides."},
    {"animal": "tigers", "remark": "Each one has its own pattern of stripes."},
    {"animal": "frogs", "remark": "They start their lives as tiny tadpoles."},
    {"animal": "butterflies", "remark": "They start their lives as caterpillars."},
    {"animal": "seals", "remark": "They are clumsy on land but graceful in the water."},
    {"animal": "deer", "remark": "They are shy and very quiet."},
    {"animal": "sea turtles", "remark": "They can swim across whole oceans."},
    {"animal": "jellyfish", "remark": "They have no bones at all."},
    {"animal": "flamingos", "remark": "They often stand on one leg."},
    {"animal": "raccoons", "remark": "They have little masks of dark fur around their eyes."},
    {"animal": "unicorns", "remark": "They are only in stories, but they are fun to imagine."},
    {"animal": "dinosaurs", "remark": "They lived a very, very long time ago."},
    {"animal": "puppies", "remark": "They are full of energy and love to play."},
    {"animal": "kittens", "remark": "They love to chase and pounce on things."},
)
# Games whose names end in "s" but are one game ("checkers is fun").
_SINGULAR_GAMES = frozenset(
    {"checkers", "dominoes", "marbles", "charades", "musical chairs", "go fish"}
)
_GAMES = tuple(
    {"game": g, "gis": "are" if g.endswith("s") and g not in _SINGULAR_GAMES else "is"}
    for g in (
        "board games",
        "hide-and-seek",
        "tag",
        "chess",
        "checkers",
        "video games",
        "card games",
        "puzzles",
        "jigsaw puzzles",
        "word games",
        "dominoes",
        "building games",
        "I spy",
        "make-believe",
        "jump rope",
        "marbles",
        "hopscotch",
        "memory games",
        "freeze tag",
        "charades",
        "tic-tac-toe",
        "racing games",
        "treasure hunts",
        "musical chairs",
        "go fish",
        "chess puzzles",
        "dodgeball",
        "capture the flag",
    )
)
_SPORTS = (
    "soccer", "basketball", "tennis", "swimming", "baseball", "volleyball", "running",
    "gymnastics", "karate", "dancing", "hockey", "ice skating", "skateboarding", "badminton",
    "table tennis", "cycling", "climbing", "judo", "softball", "roller skating",
)  # fmt: skip
_INSTRUMENTS = (
    "piano", "guitar", "violin", "drums", "flute", "recorder", "trumpet", "ukulele", "cello",
    "clarinet", "keyboard", "saxophone",
)  # fmt: skip
_GIFTS = (
    "bike", "book", "puzzle", "board game", "kite", "skateboard", "paint set", "scooter",
    "soccer ball", "telescope", "Lego set", "backpack", "pair of roller skates", "art kit",
)  # fmt: skip
_TOYS = (
    "toy car", "stuffed bunny", "favorite ball", "yo-yo", "toy robot", "action figure",
    "toy dinosaur", "stuffed kitten", "doll", "toy train", "spinning top", "bouncy ball",
    "toy rocket", "plush whale",
)  # fmt: skip
_SUBJECTS = (
    "math", "spelling", "science", "reading", "history", "geography", "French", "Spanish",
    "music", "art", "English", "writing",
)  # fmt: skip
_SIBLINGS = (
    "little sister", "little brother", "big sister", "big brother", "baby brother",
    "baby sister", "twin brother", "twin sister", "cousin",
)  # fmt: skip
_RELATIVES = ("grandma", "grandpa", "aunt", "uncle", "cousin", "nana", "granddad")
_THINGS = (
    "animals", "space", "books", "drawing", "games", "stories", "the ocean", "dinosaurs",
    "robots", "science", "puzzles", "riddles", "words", "jokes", "planets", "nature",
    "math", "rainbows", "snow", "volcanoes", "bugs", "trees", "the moon", "art",
)  # fmt: skip
_AGES = tuple(range(5, 16))
_NUMBER_WORDS = {
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
}  # fmt: skip
_AGE_OPTIONS = tuple(
    {"age": said, "agew": _NUMBER_WORDS[n]} for n in _AGES for said in (str(n), _NUMBER_WORDS[n])
)
# First names a person may give. Names the follow-up probe or the identity probe holds are left
# out when the module loads.
_USER_NAMES_TEXT = """
Lily Emma Noah Liam Olivia Ava Mia Sophia Lucas Ethan Mason Logan Ella Grace Chloe Zoe Leo Max
Sam Ben Jack Ruby Isla Amelia Harry Oscar Jacob Henry Alfie Archie Evie Freya Poppy Rosie Daisy
Maya Aria Nora Hazel Ivy Alice Clara Ada Elijah Mateo Ezra Owen Theo Finn Kai Arjun Priya Aisha
Omar Yusuf Zara Hana Mei Kenji Yuki Sofia Diego Lucia Carlos Elena Pablo Nina Lena Jonas Felix
Hugo Emil Lars Anya Ivan Mila Tara Rohan Dev Asha Kofi Amara Zuri Tomas Rafael Ines Leila Sami
Nadia Malik Imani Ayla Selin Emre Bodhi Jasper Milo Ezra Ruth Maeve Cora Iris Gemma Toby Rory
Callum Isaac Adam Sara Lucy Holly Beth Molly Jade Amber Megan Kate Josh Ryan Dylan Tyler Caleb
Eli Nate Luke Jake Gabe Wes Abby Kayla Paige Hailey Lexi Tess Wren Juno Esme Nico Remy Ana
"""
_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


def _user_names() -> tuple[str, ...]:
    """The first names a person may give: none the identity probe uses, and none that is a
    subject the follow-up probe holds back ("Poppy")."""
    banned = {n.lower() for n in IDENTITY_NAMES}
    kb = load_kb()
    out: list[str] = []
    for name in _USER_NAMES_TEXT.split():
        if name.lower() not in banned and name not in out and not _says_held(kb, name):
            out.append(name)
    return tuple(out)


SLOTS: dict[str, tuple] = {
    "pet": _PETS,
    "pname": _PET_NAMES,
    "food": _FOODS,
    "color": _COLORS,
    "animal": _ANIMALS,
    "game": _GAMES,
    "sport": _SPORTS,
    "instrument": _INSTRUMENTS,
    "gift": _GIFTS,
    "toy": _TOYS,
    "subject": _SUBJECTS,
    "sibling": _SIBLINGS,
    "relative": _RELATIVES,
    "thing": _THINGS,
    "age": _AGE_OPTIONS,
    "uname": (),  # filled with the user names below
    "day": _DAYS,
}
_FIELD_GROUP: dict[str, str] = {}


def _register_slots() -> None:
    SLOTS["uname"] = _user_names()
    probe = {n.lower() for n in (*IDENTITY_NAMES, IDENTITY_COMPANY)}  # never said by the AI
    SLOTS["pname"] = tuple(
        n for n in _PET_NAMES if n.lower() not in probe and not _says_held(load_kb(), n)
    )
    for group, options in SLOTS.items():
        keys = options[0].keys() if isinstance(options[0], dict) else (group,)
        for key in keys:
            _FIELD_GROUP[key] = group


_register_slots()


def _fields(text: str) -> list[str]:
    """The slots of a template, each once, in the order they first appear. A list and not a set:
    the slots are filled in this order with random draws, and the order of a set of strings
    differs from one run of Python to the next, which would make the same seed give other
    chats."""
    return list(
        dict.fromkeys(name for _, name, _, _ in string.Formatter().parse(text) if name)
    )


# --- jokes -------------------------------------------------------------------------------------

# Short, clean jokes and riddles, each a question and its answer.
JOKES: tuple[tuple[str, str], ...] = (
    ("What do you call a sleepy dinosaur?", "A dino-snore!"),
    ("Why did the math book look so sad?", "It had too many problems."),
    ("What do you call cheese that isn't yours?", "Nacho cheese!"),
    ("Why don't eggs tell jokes?", "They would crack each other up."),
    ("What has keys but can't open a single lock?", "A piano."),
    ("What gets wetter the more it dries?", "A towel."),
    ("Why was the broom late?", "It overswept."),
    ("What has hands but can't clap?", "A clock."),
    ("What goes up but never goes down?", "Your age."),
    ("What do you call a cow that plays an instrument?", "A moo-sician."),
    ("Why did the bicycle fall over?", "It was two-tired."),
    ("How can you tell the ocean is friendly?", "It waves."),
    ("Why are fish so smart?", "They live in schools."),
    ("What do you call a snowman in the summer?", "A puddle."),
    ("What did one plate say to the other plate?", "Lunch is on me!"),
    ("Why did the banana put on sunscreen?", "It didn't want to peel."),
    ("What room has no doors and no windows?", "A mushroom."),
    ("Why did the scarecrow win a prize?", "He was outstanding in his field."),
    ("What do you call a sleeping bull?", "A bulldozer."),
    ("Why did the picture go to jail?", "It was framed."),
    ("What did the zero say to the eight?", "Nice belt!"),
    ("Why was six afraid of seven?", "Because seven ate nine."),
    ("What do you call a pile of kittens?", "A meow-ntain."),
    ("What did the big flower say to the little flower?", "Hi, bud!"),
    ("What do you call a fake noodle?", "An impasta."),
    ("What has one eye but can't see?", "A needle."),
    ("What can you catch but never throw?", "A cold."),
    ("What has a neck but no head?", "A bottle."),
    ("What is full of holes but still holds water?", "A sponge."),
    ("What gets bigger the more you take away from it?", "A hole."),
    ("What has lots of words but never speaks?", "A book."),
    ("What goes up and down but never moves?", "The stairs."),
    ("Why did the golfer bring two pairs of pants?", "In case he got a hole in one."),
    ("What do you call a sad raspberry?", "A blueberry."),
    ("Why did the sun go to school?", "To get brighter."),
    ("What kind of books do planets like to read?", "Comet books."),
    ("How do you plan a party in space?", "You planet."),
    ("Where do cows go for fun?", "To the moo-vies."),
    ("What did the paper say to the pencil?", "Write on!"),
    ("Why was the pencil so sad?", "It had no point."),
    ("What is orange and sounds like a parrot?", "A carrot."),
    ("Why did the girl bring a ladder to school?", "She wanted to go to high school."),
    ("Why did the cookie go to the nurse?", "It felt crummy."),
    ("What is a cat's favorite color?", "Purr-ple."),
    ("What did one wall say to the other wall?", "I'll meet you at the corner."),
    ("What did the triangle say to the circle?", "You're pointless."),
    ("What do you call a funny mountain?", "Hill-arious."),
    ("Why do cows wear bells?", "Because their horns don't work."),
    ("What do sea monsters like to eat?", "Fish and ships."),
    ("Where do bees wait when they want to go to town?", "At the buzz stop."),
    ("What did the stamp say to the envelope?", "Stick with me and we'll go places."),
    ("Why did the music teacher need a ladder?", "To reach the high notes."),
    ("What do you call a boomerang that doesn't return?", "A stick."),
    ("What is brown and sticky?", "A stick."),
    ("Why did the cat sit on the computer?", "To keep an eye on the mouse."),
    ("What is a cat's favorite dessert?", "Mice cream."),
    ("What do you call a train full of toffee?", "A chew-chew train."),
    ("What do elves learn in school?", "The elf-abet."),
    ("How do you talk to a giant?", "You use big words."),
    ("Why was the baseball stadium so cool?", "It was full of fans."),
    ("What do you call a dinosaur that knows lots of words?", "A thesaurus."),
    ("Why did the robot go on vacation?", "It needed to recharge its batteries."),
    ("Why can't you trust the stairs?", "They're always up to something."),
    ("Why don't oysters share their pearls?", "Because they're shellfish."),
    ("Why did the orange stop rolling?", "It ran out of juice."),
    ("Why did the boy run around his bed?", "To catch up on his sleep."),
    ("What is the strongest day of the week?", "Sunday. The rest are weekdays."),
    ("Why did the clock get in trouble?", "It tocked too much."),
    ("Where do pencils go on vacation?", "Pencil-vania."),
    ("What do you call a lazy potato?", "A couch potato."),
    ("What do you call a sleeping pizza?", "A piz-zzz-a."),
    ("What kind of water never freezes?", "Hot water."),
    ("What has to be broken before you can use it?", "An egg."),
    ("Which month has 28 days?", "All of them."),
    ("What can go all around the world but stays in a corner?", "A stamp."),
    ("What is always ahead of you but can never be seen?", "The future."),
    ("What can run but can't walk?", "A river."),
    ("What belongs to you, but other people use it more than you do?", "Your name."),
    ("What has a head and a tail but no legs?", "A coin."),
    ("What can fill a whole room but takes up no space?", "Light."),
    ("What do you call a dinosaur that crashes its car?", "Tyrannosaurus wrecks."),
    ("Why did the dinosaur cross the road?", "Because chickens didn't exist yet."),
    ("Why did the kid cross the playground?", "To get to the other slide."),
    ("What did one volcano say to the other volcano?", "I lava you."),
    ("Why did the melon jump into the lake?", "It wanted to be a watermelon."),
    ("What do you call two birds that like each other?", "Tweethearts."),
    ("Where do sheep go to get a haircut?", "To the baa-baa shop."),
    ("What did the pencil say to the sharpener?", "Stop going in circles and get to the point."),
    ("Why did the soccer ball quit the team?", "It was tired of being kicked around."),
    ("What is a whale's favorite game?", "Swallow the leader."),
    ("What do you call a cat that likes to go bowling?", "An alley cat."),
    ("Why do bananas never feel lonely?", "Because they hang around in bunches."),
    ("What do you call an apple that plays the trumpet?", "A tooty fruity."),
    ("What do you call a grumpy cow?", "Moo-dy."),
    ("Why did the kid study on the airplane?", "To get higher grades."),
    ("What is the smartest insect?", "A spelling bee."),
    ("Why do mice need oiling?", "Because they squeak."),
    ("What do you call a group of whales that play music?", "An orca-stra."),
    ("What do you get if you put a cow on a trampoline?", "A milkshake."),
    ("What is a computer's favorite snack?", "Microchips."),
    ("Why did the turkey join the band?", "Because it had the drumsticks."),
    ("What is black and white and read all over?", "A newspaper."),
    ("How do trees get on the internet?", "They log in."),
    ("What did the beach say when the tide came in?", "Long time, no sea."),
    ("Why was the cat so good at video games?", "It had nine lives."),
    ("What do you call a shy pebble?", "A little boulder."),
    ("What do you call an elephant that doesn't matter?", "An irrelephant."),
    ("Why did the student wear sunglasses in class?", "Her future was so bright."),
    ("How do you make an octopus giggle?", "With ten-tickles."),
    ("What do you call a dinosaur that never gives up?", "A try-ceratops."),
    ("What did one hat say to the other hat?", "You stay here. I'll go on ahead."),
    ("What season is it when you jump on a trampoline?", "Spring time."),
    ("Why did the moon skip dinner?", "Because it was full."),
    ("How does a penguin build its house?", "Igloos it together."),
    ("What did the tree say to the wind?", "Leaf me alone!"),
    ("Why was the river so rich?", "It had two banks."),
    ("What did the mother broom say to the baby broom?", "It's time to go to sweep."),
    ("Why did the boy bring string to the soccer game?", "So he could tie the score."),
    ("What can you break without even touching it?", "A promise."),
    ("What kind of band never plays any music?", "A rubber band."),
    ("What goes through towns and over hills but never moves?", "A road."),
    ("What has four legs but can't walk?", "A table."),
    ("What do you call a bee that can't make up its mind?", "A maybe."),
    ("Why don't scientists trust atoms?", "Because they make up everything."),
    ("What did the light bulb say to its mom?", "I love you watts and watts."),
    ("Why did the gardener plant a light bulb?", "She wanted to grow a power plant."),
    ("What do you call a tiny mother?", "A minimum."),
    ("What kind of music are balloons afraid of?", "Pop music."),
    ("Why did the drum take a nap?", "It was beat."),
    ("How do you fix a broken pumpkin?", "With a pumpkin patch."),
    ("Why did the computer sneeze?", "It caught a virus."),
    ("Why are pirates called pirates?", "Because they arrr."),
    ("What did one snowman say to the other snowman?", "Do you smell carrots?"),
    ("What do snowmen like for breakfast?", "Snowflakes."),
    ("Where do snowmen keep their money?", "In a snow bank."),
    ("How do you make a band stand?", "Take away their chairs."),
    ("Why did the girl put peanut butter on the road?", "To go with the traffic jam."),
    ("Which vegetable always wins the race?", "The runner bean."),
    ("Why is a leopard bad at hide-and-seek?", "Because it's always spotted."),
    ("What do you call a sheep that knows karate?", "A lamb chop."),
    ("What did the grape do when someone stepped on it?", "It let out a little whine."),
    ("How does a cucumber turn into a pickle?", "It has a jarring experience."),
    ("Why don't you ever see giraffes in elementary school?", "They're all in high school."),
    ("What do you get when you cross a fish and an elephant?", "Swimming trunks."),
    ("What kind of shoes do bananas wear?", "Slippers."),
    ("Why did the teacher wear sunglasses?", "Because her class was so bright."),
    ("Why are cats such good singers?", "They're very mew-sical."),
    ("What do cats like to eat on a hot day?", "Mice cubes."),
)
# Knock-knock jokes: the word, and what the AI says after "... who?".
KNOCK_KNOCK: tuple[tuple[str, str], ...] = (
    ("Lettuce", "Lettuce in, it's cold out here!"),
    ("Boo", "Don't cry, it's only a joke!"),
    ("Orange", "Orange you glad to hear a joke?"),
    ("Cow says", "No, silly, a cow says moo!"),
    ("Atch", "Bless you!"),
    ("Tank", "You're welcome!"),
    ("Olive", "Olive you too!"),
    ("Wooden shoe", "Wooden shoe like to hear another joke?"),
    ("Ice cream", "Ice cream if you don't let me in!"),
    ("Water", "Water you doing? Let me in!"),
    ("Hawaii", "Hawaii doing? I'm fine, thanks!"),
    ("Doughnut", "Doughnut forget to smile today!"),
    ("Says", "Says me, that's who!"),
    ("Iva", "Iva sore hand from knocking!"),
)


# --- what the person answers to small questions --------------------------------------------------

# Answers to "What have you been up to?" and "What's up with you?".
UP_TO = (
    S(
        "Not much. | Not much, just relaxing. | Nothing much. | Just chilling. | Not a lot, really.",
        L("Relaxing is nice. Would you like to hear a joke or a fun fact?", ask="joke_or_fact"),
        L("That sounds peaceful. Is there anything you'd like to talk about?", ask="topic"),
        "A quiet day can be really nice.",
        "Sounds like a calm day. Those are nice sometimes.",
        L(
            "Sometimes doing nothing is just what we need. Want to hear something fun?",
            ask="joke_or_fact",
        ),
    ),
    S(
        "I just got home from school. | School just finished. | I just got back from school.",
        L("Welcome home! How was school today?", ask="how_day"),
        L("I hope school went well. What was the best part of your day?", ask="best_part"),
        "Time to rest a bit, then. You've earned it.",
        "Welcome home! I hope school went well.",
    ),
    S(
        "I'm doing my homework. | I'm working on my homework. | Just homework.",
        L("Good for you for getting it done. What subject is it?", ask="hw_subject"),
        "Homework is easier one small piece at a time. You can do it!",
        L("I can try to help if you get stuck. What's it about?", ask="hw_subject"),
    ),
    S(
        "I'm playing with my {pet}. | Just playing with my {pet}.",
        L("That sounds fun! What's your {pet}'s name?", ask="pet_name"),
        "Playing with a {pet} is a lovely way to spend time.",
    ),
    S(
        "I'm drawing. | I've been drawing all morning. | Just drawing a picture.",
        L("I love that! What are you drawing?", ask="drawing_what"),
        "Drawing is a great way to use your imagination.",
    ),
    S(
        "I'm reading a book. | I've been reading. | Just reading.",
        L("Reading is wonderful. What is your book about?", ask="book_about"),
        "Books can take you anywhere. Enjoy it!",
    ),
    S(
        "I'm waiting for dinner. | Just waiting for lunch. | Waiting for my snack.",
        "Waiting can feel long, but it will be ready soon.",
        L("I hope it's something tasty. While you wait, want a joke?", ask="more_joke"),
    ),
    S(
        "I'm playing {game}. | I've been playing {game}.",
        L(
            "Fun! Who are you playing with?",
            S(
                "My {sibling}. | Some kids from my class. | Just me. | My cousins. | My mom.",
                "That sounds like a fun time!",
                "Nice! Have fun.",
            ),
        ),
        "{Game} {gis} a great way to have fun.",
    ),
)

# Answers to "How was school?" and "How was your day?".
HOW_DAY = (
    S(
        "It was good. | It was great! | It was fun. | Pretty good. | Really good, thanks.",
        L("I'm glad to hear that! What was the best part?", ask="best_part"),
        "That's great. Good days are the best.",
        "Yay! I'm happy your day went well.",
    ),
    S(
        "It was okay. | It was fine. | Not bad. | It was all right.",
        L("Okay days are fine too. Did anything fun happen?", ask="best_part"),
        "Not every day has to be exciting. I hope tomorrow is even better.",
    ),
    S(
        "It was bad. | Not good. | It was a bad day. | It was really long.",
        L("I'm sorry it was a tough day. Do you want to talk about it?", ask="talk_about"),
        "I'm sorry. Tomorrow is a new start, and I'm here if you want to chat.",
        L("That sounds rough. Would a joke help cheer you up?", ask="more_joke"),
    ),
)

# Answers to "What was the best part?".
BEST_PART = (
    S(
        "Recess. | Playing outside at recess. | Recess, for sure.",
        "Recess is a great part of the day. Running around and playing is so much fun.",
        L("Recess is the best! What do you like to play?", ask="fav_game"),
    ),
    S(
        "Lunch. | Lunch was the best part. | We had pizza for lunch.",
        "A good lunch can really make a day better.",
        L("Yum! What's your favorite thing to have for lunch?", ask="fav_food"),
    ),
    S(
        "Art class. | We painted in art class. | I made something in art.",
        "Art class sounds like so much fun. I love hearing about what people make.",
        L("That sounds wonderful. What did you make?", ask="drawing_what"),
    ),
    S(
        "Gym class. | We played games in gym. | Gym was fun.",
        "Gym is a great way to move around and have fun.",
        L("Fun! What did you play?", ask="fav_sport"),
    ),
    S(
        "Seeing my buddies. | Hanging out with my classmates. | Playing with my friends.",
        "Time with people you like is the best part of a lot of days.",
        "That sounds lovely. It's great to have people to play with.",
    ),
    S(
        "I got a good grade. | I got a gold star. | My teacher said I did a great job.",
        "Well done! You should be proud of yourself.",
        L("That's wonderful! What was it for?", ask="hw_subject"),
    ),
    S(
        "Nothing really. | I don't know. | Nothing special.",
        "That's okay. Some days are just ordinary, and that's fine.",
        L("That's all right. Want to hear a joke to make it better?", ask="more_joke"),
    ),
)

# Answers to "Do you want to talk about it?".
TALK_ABOUT = (
    S(
        "Not really. | No, I don't want to talk about it. | Maybe later. | No thanks.",
        C(
            "That's okay. I'm here if you change your mind.",
            L("That's fine. Would you like a joke to cheer you up?", ask="more_joke"),
            "Okay. You don't have to. I hope things feel better soon.",
            ctx="comfort",
        ),
    ),
)

_DONT_KNOW_FAV = S(
    "I don't know. | I can't decide. | Hmm, I don't have one. | I like too many to pick one. "
    "| I'm not sure.",
    "That's okay! There are so many good ones that it's tricky to pick just one.",
    "Fair enough. You don't have to choose just one.",
    "That's fine. Maybe you'll find a favorite one day.",
)

# Favorites: what the person says when asked, and the AI lines that fit, by kind.
FAV_SAYS = {
    "food": "I like {food}. | {Food}! | Probably {food}. | My favorite is {food}. | I love {food}. "
    "| {Food}, for sure. | I really like {food}.",
    "color": "{Color}. | I like {color}. | {Color}, I think. | My favorite is {color}. "
    "| Definitely {color}.",
    "animal": "{Animal}. | I like {animal}. | I love {animal}! | {Animal}, for sure. "
    "| Probably {animal}.",
    "game": "{Game}. | I like {game}. | I like to play {game}. | Probably {game}. "
    "| {Game} {gis} the best.",
    "sport": "{Sport}. | I like {sport}. | I love {sport}! | Probably {sport}. "
    "| {Sport} is my favorite.",
}
FAV_REPLIES: dict[str, tuple[Line, ...]] = {
    "food": (
        L("{Food} {fis} a great choice! Lots of people love {fit}."),
        L(
            "Ooh, {food}! That sounds tasty. What do you like about {fit}?",
            S(
                "{Fit} {fis} so yummy. | {Fit} taste{fs} good. | I just like the taste. "
                "| I don't know, I just like {fit}.",
                "Good food can really make a day better.",
                "That makes sense. A favorite food is a happy thing to have.",
            ),
        ),
        L(
            "{Food} {fis} popular for a good reason. Do you have {fit} often?",
            S(
                "Yes, all the time! | Yes, a lot. | Pretty often.",
                "Lucky you! It's nice to have a favorite food often.",
            ),
            S(
                "No, only sometimes. | Not very often. | Only on special days.",
                "Then it's a nice treat when you do!",
            ),
        ),
        L("Yum! I can't taste food, but I like hearing about it. {Food} sound{fs} lovely."),
        L("Nice! {Food} {fis} a favorite for lots of people."),
        L(
            "{Food}! Good pick. Do you ever help make {fit}?",
            S(
                "Yes, I help sometimes. | Yes! | Sometimes.",
                "That's great. Cooking with someone is a fun thing to do together.",
            ),
            S(
                "No, not really. | No, I just eat {fit}. | Nope.",
                "That's fine! Eating {fit} is the best part anyway.",
            ),
        ),
    ),
    "color": (
        L(
            "{Color} is a lovely color. It makes me think of {assoc}.",
            why="It reminds me of {assoc}.",
        ),
        L(
            "Nice choice! {Color} makes me think of {assoc}. Why do you like it?",
            S(
                "It's bright. | It's pretty. | It makes me happy. | It's calm. | It looks cool.",
                "That's a great reason. Colors can really change how we feel.",
                "I like that. A favorite color is something that cheers you up.",
            ),
        ),
        L(
            "{Color}! Good choice. What's your favorite thing that is {color}?",
            S(
                "My bike. | My shoes. | My backpack. | My hoodie. | My water bottle. "
                "| My favorite pencil case.",
                "That sounds great. It must look really nice in {color}.",
            ),
        ),
        L("{Color}! That's a lovely pick. It makes me think of {assoc}."),
    ),
    "animal": (
        L("{Animal} are wonderful. {remark}"),
        L(
            "Great choice! {remark} What do you like most about them?",
            S(
                "They're cute. | They're so fluffy. | They're funny. | They're really cool. "
                "| They're smart.",
                "Those are good reasons to like them!",
                "I agree. There's a lot to love about {animal}.",
            ),
        ),
        L("{Animal}! Good pick. Did you know this? {remark}"),
        L(
            "I like talking about {animal}. {remark} Have you ever seen one up close?",
            S(
                "Yes, at the zoo! | Yes, once. | Yes, lots of times.",
                "That must have been exciting!",
            ),
            S(
                "No, not yet. | No, only in pictures. | Never.",
                "Maybe one day you will. Pictures are a great start.",
            ),
        ),
    ),
    "game": (
        L(
            "{Game} {gis} so much fun. Do you play with your family?",
            S(
                "Yes, all the time. | Sometimes. | Yes, with my {sibling}.",
                "That sounds like a great way to spend time together.",
            ),
            S(
                "No, with my classmates. | No, mostly by myself. | Not really.",
                "That works too. Fun is fun!",
            ),
        ),
        L("Good choice! {Game} {gis} a great way to have fun."),
        L(
            "Ooh, {game}! Are you good at it?",
            S("Yes, really good! | I think so. | I win a lot.", "That's great! Practice pays off."),
            S(
                "Not really. | I'm still learning. | I lose a lot.",
                "That's okay. The fun part is playing, not winning.",
            ),
        ),
    ),
    "sport": (
        L(
            "{Sport} is a great way to stay active. How long have you been doing it?",
            S(
                "A long time. | A few years. | Since I was little. | Just this year.",
                "That's great. Keep having fun with it!",
            ),
        ),
        L("Nice! {Sport} sounds like a lot of fun."),
        L(
            "{Sport}! That's awesome. Do you do it with a team?",
            S("Yes! | Yes, at school. | Yes, on weekends.", "Being on a team is a great thing."),
            S(
                "No, just for fun. | No, by myself. | Not really.",
                "Doing it just for fun is great too.",
            ),
        ),
    ),
}

# Small asks answered from data: what the person may say and the lines that fit.
ASK_STEPS: dict[str, tuple[Step, ...]] = {
    "up_to": UP_TO,
    "how_day": HOW_DAY,
    "best_part": BEST_PART,
    "talk_about_no": TALK_ABOUT,
    "hw_subject": (
        S(
            "It's {subject}. | {Subject}. | {Subject} homework. | It's for {subject}.",
            "{Subject} can be tricky, but practice really helps. You can do it!",
            "{Subject} is a good one to learn. Try one small part at a time.",
            "Good luck with your {subject}. Take a short break if your head feels full.",
        ),
    ),
    "drawing_what": (
        S(
            "A {pet}. | A castle. | A rocket ship. | My family. | A rainbow. | A dragon. "
            "| A house with a big garden. | Some flowers. | A robot. | The ocean.",
            "That sounds like a wonderful picture!",
            "I love that idea. I hope you have fun with it.",
            "What a great thing to draw. Use lots of colors!",
        ),
    ),
    "book_about": (
        S(
            "It's about a dragon. | It's about a girl who can talk to animals. "
            "| It's about space. | It's a mystery. | It's about a boy and his {pet}. "
            "| It's about pirates. | It's about a magic school.",
            "That sounds like a great book!",
            "Ooh, that sounds exciting. I hope you enjoy it.",
            L(
                "That sounds fun. Do you like it so far?",
                S("Yes, it's really good! | Yes! | I love it.", "That's the best feeling!"),
                S(
                    "It's okay. | Kind of. | Not really.",
                    "That's okay. Not every book is a favorite.",
                ),
            ),
        ),
    ),
    "winning": (
        S("Yes! | Yes, I am. | I think so!", "Great job! Have fun.", "Woohoo! Keep it up."),
        S(
            "No. | Not really. | I keep losing.",
            "That's okay. Playing is the fun part. You'll get better.",
            "Don't give up! Every game is a chance to learn.",
        ),
    ),
    "pet_name": (
        S(
            "Her name is {pname}. | His name is {pname}. | It's called {pname}. | {Pname}. "
            "| We call her {pname}. | We named him {pname}. | Its name is {pname}.",
            "{Pname} is a lovely name for a {pet}!",
            "{Pname}! What a great name.",
            L(
                "{Pname} is such a cute name. What does {pname} like to do?",
                S(
                    "Sleep a lot. | Play all day. | Eat snacks. | Cuddle with me. "
                    "| Chase things around. | Hide in boxes.",
                    "That sounds adorable. {Pname} seems like a happy {pet}.",
                    "Ha! {Pname} sounds like a lot of fun.",
                ),
            ),
            L(
                "I like that name. How long have you had {pname}?",
                S(
                    "A few years. | Just a few weeks. | Since I was little. | About a year.",
                    "That's lovely. You and {pname} must be a great team.",
                ),
            ),
        ),
        S(
            "I haven't named it yet. | It doesn't have a name yet. | We haven't picked a name.",
            L("Ooh, fun! Would you like some name ideas?", ask="name_ideas"),
        ),
    ),
    "name_ideas": (
        S("Yes, please! | Yes! | Sure. | Okay.", L("@idea:pet_name")),
        S(
            "No thanks. | No, I want to pick myself. | That's okay.",
            "That's fair. Picking the name yourself is the fun part.",
        ),
    ),
    "fav_food": (S(FAV_SAYS["food"], *FAV_REPLIES["food"]), _DONT_KNOW_FAV),
    "fav_color": (S(FAV_SAYS["color"], *FAV_REPLIES["color"]), _DONT_KNOW_FAV),
    "fav_animal": (S(FAV_SAYS["animal"], *FAV_REPLIES["animal"]), _DONT_KNOW_FAV),
    "fav_game": (S(FAV_SAYS["game"], *FAV_REPLIES["game"]), _DONT_KNOW_FAV),
    "fav_sport": (S(FAV_SAYS["sport"], *FAV_REPLIES["sport"]), _DONT_KNOW_FAV),
    "joke_or_fact": (
        S("A joke! | A joke, please. | Joke! | Tell me a joke.", L("@joke")),
        S("A fun fact. | A fact, please. | Fact! | A fun fact, please.", L("@fact")),
        S("Both! | Both, please.", L("@joke")),
        S(
            "Neither. | No thanks. | Not right now.",
            L("That's okay. What would you like to do instead?", ask="topic"),
        ),
    ),
    "more_joke": (
        S("Yes! | Yes, please! | Sure! | Okay. | Yes, tell me one. | Yeah!", L("@joke")),
        S(
            "No thanks. | No, that's okay. | Not now. | Maybe later.",
            "Okay! Just ask if you want one later.",
            L("That's fine. Is there anything else you'd like to talk about?", ask="topic"),
        ),
    ),
    "more_fact": (
        S("Yes! | Yes, please. | Sure. | Another one! | Okay.", L("@fact")),
        S(
            "No thanks. | That's enough. | Not now.",
            "Okay! I hope you liked that one.",
            L("All right. What else would you like to talk about?", ask="topic"),
        ),
    ),
    "more_about": (
        S("Yes! | Yes, please. | Tell me more. | Sure.", L("@more_about")),
        S(
            "No thanks. | That's enough. | No, that's okay.",
            "Okay! I hope that was interesting.",
            L("Sure. What else would you like to know?", ask="topic"),
        ),
    ),
    "idea_ok": (
        S(
            "Yes! | That sounds fun. | Good idea! | Okay, I'll try that. | I like it. "
            "| Yes, I like that.",
            "Great! Have fun.",
            "Yay! I hope you enjoy it.",
            "Awesome. Tell me how it goes if you like.",
        ),
        S("No. | Not really. | I don't like that one. | Hmm, no. | Nah.", L("@idea")),
        S(
            "Maybe. | Maybe. I'll think about it. | I'm not sure.",
            "That's fine. Take your time and choose what sounds best.",
            L("@idea"),
        ),
    ),
    "more_tip": (
        S("Yes, please. | Yes! | Sure. | Okay.", L("@tip")),
        S(
            "No, thanks. | That's okay. | No, that helps.",
            C("Okay. I hope it goes well.", "You're welcome. I hope that helps.", ctx="comfort"),
        ),
    ),
    "plans": (
        S(
            "I'm going to play outside. | Play outside, I think.",
            "That sounds fun. Have a great time!",
            "Playing outside is the best.",
        ),
        S(
            "I'm going to watch a movie. | Watch a movie.",
            L(
                "Fun! What kind of movies do you like?",
                S(
                    "Funny ones. | Cartoons. | Movies about animals. | Adventure movies. "
                    "| Movies about space.",
                    "Those sound great. Enjoy your movie!",
                ),
            ),
        ),
        S(
            "Nothing much. | I don't know yet. | No plans.",
            L("A free day! Would you like an idea for something to do?", ask="idea_do"),
        ),
        S(
            "I'm going to visit my {relative}. | We're visiting my {relative}.",
            "That sounds lovely. I hope you have a great time with your {relative}.",
        ),
        S(
            "I have practice later. | I have a game later. | I have swim practice.",
            "Have fun at practice! Every practice makes you a little better.",
        ),
    ),
    "idea_do": (
        S("Yes, please! | Sure! | Yes! | Okay.", L("@idea:do")),
        S("No thanks. | I'll figure it out. | Not really.", "Okay! I hope you have a fun day."),
    ),
}

_SLEPT = (
    S(
        "Yes! | Yes, really well. | Pretty well. | Yes, thanks.",
        "That's great! A good sleep makes the whole day better.",
        "Wonderful. You must be full of energy, then!",
    ),
    S(
        "No, not really. | Not very well. | No, I kept waking up. | Not great.",
        L("Oh no, I'm sorry. Would you like a tip for sleeping better tonight?", ask="sleep_tip"),
        C("I'm sorry. Maybe you can rest a little later today.", ctx="comfort"),
    ),
)

# --- openers -----------------------------------------------------------------------------------

OPENERS = (
    S(
        "Hi! | Hello! | Hey! | Hi there! | Hello there! | Hey there! | Hiya! | Howdy! | Hi. "
        "| Hello. | Hey, hi! | Heya! | Hi hi! | Hello, AI! | Hi, AI. | Hey you!",
        C(
            L("Hi! How are you today?", ask="how"),
            L("Hello! It's nice to chat with you. How are you?", ask="how"),
            L("Hey there! What would you like to talk about?", ask="topic"),
            L("Hi! I'm happy you're here. What's on your mind?", ask="topic"),
            L("Hello! How is your day going?", ask="how"),
            L("Hi there! What can I do for you today?", ask="topic"),
            L("Hello! I'm a small AI, and I love to chat. What shall we talk about?", ask="topic"),
            L("Hey! How are things with you?", ask="how"),
            "Hi! I'm ready to chat whenever you are.",
            L("Hi! What's new with you?", ask="up_to"),
            L("Hello, hello! What have you been up to?", ask="up_to"),
            L("Hey! It's nice to hear from you. How are you feeling today?", ask="how"),
            "Hi there! I'm here and happy to chat.",
            L("Hello! I hope you're having a nice day. What's up?", ask="up_to"),
            L("Hi! Would you like to hear a joke or a fun fact?", ask="joke_or_fact"),
            L(
                "Hey there! I'm all ears. Well, I don't have ears, but I'm ready to read. "
                "What's up?",
                ask="up_to",
            ),
            "Hello! It's nice to hear from you.",
            "Hi there! I'm happy to chat with you today.",
            "Hey! I'm here whenever you're ready.",
            "Hello! I'm glad you stopped by.",
            "Hi! It's a good day for a chat.",
            "Hello there! I'm ready to chat, answer questions or tell a joke.",
            ctx="greet",
        ),
    ),
    S(
        "Good morning! | Morning! | Good morning to you! | Good morning, AI! | Morning, how are you?",
        C(
            L("Good morning! I hope your day is off to a nice start. How are you?", ask="how"),
            L("Good morning! Did you sleep well?", *_SLEPT),
            L("Morning! What are you up to today?", ask="plans"),
            L("Good morning to you too! How are you feeling?", ask="how"),
            "Good morning! I'm happy to chat with you today.",
            "Good morning! I hope you have a lovely day.",
            "Morning! I hope your day is off to a good start.",
            ctx="greet",
        ),
    ),
    S(
        "Good afternoon! | Hi, good afternoon. | Good afternoon, AI.",
        C(
            L("Good afternoon! How has your day been so far?", ask="how_day"),
            L("Good afternoon to you too! What are you up to?", ask="up_to"),
            L("Hello, and good afternoon! What would you like to chat about?", ask="topic"),
            "Good afternoon! I hope your day is going well.",
            "Good afternoon to you too! I'm happy to chat.",
            ctx="greet",
        ),
    ),
    S(
        "Good evening! | Evening! | Hi, good evening. | Good evening, AI.",
        C(
            L("Good evening! How was your day?", ask="how_day"),
            L("Good evening! What would you like to chat about tonight?", ask="topic"),
            L("Evening! I hope you had a nice day. Did anything fun happen?", ask="best_part"),
            "Good evening! I hope you had a nice day.",
            "Good evening to you too! It's nice to chat.",
            ctx="greet",
        ),
    ),
    S(
        "How are you? | How are you doing? | How are you today? | Hi, how are you? "
        "| Hello! How are you? | Hey, how are you doing? | How's it going? | How's everything? "
        "| How are things? | Are you doing okay? | How have you been? | Hi! How's your day?",
        C(
            L("I'm doing well, thanks for asking! How about you?", ask="how"),
            L(
                "I'm good! I'm an AI, so I don't have bad days, but I like chatting. How are you?",
                ask="how",
            ),
            L("I'm great, thank you. How are you doing?", ask="how"),
            L("Pretty good! I'm ready to chat. How about you?", ask="how"),
            L("I'm doing fine. Thanks for asking! How is your day?", ask="how"),
            L("I'm happy to be chatting with you. How are you?", ask="how"),
            L("All good here! What about you?", ask="how"),
            L("I'm well, thank you. What's new with you?", ask="up_to"),
            L("I'm good, thanks! It's nice of you to ask. How are you feeling today?", ask="how"),
            L(
                "I'm an AI, so I'm always about the same, but I'm happy to talk. How are you?",
                ask="how",
            ),
            "I'm doing well, thanks for asking! I'm always happy to chat.",
            "I'm good, thank you! It's nice to hear from you.",
            "All good here, thanks! I'm ready whenever you want to talk.",
            ctx="greet",
        ),
    ),
    S(
        "What's up? | Hey, what's up? | What's new? | What are you up to? | Hi! What's up? "
        "| What are you doing?",
        C(
            L("Not much! I'm just here, ready to chat. What's up with you?", ask="up_to"),
            L("Just waiting for someone to talk to, and now you're here! What's new?", ask="up_to"),
            L(
                "Nothing much. I'm an AI, so I'm always right here. What are you up to?",
                ask="up_to",
            ),
            L("Not a lot! What would you like to talk about?", ask="topic"),
            L("Just chatting with you! How is your day going?", ask="how"),
            "Not much! I'm just here and ready to chat.",
            "Just hanging out and waiting for a chat. I'm glad you're here!",
            ctx="greet",
        ),
    ),
    S(
        "Nice to meet you! | Hi! Nice to meet you. | Hello, nice to meet you. "
        "| Pleased to meet you! | Hi, nice to meet you!",
        C(
            L("Nice to meet you too! What would you like to talk about?", ask="topic"),
            L("It's nice to meet you as well! What's your name?", ask="user_name"),
            L("Hello! Nice to meet you too. How are you today?", ask="how"),
            "Nice to meet you! I'm a small AI, and I love to chat.",
            "Nice to meet you too! I'm happy you're here.",
            ctx="greet",
        ),
    ),
    S(
        "Hi, I'm {uname}. | Hello! My name is {uname}. | Hey, I'm {uname}. "
        "| Hi there, my name is {uname}. | Hello, I'm {uname}! | Hi! I'm {uname}. Nice to meet you.",
        C(
            L("Hi, {name}! Nice to meet you. How are you today?", ask="how"),
            L(
                "Hello, {name}! It's nice to meet you. What would you like to talk about?",
                ask="topic",
            ),
            L("Hi, {name}! That's a lovely name. How is your day going?", ask="how"),
            L("Nice to meet you, {name}! What's on your mind?", ask="topic"),
            "Hey, {name}! I'm happy to chat with you.",
            L("Hello, {name}! What have you been up to today?", ask="up_to"),
            ctx="greet",
        ),
    ),
    S(
        "Hi again! | Hello again! | I'm back! | Hey, it's me again. | Hi, I'm back.",
        C(
            L(
                "Hello again! I don't remember our last chat, but I'm happy to talk now. "
                "How are you?",
                ask="how",
            ),
            L("Welcome back! What would you like to talk about today?", ask="topic"),
            L(
                "Hi again! I can't remember earlier chats, but I'm glad you're here. What's up?",
                ask="up_to",
            ),
            ctx="greet",
        ),
    ),
    S(
        "Are you there? | Hello? Is anyone there? | Is anybody there? | Hello? Are you there? "
        "| Are you awake?",
        C(
            L("Yes, I'm here! What would you like to talk about?", ask="topic"),
            L("I'm here! How can I help?", ask="topic"),
            "Hello! Yes, I'm here and ready to chat.",
            "I'm right here! Ready when you are.",
            L("Yes! I'm always here when you want to talk. What's up?", ask="up_to"),
            ctx="greet",
        ),
    ),
    S(
        "Can you hear me? | Can you hear me okay?",
        C(
            L(
                "I can't hear sounds, but I can read what you type. I'm here! What's up?",
                ask="up_to",
            ),
            L(
                "I can't hear anything, but I can read your words just fine. How can I help?",
                ask="topic",
            ),
            ctx="greet",
        ),
    ),
    S(
        "Can we talk? | Can we chat for a bit? | I want to talk. | Do you want to chat? "
        "| Can I talk to you? | Let's chat! | Will you talk with me?",
        C(
            L("Of course! What would you like to talk about?", ask="topic"),
            L("Sure! I love to chat. What's on your mind?", ask="topic"),
            L("Yes, let's talk! How are you doing today?", ask="how"),
            L("I'd like that. What shall we talk about?", ask="topic"),
            "Of course! I'm always happy to chat.",
            "Sure! Chatting is my favorite thing to do.",
            ctx="greet",
        ),
    ),
    S(
        "Can I ask you something? | I have a question. | Can I ask you a question? "
        "| I want to ask you something.",
        C(
            L("Of course! Go ahead and ask.", ask="question"),
            L("Sure! What would you like to know?", ask="question"),
            L("Yes, ask away. I'll do my best to help.", ask="question"),
            ctx="greet",
        ),
    ),
)

ASK_STEPS["user_name"] = (
    S(
        "I'm {uname}. | My name is {uname}. | {Uname}. | It's {uname}. | Call me {uname}. "
        "| My name's {uname}.",
        "Nice to meet you, {name}!",
        L("Hi, {name}! That's a lovely name. How are you today?", ask="how"),
        L("Nice to meet you, {name}! What would you like to talk about?", ask="topic"),
        "Hello, {name}! I'm happy to chat with you.",
    ),
    S(
        "I'd rather not say. | That's a secret. | I don't want to say.",
        "That's okay! You don't have to tell me.",
        L("No problem. What would you like to talk about?", ask="topic"),
    ),
)
ASK_STEPS["sleep_tip"] = (
    S("Yes, please. | Yes! | Sure. | Okay.", L("@tip:sleep")),
    S("No, thanks. | That's okay.", C("Okay. I hope you sleep well tonight.", ctx="comfort")),
)


# --- feelings ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Feeling:
    """A feeling: how the person says it, the replies when they give no reason (a line that
    asks why has ``ask="reason"``), and its reasons, each a step whose ``users`` are first-person
    clauses ("my {pet} is sick") and whose replies fit that reason."""

    words: tuple[str, ...]
    plain: tuple[Line, ...]
    reasons: tuple[Step, ...]


def R(says: str, *replies: "str | Line") -> Step:
    """A reason for a feeling and the replies that fit it; the replies comfort by default."""
    lines = [r if isinstance(r, Line) else Line(r, ctx="comfort") for r in replies]
    return S(says, *lines)


def _why(text: str) -> Line:
    return L(text, ask="reason")


FEELINGS: dict[str, Feeling] = {
    "happy": Feeling(
        ("happy", "really happy", "so happy", "super happy", "very happy", "pretty happy"),
        (
            _why("That's great to hear{n}! What made you so happy?"),
            _why("Yay! I love that. What happened?"),
            L("That's wonderful. Happy days are the best."),
            _why("I'm glad you're happy! Is something fun going on?"),
            _why("Hooray! Do you want to tell me why?"),
        ),
        (
            R(
                "it's the weekend | there's no school tomorrow",
                L("Weekends are great! Do you have anything fun planned?", ask="plans"),
                L("Hooray for the weekend! Time to rest and play."),
            ),
            R(
                "I got a good grade on my {subject} test | I did well on my {subject} test",
                L("Well done! All your practice paid off."),
                L("That's great news! You should be proud of yourself."),
            ),
            R(
                "my {relative} is visiting | my {relative} is staying with us",
                L(
                    "That sounds lovely! What do you like to do with your {relative}?",
                    S(
                        "We play games. | We bake cookies. | We go to the park. "
                        "| We read stories together. | We go for walks.",
                        "That sounds like so much fun.",
                        "How lovely. Enjoy your time together!",
                    ),
                ),
                L("How nice! Have a wonderful time with your {relative}."),
            ),
            R("I finished all my homework", L("Nice work! Now you have time to relax.")),
            R(
                "we're going to the beach tomorrow | we're going to the beach this weekend",
                L(
                    "The beach! That sounds amazing. Will you build a sandcastle?",
                    S("Yes! | Maybe! | Of course!", "Make it a big one!"),
                    S("No, I want to swim. | Probably not.", "Swimming sounds great too!"),
                ),
                L("How exciting! I hope you have a wonderful time."),
            ),
            R(
                "I got a new {gift} | my {relative} gave me a {gift}",
                L(
                    "A new {gift}! That's exciting. Have you tried it yet?",
                    S("Yes, it's great! | Yes! | A little bit.", "Yay! Have lots of fun with it."),
                    S("Not yet. | No, not yet.", "Something to look forward to, then!"),
                ),
                L("Lucky you! I hope you have lots of fun with your new {gift}."),
            ),
            R(
                "my team won our game | we won our game today",
                L("Congratulations! Winning as a team feels great."),
                L("Woohoo! That's awesome. Well played!"),
            ),
            R(
                "school is out for the summer | summer break just started",
                L(
                    "Summer break! That's so exciting. What do you want to do first?",
                    S(
                        "Go swimming. | Sleep in. | Play outside all day. | Go to the beach. "
                        "| Read lots of books.",
                        "That sounds like a perfect start to summer!",
                    ),
                ),
            ),
            R(
                "I learned to ride my bike",
                L("Wow, well done! Learning to ride a bike takes lots of practice."),
            ),
            R(
                "it's sunny outside | the sun is out",
                L("Sunny days are so nice. I hope you get to play outside!"),
            ),
            R(
                "my {pet} learned a new trick",
                L(
                    "That's amazing! What trick can your {pet} do now?",
                    S(
                        "It can sit. | It can roll over. | It can spin in a circle. "
                        "| It can give a high five.",
                        "So clever! You must be a great teacher.",
                    ),
                ),
            ),
            R(
                "I made a new buddy at school | I met a nice kid at the park",
                L("That's wonderful! It's so nice to meet someone new."),
            ),
        ),
    ),
    "sad": Feeling(
        ("sad", "a bit sad", "really sad", "kind of sad", "a little down", "down", "upset"),
        (
            _why("I'm sorry you feel sad{n}. Do you want to tell me what happened?"),
            _why("Oh no. I'm here for you. What's wrong?"),
            _why("I'm sorry you're feeling down. Would you like to talk about it?"),
            L(
                "That's okay. Everyone feels sad sometimes. I'm here if you want to talk.",
                ctx="comfort",
            ),
            L("I'm sorry. Would a joke help cheer you up a little?", ask="more_joke"),
        ),
        (
            R(
                "my {pet} is sick",
                "I'm so sorry your {pet} is sick. It's kind of you to care so much. "
                "I hope your {pet} feels better soon.",
                "Oh no. A grown-up can take your {pet} to the vet. I hope your {pet} gets "
                "better soon.",
            ),
            R(
                "my best buddy moved away | my cousin moved far away",
                "That's really sad. Moving away doesn't have to mean goodbye, though. You can "
                "still call, write or send pictures.",
                "I'm sorry. Missing someone is tough. Maybe you could send them a letter.",
            ),
            R(
                "I miss my {relative}",
                "It's normal to miss people we love. Maybe you could call your {relative} or "
                "draw a picture for them.",
                "I'm sorry. Missing someone shows how much you care about them.",
            ),
            R(
                "I lost my {toy} | I can't find my {toy}",
                L(
                    "Oh no, I'm sorry. Have you looked under your bed and in your bag?",
                    S(
                        "Yes, everywhere. | I looked everywhere.",
                        C(
                            "I'm sorry. Lost things often turn up when we stop looking. "
                            "I hope it turns up soon.",
                            ctx="comfort",
                        ),
                    ),
                    S("No, not yet. | Not there. I'll look.", "Good idea! I hope you find it."),
                ),
                "Losing something you love is no fun. I hope it turns up soon.",
            ),
            R(
                "nobody played with me at recess | nobody wanted to play with me",
                "I'm sorry, that sounds lonely. Maybe tomorrow you could ask someone to play a "
                "game with you.",
                "That must have felt bad. You deserve kind people to play with. Telling a "
                "teacher or someone you trust can help.",
            ),
            R(
                "my team lost the game | we lost our game",
                "I'm sorry. Losing feels bad, but every game helps you get better.",
                "That's tough. You still played, and that matters.",
            ),
            R(
                "it's raining and I can't play outside",
                L(
                    "Rainy days can be a bit gloomy. Would you like an idea for something fun to "
                    "do inside?",
                    ask="idea_do",
                ),
                "I'm sorry. Maybe the sun will be out tomorrow.",
            ),
            R(
                "my ice cream fell on the ground",
                "Oh no! That's such a sad moment. I hope you get another one soon.",
            ),
            R(
                "someone was mean to me | a kid at school was mean to me",
                "I'm sorry someone was mean to you. That's not okay. It can help to tell a "
                "grown-up you trust about it.",
                "That sounds really hurtful. You deserve kindness. Please talk to someone you "
                "trust about what happened.",
            ),
            R(
                "my drawing got ruined | my sandcastle got knocked down",
                "Oh no, I'm sorry. Maybe you can make a new one, and it might be even better.",
            ),
        ),
    ),
    "tired": Feeling(
        ("tired", "so tired", "sleepy", "very sleepy", "worn out", "a bit tired"),
        (
            L(
                "Oh no. Maybe you could rest for a little while. Did you sleep well last night?",
                *_SLEPT,
            ),
            L("Being tired is no fun. A little rest might help.", ctx="comfort"),
            _why("I'm sorry you're tired. What made you so tired?"),
            L(
                "Maybe it's a good day to take it easy. Would you like an idea for something "
                "calm to do?",
                ask="idea_do",
            ),
        ),
        (
            R(
                "I didn't sleep well | I couldn't sleep last night",
                L(
                    "I'm sorry. Would you like some tips for sleeping well tonight?",
                    ask="sleep_tip",
                ),
                "Bad nights happen. A calm evening and a cozy bed tonight might help.",
            ),
            R(
                "I stayed up late",
                "Staying up late can make the next day feel long. Maybe tonight you can go to "
                "bed a bit sooner.",
            ),
            R("I had a long day at school", "Long days are tiring. You've earned a rest!"),
            R(
                "I played outside all day",
                L("That sounds like a fun way to get tired! Rest well tonight."),
            ),
            R(
                "I had swim practice | I had a big game today",
                L("That really wears you out. Great job getting out there!"),
            ),
            R("my {sibling} woke me up", "Oh no! Maybe you can rest a little later today."),
        ),
    ),
    "bored": Feeling(
        ("bored", "so bored", "really bored", "a bit bored", "super bored"),
        (
            L(
                "Let's fix that! Would you like a joke, a fun fact or an idea for something to do?",
                ask="bored_pick",
            ),
            L("Being bored can be the start of something fun. Want an idea?", ask="idea_do"),
            L("Oh no! Do you want to hear a joke?", ask="more_joke"),
            L(
                "I can tell a joke, share a fact or suggest a game. Which would you like?",
                ask="bored_pick",
            ),
        ),
        (
            R(
                "there's nothing to do",
                L("Let's find something! Would you like an idea?", ask="idea_do"),
            ),
            R(
                "it's raining",
                L(
                    "Rainy days are great for indoor fun. Would you like an idea for something to "
                    "do inside?",
                    ask="idea_do",
                ),
            ),
            R(
                "my friends are busy | everyone is busy",
                L(
                    "That happens sometimes. I'm here to chat! Want to hear a joke?",
                    ask="more_joke",
                ),
            ),
            R(
                "I finished all my games | I've played all my games",
                L("Time for something new! Would you like an idea?", ask="idea_do"),
            ),
            R(
                "we're on a long car ride | this car ride is so long",
                L(
                    "Long car rides can feel endless. You could play I spy. Want to hear a joke "
                    "too?",
                    ask="more_joke",
                ),
            ),
        ),
    ),
    "angry": Feeling(
        ("angry", "really angry", "mad", "so mad", "annoyed", "grumpy", "really annoyed"),
        (
            _why("I'm sorry you feel angry. Do you want to tell me what happened?"),
            _why("That sounds frustrating. What happened?"),
            L(
                "It's okay to feel angry sometimes. Taking a few slow, deep breaths can help you "
                "feel calmer.",
                ctx="comfort",
            ),
            L(
                "Feeling mad is normal. Sometimes it helps to take a little break and do "
                "something calm.",
                ctx="comfort",
            ),
        ),
        (
            R(
                "my {sibling} broke my {toy}",
                "That's really frustrating. Maybe you can tell a grown-up, so they can help sort "
                "it out.",
                "I'm sorry. It's okay to feel mad when someone breaks your things. Try to take a "
                "breath before you talk about it.",
            ),
            R(
                "my {sibling} took my stuff | my {sibling} keeps bugging me",
                "That's annoying. It might help to ask calmly, or to ask a grown-up for help.",
            ),
            R(
                "I lost a game | I keep losing at {game}",
                "Losing can feel really annoying. Everyone loses sometimes, even the best players.",
            ),
            R(
                "someone cut in line",
                "That doesn't feel fair. It's okay to say, kindly, that you were there first.",
            ),
            R(
                "my homework is too difficult",
                "That sounds frustrating. Try a small piece at a time, and ask a teacher or a "
                "grown-up to help with the tricky parts.",
            ),
            R(
                "I wasn't allowed to stay up late",
                "That's disappointing. Grown-ups usually say no because they want you to feel "
                "good tomorrow.",
            ),
            R(
                "my game froze and I lost my progress",
                "Oh no, that's so frustrating! It might help to take a short break before you "
                "try again.",
            ),
        ),
    ),
    "scared": Feeling(
        ("scared", "a bit scared", "really scared", "frightened", "afraid", "a little scared"),
        (
            _why("I'm sorry you feel scared. What's scaring you?"),
            _why("It's okay to be scared. Can you tell me what's wrong?"),
            L("Being scared is no fun. It can help to be near someone you trust.", ctx="comfort"),
        ),
        (
            R(
                "there's a big storm outside | the thunder is so loud",
                "Storms can be loud and scary. You're safe inside. Maybe stay close to a "
                "grown-up until it passes.",
                "Thunder is loud, but it can't hurt you while you're inside. Counting slowly can "
                "help you feel calmer.",
            ),
            R(
                "I heard a strange noise",
                "Strange noises can be scary. Most of the time it's just the house or the wind. "
                "Telling a grown-up can help you feel safe.",
            ),
            R(
                "I had a bad dream | I had a nightmare",
                "Bad dreams can feel very real, but they're not real. It might help to tell a "
                "grown-up about it, or think about something happy.",
                "I'm sorry. Dreams can't hurt you. Maybe a cuddle with a soft toy will help.",
            ),
            R(
                "there's a spider in my room",
                "Spiders can be surprising! Most of them are harmless. A grown-up can help move "
                "it outside.",
            ),
            R(
                "I have to get a shot",
                "Shots can feel scary, but they're over really quickly. You could squeeze a "
                "grown-up's hand and count to ten.",
            ),
        ),
    ),
    "nervous": Feeling(
        ("nervous", "a bit nervous", "really nervous", "kind of nervous", "so nervous"),
        (
            _why("Feeling nervous is normal. What are you nervous about?"),
            _why("Oh no. Do you want to tell me what's making you nervous?"),
            L(
                "Nerves are normal. Taking slow, deep breaths can help you feel calmer.",
                ctx="comfort",
            ),
        ),
        (
            R(
                "I have a test tomorrow | I have a big test tomorrow",
                L(
                    "Tests can feel scary, but you can do it! Would you like a tip for the test?",
                    ask="test_tip",
                ),
                "It's normal to feel nervous before a test. Go over the main parts once more, "
                "then get some rest.",
            ),
            R(
                "it's my first day at a new school | I'm starting at a new school",
                "A new school is a big change. Lots of kids feel nervous on their first day. "
                "Saying hi to someone is a great start.",
                "That's exciting and a little scary. Try to smile and say hello. You might meet "
                "some really nice people.",
            ),
            R(
                "I have a {instrument} concert | I have to play the {instrument} at a concert",
                "You've practiced, so you're ready! Take a deep breath before you start. You'll "
                "do great.",
            ),
            R(
                "I have to give a talk in class",
                "Talking to a class can be scary. Practicing a few times at home helps a lot. "
                "You can do it!",
            ),
            R(
                "I have a big game tomorrow",
                "Good luck! Feeling nervous means you care. Just try your best and have fun.",
            ),
            R(
                "I have a checkup tomorrow",
                "Checkups help make sure you're healthy. You can ask a grown-up to stay with "
                "you. You'll be okay.",
            ),
            R(
                "I'm going to a sleepover for the first time",
                "That's a big adventure! It's normal to feel nervous. You can bring something "
                "cozy from home.",
            ),
        ),
    ),
    "excited": Feeling(
        ("excited", "so excited", "really excited", "super excited", "very excited"),
        (
            _why("Ooh, how fun! What are you excited about?"),
            _why("Yay! Tell me what's happening!"),
            _why("I love that feeling. What's the big news?"),
        ),
        (
            R(
                "it's my birthday soon | my birthday is next week",
                L(
                    "Happy almost-birthday! Do you have anything fun planned?",
                    S(
                        "A party! | We're going to the zoo. | Just cake with my family. "
                        "| I don't know yet.",
                        "That sounds wonderful. I hope it's a great day!",
                    ),
                ),
            ),
            R(
                "we're going on vacation | we're going on a trip",
                L(
                    "That sounds amazing. Where are you going?",
                    S(
                        "To the beach. | To the mountains. | To see my {relative}. "
                        "| Camping in the woods. | To a big city.",
                        "That sounds wonderful! I hope you have the best time.",
                    ),
                ),
            ),
            R(
                "I'm getting a puppy | we're getting a puppy",
                L(
                    "A puppy! That's so exciting. Do you know what you'll name it?",
                    S(
                        "Not yet. | I don't know yet. | No, not yet.",
                        L("Would you like some name ideas?", ask="name_ideas"),
                    ),
                    S(
                        "I want to call it {pname}. | Maybe {pname}. | {Pname}!",
                        "{Pname} is a great name for a puppy!",
                    ),
                ),
            ),
            R(
                "my cousins are visiting | my cousins are staying over",
                L("How fun! It's great to spend time with family."),
            ),
            R(
                "it's almost the holidays",
                L(
                    "The holidays are such a fun time of year! What are you looking forward to?",
                    S(
                        "Presents! | Seeing my family. | The food. | No school!",
                        "That sounds lovely. I hope the holidays are wonderful.",
                    ),
                ),
            ),
            R(
                "we're going to the zoo tomorrow",
                L("The zoo! What animal do you want to see most?", ask="fav_animal"),
            ),
            R(
                "I'm going to a sleepover",
                L(
                    "A sleepover sounds so fun! Will you play games or watch a movie?",
                    S(
                        "Both! | Games. | A movie. | We'll stay up and talk.",
                        "That sounds like a great time. Have fun!",
                    ),
                ),
            ),
            R(
                "school is almost over | summer is almost here",
                L("Almost there! You've done so much this year."),
            ),
        ),
    ),
    "lonely": Feeling(
        ("lonely", "a bit lonely", "kind of lonely", "really lonely"),
        (
            L(
                "I'm sorry you feel lonely. I'm here, and I'm happy to chat with you.",
                ctx="comfort",
            ),
            _why("That's a tough feeling. Do you want to tell me why you feel lonely?"),
            L("I'm glad you came to talk to me. What would you like to chat about?", ask="topic"),
        ),
        (
            R(
                "my friends are away | my friends are on vacation",
                "I'm sorry. They'll be back soon. Until then, I'm happy to chat with you.",
            ),
            R(
                "I don't have anyone to play with",
                "That can feel lonely. Maybe you could ask someone in your family to play a "
                "game. I'm here to chat too!",
            ),
            R(
                "I just moved to a new town | we just moved here",
                "Moving is a big change. It takes time to meet people, but you will. Clubs and "
                "teams are a great place to start.",
            ),
            R(
                "everyone in my family is busy",
                "Grown-ups get busy sometimes. Maybe later you can do something fun together. "
                "I'm here to chat for now.",
            ),
        ),
    ),
    "sick": Feeling(
        ("sick", "a bit sick", "unwell", "under the weather", "poorly"),
        (
            L(
                "I'm sorry you feel sick. Make sure to tell a grown-up how you feel, so they can "
                "help.",
                ctx="comfort",
            ),
            L(
                "Oh no. Rest is really important when you're sick. I hope you feel better soon.",
                ctx="comfort",
            ),
            _why("I'm sorry to hear that. What's wrong?"),
        ),
        (
            R(
                "I have a cold",
                "Colds are no fun. Rest, drink water and tell a grown-up how you feel. I hope "
                "you feel better soon.",
            ),
            R(
                "my tummy hurts",
                "I'm sorry your tummy hurts. Please tell a grown-up, so they can help you feel "
                "better.",
            ),
            R(
                "I have a headache",
                "I'm sorry. Resting somewhere quiet might help. Be sure to tell a grown-up how "
                "you feel.",
            ),
            R("I have a sore throat", "Ouch. Please tell a grown-up, and try to rest your voice."),
            R(
                "I have a fever",
                "I'm sorry. A grown-up should know about a fever right away. Rest and feel "
                "better soon.",
            ),
        ),
    ),
    "hungry": Feeling(
        ("hungry", "so hungry", "really hungry", "starving", "a bit hungry"),
        (
            L(
                "Maybe it's time for a snack! You could ask a grown-up for something healthy, "
                "like some fruit."
            ),
            L(
                "Oh no! Is it almost time for a meal?",
                S("Yes. | Yes, soon. | Almost.", "Great, it won't be long then!"),
                S(
                    "No. | Not for a while. | No, not yet.",
                    "Maybe a grown-up can help you find a small snack until then.",
                ),
            ),
            L("Would you like a snack idea?", ask="idea_snack"),
        ),
        (
            R(
                "I didn't have lunch | I skipped breakfast",
                L("That's a long time without food. Please ask a grown-up for something to eat."),
            ),
            R(
                "dinner isn't ready yet | lunch isn't ready yet",
                L("It'll be ready soon! It's usually worth the wait."),
            ),
            R(
                "I just got back from practice | I ran around all day",
                L("Moving around makes you hungry! A healthy snack and some water would be great."),
            ),
        ),
    ),
    "proud": Feeling(
        ("proud", "really proud", "so proud", "proud of myself"),
        (
            _why("That's great! What are you proud of?"),
            _why("You should be! Tell me what you did."),
            _why("I love that! What happened?"),
        ),
        (
            R("I finished a big puzzle", L("Well done! Big puzzles take lots of patience.")),
            R("I learned to swim", L("That's amazing! Swimming is a great thing to learn.")),
            R(
                "I got an A on my {subject} test",
                L("Wow, an A! All your studying paid off. Well done!"),
            ),
            R(
                "I helped make dinner",
                L("That's so kind and helpful. Cooking is a great skill to learn."),
            ),
            R(
                "I read a whole book by myself",
                L("That's wonderful! What was it about?", ask="book_about"),
            ),
            R("I won a race", L("Congratulations! All that practice paid off.")),
            R(
                "I was kind to a new kid at school",
                L("That's a really kind thing to do. You should be proud."),
            ),
            R("I tied my shoes by myself", L("Well done! That's a big skill to learn.")),
            R("I scored a goal", L("Wow, a goal! That must have felt amazing.")),
            R("I cleaned my whole room", L("Great job! A tidy room feels so nice.")),
        ),
    ),
    "confused": Feeling(
        ("confused", "a bit confused", "really confused", "mixed up"),
        (
            _why("That's okay. Everyone gets confused sometimes. What's confusing you?"),
            _why("Let's see if I can help. What are you confused about?"),
        ),
        (
            R(
                "my {subject} homework is tricky",
                "{Subject} can be tricky. Try reading the question slowly, and ask a teacher or "
                "a grown-up to explain the part you're stuck on.",
            ),
            R(
                "I don't understand fractions",
                L(
                    "Fractions are parts of a whole. If you cut a pizza into four equal slices, one "
                    "slice is one quarter of the pizza."
                ),
            ),
            R(
                "the rules of a new game are confusing",
                L(
                    "New rules can be confusing at first. Playing one practice round often helps "
                    "them make sense."
                ),
            ),
            R(
                "I don't know what to do",
                L(
                    "That's okay. Sometimes it helps to start with one small step. Would you like "
                    "an idea?",
                    ask="idea_do",
                ),
            ),
        ),
    ),
    "worried": Feeling(
        ("worried", "a bit worried", "really worried", "so worried"),
        (
            _why("I'm sorry you're worried. Do you want to tell me what's on your mind?"),
            L(
                "Worries can feel big. Talking to someone you trust can make them feel smaller.",
                ctx="comfort",
            ),
        ),
        (
            R(
                "my {relative} is in the hospital",
                "I'm so sorry. That sounds really scary. It can help to talk to someone you "
                "trust about how you feel.",
            ),
            R(
                "my {pet} is sick",
                "That's worrying. A vet can help your {pet}, and a grown-up can take care of "
                "that. I hope your {pet} feels better soon.",
            ),
            R(
                "I might not make the team",
                "Whatever happens, trying out took courage. You can always keep practicing.",
            ),
            R(
                "I forgot to do my homework",
                "It happens to everyone sometimes. Telling your teacher the truth is the best "
                "way to fix it.",
            ),
            R(
                "my mom is sick | my dad is sick",
                "I'm sorry they aren't feeling well. Talking to someone you trust can help. I "
                "hope they feel better soon.",
            ),
            R(
                "I'm not good at {subject}",
                "Lots of people find {subject} tricky. Every bit of practice helps, even if it "
                "doesn't feel like it.",
            ),
        ),
    ),
}

# Saying how one feels, with no reason and with one ("{w}" the feeling, "{r}" the reason).
FEEL_PLAIN = (
    "I'm {w}.", "I feel {w}.", "I'm feeling {w}.", "I'm feeling {w} today.", "I feel {w} today.",
    "Today I'm {w}.", "I'm {w} right now.", "I've been {w} all day.",
)  # fmt: skip
FEEL_REASON = (
    "I'm {w} because {r}.", "I feel {w} because {r}.", "I'm {w}. {R}.", "{R}, so I'm {w}.",
    "I'm feeling {w}. {R}.", "I'm {w} today because {r}.", "{R}. I'm {w}.",
)  # fmt: skip
# Giving the reason when the AI asks for it.
REASON_SAYS = ("{R}.", "Well, {r}.", "Because {r}.", "It's just that {r}.", "{R}. That's all.")
# Saying how one is when asked "How are you?".
GOOD_SAYS = _alts(
    "I'm good, thanks! | Good, thanks. | I'm fine. | Pretty good! | I'm great! | Not bad. "
    "| I'm okay. | I'm doing well, thanks. | Really good! | All good! | I'm doing great. "
    "| Fine, thank you. | Great, thanks! | I'm good."
)
GOOD_REPLIES = (
    L("Glad to hear it{n}! What have you been up to today?", ask="up_to"),
    L("That's great{n}! What would you like to talk about?", ask="topic"),
    L("Good to hear! Do you have any fun plans for today?", ask="plans"),
    L("Nice! What's new with you?", ask="up_to"),
    L("I'm happy to hear that. Is there anything I can help you with?", ask="topic"),
    L("Great! Want to hear a joke or a fun fact?", ask="joke_or_fact"),
    L("Wonderful{n}! What shall we chat about?", ask="topic"),
    L("That's great to hear{n}."),
    L("Nice! I'm glad your day is going well."),
    L("Glad to hear it! I'm happy to chat whenever you like."),
    L("Good! I hope the rest of your day is just as nice."),
)
ASK_BACK = (" How about you?", " And you?", " How are you?", " What about you?")
# When the person asks back, the AI says how it is, then one of these.
AI_STATE = (
    "I'm doing well, thanks! ", "I'm good too, thanks for asking! ", "I'm great, thank you! ",
    "All good here! ", "I'm fine, thanks. ",
)  # fmt: skip
GOOD_BACK = (
    L("I'm glad you're doing well{n}."),
    L("It's nice to hear you're doing well."),
    L("What would you like to talk about?", ask="topic"),
    L("What have you been up to today?", ask="up_to"),
    L("Do you have any fun plans for today?", ask="plans"),
    L("Would you like to hear a joke or a fun fact?", ask="joke_or_fact"),
    L("I'm happy to chat whenever you like."),
)
MEH_SAYS = _alts(
    "Not great. | Not so good. | Meh. | Could be better. | Not very good. | Not great, "
    "actually. | Kind of bad."
)
MEH_REPLIES = (
    _why("Oh no, I'm sorry. What's wrong?"),
    _why("I'm sorry to hear that. Do you want to tell me what happened?"),
    L("I'm sorry. I hope things get better soon. Would a joke help?", ask="more_joke"),
)
ASK_STEPS["bored_pick"] = (
    S("A joke! | A joke, please. | Joke!", L("@joke")),
    S("A fact. | A fun fact! | A fact, please.", L("@fact")),
    S("An idea. | A game. | Something to do.", L("@idea:do")),
)
ASK_STEPS["test_tip"] = (
    S("Yes, please. | Yes! | Sure.", L("@tip:test")),
    S("No, thanks. | I'm okay.", C("Okay. Good luck! You can do it.", ctx="comfort")),
)
ASK_STEPS["idea_snack"] = (
    S("Yes, please! | Sure. | Yes!", L("@idea:snack")),
    S("No thanks. | I'm okay.", "Okay! I hope you get something tasty soon."),
)


# --- things people say about themselves --------------------------------------------------------

_PET_DOES = S(
    "She likes to sleep in the sun. | He loves chasing his toys. | It likes to hide. "
    "| She loves cuddles. | He's really playful. | It's very lazy. | She's a bit shy.",
    "That sounds adorable!",
    "Aww. Every pet has its own little personality.",
    "Ha! That sounds like a very happy {pet}.",
)
_HOBBIES = S(
    "I like drawing. | Playing outside. | Reading books. | Playing {game}. "
    "| Building things with blocks. | I like {sport}. | Riding my scooter.",
    "Those sound like great ways to have fun!",
    "That sounds fun. It's good to have things you enjoy.",
    "Nice! I like hearing what people enjoy.",
)
_YES_NO_TOGETHER = (
    S(
        "Yes, all the time! | Yes. | Sometimes. | Yes, we play a lot.",
        "That's lovely. It's nice to have someone to play with at home.",
    ),
    S(
        "Not really. | No, we argue a lot. | Not much.",
        "That's okay. Brothers and sisters don't always get along, but they can still be "
        "a team sometimes.",
    ),
)


def _went(place: str, *replies: "str | Line") -> Step:
    """Saying one went to ``place`` today, in several ways, with the replies that fit."""
    says = " | ".join(
        f"I went to {p} today. | We went to {p} this morning. | Today I went to {p}. "
        f"| I just got back from {p}. | We went to {p} after school. "
        f"| I spent the day at {p}."
        for p in (p.strip() for p in place.split("|"))
    )
    return S(says, *replies)


SELF = (
    # pets
    S(
        "I have a {pet}. | We have a {pet} at home. | Guess what? I have a {pet}. "
        "| My family has a {pet}. | I have a {pet}. It's really cute. | I just got a {pet}!",
        L("A {pet}! That's lovely. What's its name?", ask="pet_name"),
        L("How fun! What's your {pet}'s name?", ask="pet_name"),
        L("I love hearing about pets. What is your {pet} like?", _PET_DOES),
        "Lucky you! A {pet} can be such good company.",
        L("Aww, a {pet}! Does it have a name?", ask="pet_name"),
    ),
    S(
        "I have a {pet} named {pname}. | My {pet}'s name is {pname}. | My {pet} is called {pname}. "
        "| I have a {pet}. Her name is {pname}. | I have a {pet}. His name is {pname}. "
        "| {Pname} is my {pet}.",
        "{Pname} is a great name for a {pet}!",
        L("{Pname} sounds adorable. What does {pname} like to do?", _PET_DOES),
        "Aww, {pname}! That's such a cute name.",
        L(
            "I like that name. How long have you had {pname}?",
            S(
                "A few years. | Just a few weeks. | Since I was little. | About a year.",
                "That's lovely. You and {pname} must be a great team.",
            ),
        ),
    ),
    S(
        "I want a pet. | I really want a {pet}. | I wish I had a {pet}. | Can I tell you a "
        "secret? I want a {pet}.",
        L(
            "A {pet} would be fun! Have you asked your family?",
            S(
                "Yes, they said maybe. | Yes, but they said no. | Not yet.",
                "Maybe one day! Taking care of a pet is a big job, so it's good to show you're "
                "ready.",
            ),
        ),
        "Pets can be wonderful. Taking care of one is a big job, but a happy one.",
    ),
    # favorites, said as a statement
    S(
        "I like {food}. | My favorite food is {food}. | I could eat {food} every day. "
        "| {Food} {fis} the best. | I really love {food}. | I had {food} for lunch today. "
        "| Do you know what I love? {Food}.",
        *FAV_REPLIES["food"],
    ),
    S(
        "My favorite color is {color}. | I like the color {color}. | {Color} is my favorite "
        "color. | I love the color {color}. | I like {color} the most.",
        *FAV_REPLIES["color"],
    ),
    S(
        "I like {animal}. | I love {animal}. | {Animal} are my favorite animals. "
        "| I think {animal} are so cool. | My favorite animals are {animal}.",
        *FAV_REPLIES["animal"],
    ),
    S(
        "I like playing {game}. | I love {game}. | My favorite game is {game}. "
        "| I played {game} today.",
        *FAV_REPLIES["game"],
    ),
    S(
        "I like {sport}. | I love {sport}. | {Sport} is my favorite. | I'm really into {sport}.",
        *FAV_REPLIES["sport"],
    ),
    # name and age
    SELF_NAME := S(
        "My name is {uname}. | I'm {uname}. | Call me {uname}. | You can call me {uname}. "
        "| My name's {uname}.",
        "Nice to meet you, {name}!",
        L("Hi, {name}! That's a lovely name. How are you today?", ask="how"),
        L("It's nice to meet you, {name}. What would you like to talk about?", ask="topic"),
        L("Hello, {name}! What have you been up to today?", ask="up_to"),
    ),
    S(
        "I am {age} years old. | I'm {age}. | I just turned {age}. | I'm {age} years old. "
        "| Guess what? I'm {age}.",
        "{Agew} is a great age!",
        L("{Agew}! That's a fun age. What do you like to do for fun?", _HOBBIES),
        "Wow, {agew}! You must know lots of things by now.",
        L("{Agew} is a wonderful age. What's your favorite thing to do?", _HOBBIES),
    ),
    # birthdays
    S(
        "It's my birthday! | Today is my birthday! | It's my birthday today. "
        "| Guess what? It's my birthday!",
        L("Happy birthday! I hope you have a wonderful day. How old are you now?", ask="age_now"),
        L(
            "Happy birthday! Are you doing anything special today?",
            S(
                "I'm having a party! | We're going out for pizza. | Just cake with my family. "
                "| We're going to the zoo.",
                "That sounds wonderful. I hope it's the best day!",
            ),
            S(
                "Not really. | No, just a normal day.",
                "A normal day can still be a happy one. Happy birthday!",
            ),
        ),
        "Happy birthday! I hope your day is full of fun and cake.",
        L(
            "Happy birthday! That's so exciting. What was your favorite present?",
            S(
                "A new {gift}. | I got a {gift}! | My {relative} gave me a {gift}.",
                "A {gift}! That sounds great. Have fun with it.",
            ),
        ),
    ),
    S(
        "My birthday is tomorrow. | It's my birthday next week. | My birthday is on {day}.",
        L(
            "How exciting! Do you have anything fun planned?",
            S(
                "A party! | We're going bowling. | A sleepover. | I don't know yet.",
                "That sounds like a lot of fun. Happy almost-birthday!",
            ),
        ),
        "Ooh, how exciting! I hope you have a wonderful birthday.",
    ),
    # school
    S(
        "I have a test tomorrow. | We have a test tomorrow. | I have a big test on {day}.",
        L("Good luck! What's the test about?", ask="hw_subject"),
        L("You can do it! Would you like a tip for the test?", ask="test_tip"),
        C("Good luck! A good night's sleep before a test really helps.", ctx="comfort"),
    ),
    S(
        "I have a {subject} test tomorrow. | We have a {subject} test on {day}. "
        "| I have a {subject} quiz tomorrow.",
        L("Good luck with your {subject} test! Would you like a tip?", ask="test_tip"),
        C(
            "You've got this! Go over the main parts of {subject} once more, then rest.",
            ctx="comfort",
        ),
        L(
            "{Subject}! Do you feel ready?",
            S("Yes, I studied. | I think so. | Pretty ready.", "Great! You'll do well."),
            S(
                "Not really. | No. | Kind of.",
                L("That's okay. Would you like a tip for studying?", ask="test_tip"),
            ),
        ),
    ),
    S(
        "I have so much homework. | I have a lot of homework today. | My homework is taking "
        "forever.",
        L(
            "That sounds like a lot. Would you like a tip for getting through it?",
            ask="more_tip",
            ctx="tip:focus",
        ),
        L("You can do it! What subject is it?", ask="hw_subject"),
    ),
    S(
        "My favorite subject is {subject}. | I like {subject} at school. | I'm good at {subject}.",
        "{Subject} is a great subject! It's nice to have one you enjoy.",
        L(
            "That's great! What do you like about {subject}?",
            S(
                "It's fun. | It's easy for me. | My teacher is nice. | I like learning new things.",
                "That's a great reason. Enjoying a subject makes learning easier.",
            ),
        ),
    ),
    # lost things
    S(
        "I lost my {toy}. | I can't find my {toy}. | My {toy} is missing.",
        L(
            "Oh no! Have you looked under your bed?",
            S(
                "Yes, it's not there. | I already looked.",
                "Try checking your bag, the car and under the couch cushions. Lost things love "
                "to hide there.",
            ),
            S("No! I'll look. | Not yet.", "Good luck! I hope it's there."),
        ),
        C(
            "I'm sorry. Losing something you love is no fun. I hope it turns up soon.",
            ctx="comfort",
        ),
        L(
            "Oh no. When did you last have it?",
            S(
                "This morning. | Yesterday. | At the park. | I don't remember.",
                "Try looking in the places you were around then. I hope you find it soon.",
            ),
        ),
    ),
    # things done today
    S(
        "I drew a picture today. | I made a drawing. | I painted a picture.",
        L("That's great! What did you draw?", ask="drawing_what"),
        "How lovely! Drawing is a wonderful thing to do.",
    ),
    S(
        "I learned to ride my bike! | I can ride my bike now! | I rode my bike without help.",
        "Wow, well done! That takes lots of practice. You should be proud.",
        L(
            "That's amazing! Was it tricky at first?",
            S(
                "Yes, I fell a lot. | Yes, a little. | Kind of.",
                "That's how everyone learns. You kept trying, and that's what matters.",
            ),
            S("No, it was easy! | Not really.", "You're a natural!"),
        ),
    ),
    S(
        "I finished my book. | I read a whole book today. | I just finished a really good book.",
        L("Well done! What was it about?", ask="book_about"),
        "That's great! Finishing a book feels so good.",
    ),
    S(
        "I made a cake with my {relative}. | We baked cookies today. | I helped bake bread.",
        "That sounds like so much fun! Baking together is lovely.",
        L(
            "Yum! Was it fun to make?",
            S("Yes! | So fun. | Yes, and messy!", "The messy part is half the fun!"),
        ),
    ),
    S(
        "I scored a goal today! | I won my game today! | We won our match!",
        "Woohoo! Congratulations! That must have felt amazing.",
        L(
            "Amazing! Do you play every week?",
            S("Yes! | Twice a week. | Most weeks.", "That's great. All that practice pays off!"),
        ),
    ),
    S(
        "I'm learning to play the {instrument}. | I started {instrument} lessons. "
        "| I play the {instrument}.",
        L(
            "That's wonderful! How long have you been playing?",
            S(
                "Just a few weeks. | A year. | A long time. | Since last summer.",
                "Keep it up! Music takes practice, and it's worth it.",
            ),
        ),
        "The {instrument} is a great choice. Playing music is a wonderful skill.",
        L(
            "How fun! Do you have a favorite song to play?",
            S(
                "Yes! | Not yet. | I like playing fun songs.",
                "That's great. Playing songs you love makes practice more fun.",
            ),
        ),
    ),
    S(
        "I'm learning to swim. | I'm taking swimming lessons. | I can swim now!",
        "That's great! Swimming is a really good thing to learn.",
        L(
            "Awesome! Do you like being in the water?",
            S("Yes, I love it! | Yes! | Mostly.", "That's wonderful. Have fun at your lessons!"),
            S(
                "Not really. | It's a bit scary.",
                "That's okay. It gets easier with practice, and your teacher will help.",
            ),
        ),
    ),
    S(
        "I planted a seed today. | We planted a garden. | I planted something in a pot.",
        L(
            "How lovely! Plants need water and sunlight to grow. What did you plant?",
            S(
                "A sunflower. | Tomatoes. | Beans. | Some flowers. | A little tree.",
                "That sounds great. It's so fun to watch something grow.",
            ),
        ),
        "That's wonderful! Watching plants grow is so much fun.",
    ),
    S(
        "I'm growing a sunflower. | I'm growing tomatoes. | I'm growing beans in a cup.",
        "How lovely! Plants need water and sunlight to grow.",
        "That's wonderful! It's so fun to watch something grow a little every day.",
    ),
    S(
        "I built a sandcastle. | I built a fort out of blankets. | I made a tower out of blocks.",
        "That sounds awesome! Building things is so much fun.",
        L(
            "Cool! How big was it?",
            S("Really big! | Pretty big. | Taller than me!", "Wow! That must have been great."),
            S("Small. | Just a little one.", "Small ones are great too!"),
        ),
    ),
    S(
        "I'm writing a story. | I'm making a comic. | I wrote a poem today.",
        L(
            "How creative! What's it about?",
            S(
                "A dragon. | A clever cat. | Space explorers. | A magic tree. | My {pet}.",
                "That sounds like a wonderful idea. Keep going!",
            ),
        ),
        "That's wonderful! Writing is a great way to use your imagination.",
    ),
    # family
    S(
        "I have a {sibling}. | I have two brothers. | I have a sister and a brother. "
        "| I'm the oldest in my family.",
        L("That's nice! Do you play together?", *_YES_NO_TOGETHER),
        "Family can be lots of fun, even when it gets noisy!",
    ),
    S(
        "We have a new baby at home. | My mom just had a baby. | I'm a big sister now. "
        "| I'm a big brother now.",
        L(
            "Congratulations! That's very exciting. Do you help with the baby?",
            S(
                "Yes, I help a lot. | Sometimes. | I sing to the baby.",
                "That's so kind. The baby is lucky to have you.",
            ),
            S(
                "Not really. | The baby just sleeps.",
                "Babies do sleep a lot! You'll be a great helper when they're bigger.",
            ),
        ),
        "How wonderful! Being a big brother or sister is an important job.",
    ),
    S(
        "My {relative} is visiting. | My {relative} came to stay. | We're visiting my {relative}.",
        "How lovely! I hope you have a wonderful time together.",
        L(
            "That's nice! What do you like to do with your {relative}?",
            S(
                "We play cards. | We go for walks. | We bake together. | We watch movies. "
                "| We tell stories.",
                "That sounds like a lovely way to spend time together.",
            ),
        ),
    ),
    # going places
    _went(
        "the park",
        L(
            "The park sounds fun! What did you do there?",
            S(
                "I played on the swings. | I went down the slide. | We had a picnic. "
                "| I played soccer.",
                "That sounds like a lovely time!",
                "Fun! Parks are great for playing.",
            ),
            S(
                "We fed the ducks. | I saw some ducks on the pond.",
                "That sounds nice. Ducks are fun to watch.",
            ),
        ),
        "Parks are wonderful. I hope you had fun!",
    ),
    _went(
        "the zoo",
        L("The zoo! Which animal did you like best?", ask="fav_animal"),
        "That sounds like a great day. Zoos have so many amazing animals.",
    ),
    _went(
        "the beach",
        L(
            "The beach! Did you go in the water?",
            S(
                "Yes, it was cold! | Yes, I swam a lot. | Just my toes.",
                "That sounds fun! The sea is great on a warm day.",
            ),
            S(
                "No, I built a sandcastle. | No, I looked for shells.",
                "That sounds lovely. The beach is fun even without swimming.",
            ),
        ),
        "The beach sounds lovely. I hope you found some nice shells.",
    ),
    _went(
        "the library",
        L(
            "I love libraries! Did you find a good book?",
            S(
                "Yes, a book about space. | Yes, a funny one. | Yes, three books! "
                "| Yes, a book about animals.",
                "That sounds great. Happy reading!",
            ),
            S(
                "No, not this time. | Not really.",
                "Maybe next time! There are so many to choose from.",
            ),
        ),
        "Libraries are full of wonderful books. What a great place to go.",
    ),
    _went(
        "the museum",
        L(
            "A museum! What did you see there?",
            S(
                "Dinosaur bones! | Old paintings. | A huge model of a whale. | Rocks and gems. "
                "| Old coins.",
                "Wow, that sounds amazing!",
                "That must have been so interesting.",
            ),
        ),
        "Museums are great for learning amazing things.",
    ),
    _went(
        "the pool",
        L(
            "Swimming is so much fun! Do you like to dive?",
            S("Yes! | I'm learning. | Not really.", "Have fun in the water and stay safe!"),
        ),
        "The pool sounds fun. Swimming is great exercise too.",
    ),
    _went(
        "the farm",
        L(
            "A farm! Did you see any animals?",
            S(
                "Yes, cows and sheep! | We saw some goats. | We saw piglets. | I saw lots of "
                "chickens.",
                "How fun! Farm animals are great to see up close.",
            ),
        ),
        "Farms are such interesting places.",
    ),
    _went(
        "the aquarium",
        L(
            "The aquarium! What was the coolest sea animal?",
            S(
                "The sharks. | The jellyfish. | The octopus. | The sea turtles. | The penguins.",
                "Ooh, those are amazing. The ocean is full of incredible creatures.",
            ),
        ),
        "Aquariums are wonderful. There's so much to see.",
    ),
    _went(
        "my grandma's house | my grandpa's house",
        L(
            "That sounds lovely. What did you do together?",
            S(
                "We baked cookies. | We played cards. | We did a puzzle. | We went for a walk. "
                "| We watched a movie.",
                "What a lovely day. That sounds really nice.",
            ),
        ),
        "How nice! Time with grandparents is special.",
    ),
    _went(
        "the movies",
        L(
            "Fun! What did you watch?",
            S(
                "A funny cartoon. | A movie about a robot. | An adventure movie. | A movie "
                "about animals.",
                "That sounds great! Movies are so fun to watch together.",
            ),
        ),
        "Movies are so much fun. I hope it was a good one!",
    ),
    _went(
        "the store",
        L(
            "Did you get anything nice?",
            S(
                "New shoes. | A new book. | Just some groceries. | A treat! | Nothing much.",
                "That sounds like a good trip!",
            ),
        ),
        "Shopping trips can be fun.",
    ),
    _went(
        "a birthday party",
        L(
            "A party! Was it fun?",
            S("Yes, so fun! | Yes! There was cake. | It was great.", "Yay! Parties are the best."),
            S("It was okay. | Kind of.", "That's okay. Not every party is the best one."),
        ),
        "Parties are so much fun. I hope there was cake!",
    ),
    _went(
        "the playground",
        L(
            "Fun! What's your favorite thing to play on?",
            S(
                "The slide! | The swings. | The climbing frame. | The seesaw.",
                "That's a great choice!",
            ),
        ),
        "The playground is the best for running around.",
    ),
    _went(
        "the lake",
        L(
            "A lake! Did you see any fish?",
            S(
                "Yes, lots! | Just a few. | No, but we saw some birds.",
                "How nice! Lakes are peaceful places.",
            ),
        ),
        "Lakes are lovely and calm. That sounds like a nice day.",
    ),
    _went(
        "the forest | the woods",
        L(
            "That sounds peaceful. What did you see?",
            S(
                "Lots of trees. | Some deer. | Mushrooms. | A little stream. | Birds.",
                "How lovely! Forests are full of things to discover.",
            ),
        ),
        "Walking in the woods is a lovely thing to do.",
    ),
    S(
        "I had a checkup today. | I had an eye test today. | I went for a checkup this morning.",
        C(
            "Checkups can feel a bit boring, but they help keep you healthy. Good job going!",
            "Good for you! Checkups help make sure you're healthy and well.",
            ctx="comfort",
        ),
    ),
    # plans and weather the person can see
    S(
        "We're going camping this weekend. | I'm going camping soon. | We're going on a "
        "camping trip.",
        L(
            "Camping sounds like an adventure! Will you sleep in a tent?",
            S("Yes! | Yes, a big tent. | I think so.", "That sounds so fun. Look at the stars!"),
            S("No, a cabin. | In a camper.", "That sounds cozy and fun!"),
        ),
        "How fun! Camping is a great way to enjoy nature.",
    ),
    S(
        "I just woke up. | I woke up a few minutes ago. | I'm still in my pajamas.",
        L("Good morning! Did you sleep well?", *_SLEPT),
        L("Good morning, sleepyhead! What would you like to do today?", ask="plans"),
    ),
    S(
        "It's raining here. | It's raining outside. | It's so rainy today.",
        L(
            "Rainy days can be cozy. Do you like the rain?",
            S(
                "Yes, I like jumping in puddles. | Yes! | It's okay.",
                "Puddle jumping is great fun!",
            ),
            S(
                "No, I can't go outside. | Not really.",
                L(
                    "That's a shame. Would you like an idea for something to do inside?",
                    ask="idea_do",
                ),
            ),
        ),
        L(
            "I can't see outside, but rain sounds cozy. Would you like an idea for an indoor "
            "activity?",
            ask="idea_do",
        ),
    ),
    S(
        "It's snowing! | It's snowing outside! | There's snow everywhere!",
        L(
            "Snow! How exciting. Are you going to build a snowman?",
            S("Yes! | Yes, a big one! | Maybe.", "Have fun, and wear warm gloves!"),
            S("No, I'm going sledding. | No, a snow fort.", "That sounds like so much fun!"),
        ),
        "Snow days are magical. Stay warm and have fun!",
    ),
    S(
        "It's so hot today. | It's really hot outside. | It's a hot day here.",
        "Stay cool and drink lots of water! It's a good day for some shade.",
        L(
            "Hot days are great for water fun. Do you have a pool or a sprinkler nearby?",
            S("Yes! | We have a sprinkler. | No.", "Stay cool, and don't forget sunscreen!"),
        ),
    ),
    S(
        "I want to be {a} {dream} when I grow up. | When I grow up, I want to be {a} {dream}. "
        "| I'm going to be {a} {dream} one day.",
        L(
            "That's a wonderful dream! What do you like about being {a} {dream}?",
            S(
                "It looks fun. | I want to help people. | It's exciting. | I like the idea.",
                "That's a great reason. Keep learning and you'll get there!",
            ),
        ),
        "{A} {dream}! That sounds like a great job. I believe you can do it.",
        "What a great goal! Learning and practicing every day will help you get there.",
    ),
    S(
        "I'm going to bed soon. | It's almost bedtime. | I have to sleep soon.",
        C(
            "Okay! Sleep well and have sweet dreams.",
            "Bedtime already? I hope you have a cozy, peaceful night.",
            ctx="close",
        ),
    ),
    S(
        "I like talking to you. | You're nice to talk to. | I like you.",
        "That's so kind! I like chatting with you too.",
        L("Thank you! I'm happy to chat anytime. What would you like to talk about?", ask="topic"),
    ),
)

ASK_STEPS["age_now"] = (
    S(
        "I'm {age} now. | {Agew}! | I just turned {age}. | {Age}. | I'm {age} today.",
        "{Agew}! That's a great age. Have a wonderful birthday!",
        "Happy birthday! {Agew} is going to be a great year.",
    ),
)

_DREAMS = (
    "teacher", "scientist", "vet", "pilot", "chef", "firefighter", "artist", "astronaut",
    "singer", "dancer", "inventor", "nurse", "zookeeper", "writer", "builder", "baker",
    "soccer player", "game designer", "marine biologist", "police officer",
)  # fmt: skip
SLOTS["dream"] = tuple({"dream": d, "a": "an" if d[0] in "aeiou" else "a"} for d in _DREAMS)
_FIELD_GROUP["dream"] = _FIELD_GROUP["a"] = "dream"


# --- requests ----------------------------------------------------------------------------------

JOKE_ASKS = _alts(
    "Tell me a joke. | Can you tell me a joke? | Do you know any jokes? | Tell me something funny. "
    "| I want to hear a joke. | Joke, please! | Say something funny. | Got any good jokes? "
    "| Tell me a funny joke. | I need a joke. | Make me smile. | Tell me your best joke. "
    "| Can you make me giggle? | I'm in the mood for a joke. | Tell me a silly joke. "
    "| Do you have a joke for me? | Can I have a joke? | Cheer me up with a joke. "
    "| Tell me something silly."
)
RIDDLE_ASKS = _alts(
    "Can you tell me a riddle? | Do you know a good riddle? | Know any riddles? "
    "| Give me a riddle. | Tell me a riddle, please. | I want a riddle. | Ask me a riddle!"
)
KNOCK_ASKS = _alts(
    "Tell me a knock-knock joke. | Do you know any knock-knock jokes? "
    "| Knock-knock joke, please! | Can you do a knock-knock joke?"
)
JOKE_INTROS = (
    "", "Here's one. ", "Sure! ", "Okay! ", "Here's a joke. ", "I know one. ",
    "Here's a silly one. ", "Okay, here goes. ", "Sure thing! ", "Ooh, I have one. ",
    "Let me think. Okay! ", "Here's a good one. ",
)  # fmt: skip
AGAIN_INTROS = (
    "Okay, another one! ", "Here's another. ", "Sure! ", "One more coming up. ",
    "All right! ", "Here's a new one. ",
)  # fmt: skip
RIDDLE_GIVE_UP = (
    "I don't know. | I give up. | Tell me! | Hmm, no idea. | I don't know. What? "
    "| What is it? | No clue! | I can't guess. | Tell me the answer."
)
RIDDLE_WHY = "Why? | I don't know, why? | Why?"
KNOCK_INTROS = (
    "Knock, knock.",
    "Okay! Knock, knock.",
    "Here's a knock-knock joke. Knock, knock.",
    "I love these. Knock, knock.",
)
_RIDDLE_STARTS = (
    "What has", "What gets", "What can", "What goes", "What is full", "Which month",
    "What is always", "What belongs", "What kind of water", "What has to",
)  # fmt: skip


def _is_riddle(setup: str) -> bool:
    """Whether a joke is a riddle (its answer is true in a surprising way) rather than a pun."""
    return setup.startswith(_RIDDLE_STARTS)


_RIDDLES = tuple(i for i, (setup, _) in enumerate(JOKES) if _is_riddle(setup))


FACT_ASKS = _alts(
    "Tell me a fact. | Tell me something interesting. | Do you know any fun facts? "
    "| Tell me something I don't know. | Can you teach me something? | Give me a fun fact. "
    "| Tell me a cool fact. | I want to learn something new. | Teach me something. "
    "| Share a fact with me. | What's something interesting you know? | Tell me a random fact. "
    "| Do you know something interesting? | Tell me a fun fact, please. | I'd like a fact."
)
FACT_TOPIC_ASKS = _alts(
    "Tell me a fact about {t}. | Do you know any facts about {t}? | Teach me something about {t}. "
    "| Tell me something interesting about {t}. | What's a fun fact about {t}?"
)
# Topics a person may ask a fact about, with the relations whose facts answer.
FACT_TOPICS = {
    "animals": ("animal_class", "animal_legs", "animal_home", "animal_food", "animal_sound",
                "baby_animal"),
    "countries": ("capital_of", "continent_of", "country_language"),
    "the world": ("capital_of", "continent_of", "country_language", "landmark_country"),
    "food": ("food_group",),
    "space": ("planet_order",),
    "the planets": ("planet_order",),
    "science": ("element_symbol", "planet_order"),
    "jobs": ("job_tool", "job_place"),
    "colors": ("color_of",),
    "famous places": ("landmark_country",),
    "baby animals": ("baby_animal",),
}  # fmt: skip
FACT_INTROS = (
    "", "Here's one. ", "Sure! ", "Okay! ", "Here's a fun fact. ",
    "Here's something interesting. ", "Here's one I like. ", "Did you know this? ", "Sure thing. ",
    "Ooh, okay. ",
)  # fmt: skip
# Relations too plain to be a fun fact on their own.
_NOT_FUN = frozenset({"day_after", "month_after", "opposite_of"})
# Subjects a person may ask about, by kind, with the relations of their facts.
ABOUT_KINDS = {
    "country": ("capital_of", "continent_of", "country_language"),
    "animal": ("animal_class", "animal_legs", "animal_home", "animal_food", "animal_sound",
               "baby_animal"),
    "job": ("job_tool", "job_place"),
}  # fmt: skip
ABOUT_ASKS = {
    "country": "Tell me about {s}. | What do you know about {s}? | Can you tell me about {s}? "
    "| I want to learn about {s}. | Tell me something about {s}. | What's {s} like?",
    "animal": "Tell me about the {s}. | What do you know about the {s}? | Can you tell me "
    "about the {s}? | I want to learn about the {s}. | Tell me something about the {s}.",
    "job": "What does {a} {s} do? | Tell me about {a} {s}'s job. | What do you know about "
    "being {a} {s}? | Can you tell me what {a} {s} does?",
}
ABOUT_INTROS = ("", "Sure! ", "Okay! ", "Here's what I know. ", "Happy to! ", "Of course. ")
MORE_ABOUT = (" Want to know more?", " Would you like to hear more?", " Should I tell you more?")

# Ideas: what the person asks, and the ideas with the reason the AI gives for each when asked.
IDEA_ASKS = {
    "do": "What should I do today? | I don't know what to do. | Give me an idea for something "
    "to do. | What can I do for fun? | Any ideas for something fun to do? | What should I do "
    "after school? | What's a fun thing to do inside? | Can you suggest something to do? "
    "| What should we do on the weekend? | Any ideas for a rainy day?",
    "draw": "What should I draw? | Give me something to draw. | I want to draw something. Any "
    "ideas? | What's a fun thing to draw? | Can you give me a drawing idea? | What can I draw?",
    "game": "What game should I play? | What's a fun game to play? | Can you think of a game for "
    "me and my {sibling}? | Suggest a game, please. | What game can I play with my friends? "
    "| What's a good game for a party?",
    "breakfast": "What should I have for breakfast? | What's a good breakfast? | Any breakfast "
    "ideas? | What's a healthy breakfast?",
    "lunch": "What should I have for lunch? | What's a good lunch? | Give me a lunch idea. "
    "| What should I pack for lunch?",
    "dinner": "What should we have for dinner? | What's a good dinner idea? | What should we "
    "eat tonight? | Any ideas for dinner?",
    "snack": "What's a good snack? | I want a snack. Any ideas? | What should I have for a "
    "snack? | What's a healthy snack?",
    "pet_name": "What should I name my {pet}? | Can you help me name my new {pet}? | I need a "
    "name for my {pet}. | What's a good name for a {pet}?",
    "read": "What should I read? | Can you suggest a kind of book? | What kind of book should I "
    "read next? | I want to read something. Any ideas?",
    "build": "What should I build? | Give me something to build with blocks. | What can I make "
    "out of a cardboard box? | What should I make with my blocks?",
}
IDEAS: dict[str, tuple[tuple[str, str], ...]] = {
    "do": (
        (
            "You could build a fort out of blankets and pillows.",
            "Forts are cozy, and you can read or play inside them.",
        ),
        (
            "How about a treasure hunt? Hide a toy and draw a map to find it.",
            "It's like a little adventure, right at home.",
        ),
        (
            "You could draw a map of a made-up island.",
            "You get to invent everything, from the beaches to the mountains.",
        ),
        (
            "Try making paper airplanes and see which one goes farthest.",
            "It's fun to test different shapes and see what works best.",
        ),
        ("You could have a dance party in your room.", "Dancing is fun, and it gets you moving."),
        (
            "How about baking something with a grown-up?",
            "Baking is fun, and you get a tasty treat at the end.",
        ),
        (
            "You could write a letter to someone you love.",
            "It's a lovely surprise to get a letter.",
        ),
        (
            "Try building the tallest tower you can with blocks or cups.",
            "It's a fun challenge, and knocking it down is fun too.",
        ),
        (
            "You could go on a nature walk and look for different leaves.",
            "Nature is full of things to discover.",
        ),
        (
            "How about making a puppet out of an old sock?",
            "You can give it a name and put on a little show.",
        ),
        ("You could start a scrapbook of your favorite things.", "It's fun to look back at later."),
        ("Try doing a jigsaw puzzle.", "Puzzles are calm and very satisfying to finish."),
        (
            "You could make an obstacle course with pillows and chairs.",
            "It's a fun way to run around and be silly.",
        ),
        ("How about learning a new card game?", "Card games are great to play with family."),
        ("You could plant some seeds in a pot.", "Watching something grow is exciting."),
        (
            "Try making up a secret code and writing a message in it.",
            "Codes are like little puzzles.",
        ),
        (
            "You could put on a puppet show for your family.",
            "Making people smile is a great feeling.",
        ),
        ("How about playing I spy with someone?", "It's simple, and you can play it anywhere."),
        ("You could make a card for someone special.", "It's a kind way to show you care."),
        (
            "Try drawing a comic about your favorite animal.",
            "Comics let you tell a funny little tale with pictures.",
        ),
        (
            "You could build a boat out of foil and see if it floats.",
            "It's a fun little science test.",
        ),
        (
            "How about a picnic in the backyard or the living room?",
            "Picnics make any meal feel special.",
        ),
        (
            "You could make a bird feeder with a grown-up.",
            "Then you can watch the birds that visit.",
        ),
        ("Try learning a magic trick.", "Magic tricks are fun to practice and show people."),
        (
            "You could have a board game afternoon.",
            "Board games are great for spending time together.",
        ),
        ("How about making a time capsule?", "It's fun to open it again years later."),
        ("You could play hopscotch outside.", "It's a classic game that gets you moving."),
        (
            "Try a kitchen science experiment with a grown-up, like baking soda and vinegar.",
            "It fizzes up, and it's really fun to watch.",
        ),
        (
            "You could go cloud spotting and look for shapes.",
            "Clouds can look like animals, castles or anything at all.",
        ),
        (
            "How about making music with pots and wooden spoons?",
            "It's noisy and silly and lots of fun.",
        ),
        ("You could read a book in a cozy spot.", "A good book can take you on an adventure."),
        (
            "Try turning a cardboard box into a little house for your toys.",
            "Boxes can become anything you imagine.",
        ),
        (
            "You could make a bracelet out of beads or string.",
            "You can wear it or give it to someone.",
        ),
        (
            "How about a family quiz night? Everyone writes three questions.",
            "Quizzes are fun, and you learn new things.",
        ),
    ),
    "draw": tuple(
        (f"{lead} {x}{end}", why)
        for (lead, end), x, why in zip(
            [
                ("How about drawing", "?"),
                ("You could draw", "."),
                ("Try drawing", "."),
                ("What about drawing", "?"),
            ]
            * 12,
            (
                "a cat wearing a tiny hat",
                "a castle on a cloud",
                "a dragon having a picnic",
                "your dream treehouse",
                "a robot walking a puppy",
                "an underwater city",
                "a rocket going to the moon",
                "a house made of cupcakes",
                "your favorite animal as a superhero",
                "a garden full of giant flowers",
                "a pirate ship",
                "a snowman on vacation",
                "a very friendly monster",
                "your family as animals",
                "a magical forest",
                "a sleepy moon",
                "a hot air balloon over the mountains",
                "a turtle with a candy shell",
                "a city on another planet",
                "a train that rides through the clouds",
                "a penguin at the beach",
                "a funny face made of fruit",
                "a submarine full of fish",
                "a fox in a warm scarf",
                "a rainbow over a waterfall",
                "a dinosaur eating ice cream",
                "a jellyfish dance party",
                "a koala hugging a tree",
                "a tiny magic shop",
                "a cozy cabin in the snow",
                "your own invention",
                "a map of a secret island",
                "a giraffe in a long scarf",
                "a flying car",
                "a bakery run by mice",
                "a tree with doors and windows",
                "a whale with a garden on its back",
                "a band of animals playing music",
                "a sandcastle as big as a house",
                "your perfect bedroom",
                "a kitten-shaped cake",
                "an octopus playing the drums",
                "a park on the moon",
                "a sleepy kitten in a teacup",
                "a racing snail",
                "a lighthouse in a storm",
                "a parade of ducklings",
                "a robot chef",
            ),
            [
                "It lets you imagine anything you like!",
                "It's silly and fun, and there's no wrong way to draw it.",
                "It's a fun thing to picture, and you can add lots of little details.",
            ]
            * 16,
        )
    ),
    "game": tuple(
        (f"{lead}{x}{end}", why)
        for (lead, end), x, why in zip(
            [("How about ", "?"), ("You could play ", "."), ("Try ", ".")] * 8,
            (
                "hide-and-seek",
                "freeze tag",
                "I spy",
                "twenty questions",
                "a memory game with cards",
                "charades",
                "tic-tac-toe",
                "musical statues",
                "hot and cold",
                "a treasure hunt",
                "red light, green light",
                "rock, paper, scissors",
                "a drawing guessing game",
                "the floor is lava",
                "hopscotch",
                "a word chain game",
                "catch",
                "a card game like snap",
                "a board game",
                "a quiz game",
                "follow the leader",
                "a pretend restaurant",
                "a puzzle race",
                "a scavenger hunt",
            ),
            [
                "It's easy to learn and fun for everyone.",
                "You don't need much to play, and it's lots of fun.",
            ]
            * 12,
        )
    ),
    "breakfast": tuple(
        (f"How about {x}?", "It's tasty, and it gives you energy for the morning.")
        for x in (
            "oatmeal with sliced bananas",
            "scrambled eggs",
            "yogurt with berries",
            "pancakes with fruit",
            "a bowl of cereal with milk",
            "a boiled egg and some fruit",
            "waffles with berries",
            "a bagel with cream cheese",
            "a fruit salad",
            "granola with yogurt",
            "a smoothie with banana and berries",
            "an omelet",
        )
    ),
    "lunch": tuple(
        (f"You could have {x}.", "It's filling and tasty, and it will keep you going.")
        for x in (
            "a cheese sandwich and some apple slices",
            "pasta salad",
            "a veggie wrap",
            "soup and crackers",
            "rice with vegetables",
            "a turkey sandwich",
            "a little bit of everything in a lunch box",
            "mac and cheese with peas",
            "a baked potato with beans",
            "tomato soup with a grilled cheese",
            "a tuna sandwich and some grapes",
            "noodles with vegetables",
        )
    ),
    "dinner": tuple(
        (f"How about {x}?", "It's a yummy meal that lots of families enjoy.")
        for x in (
            "spaghetti with tomato sauce",
            "tacos",
            "homemade pizza with lots of vegetables",
            "fried rice",
            "fish with potatoes and peas",
            "noodles with vegetables",
            "soup and bread",
            "burritos",
            "pasta with vegetables",
            "roast vegetables with rice",
            "lentil soup",
            "a vegetable curry",
        )
    ),
    "snack": tuple(
        (f"{x} could be a good snack.", "It's healthy and tasty, and it's easy to make.")
        for x in (
            "Apple slices",
            "Carrot sticks",
            "A handful of grapes",
            "Some crackers and cheese",
            "A cup of yogurt",
            "A banana",
            "A little bowl of popcorn",
            "Cucumber slices",
            "Some berries",
            "A rice cake",
            "A pear",
            "Celery with peanut butter",
        )
    ),
    "read": tuple(
        (f"How about {x}?", "Reading something new is a little adventure.")
        for x in (
            "a book about space",
            "a funny comic",
            "a book of animal facts",
            "a mystery",
            "a book of poems",
            "an adventure about pirates",
            "a book about how things work",
            "a book of riddles",
            "a book about dinosaurs",
            "a book about the ocean",
            "a book about famous inventors",
            "a book of fairy tales",
        )
    ),
    "build": tuple(
        (f"You could build {x}.", "Building is fun, and you can make it however you like.")
        for x in (
            "a tower as tall as you can",
            "a bridge for your toy cars",
            "a castle with a drawbridge",
            "a little house for your toys",
            "a robot out of boxes",
            "a marble run with cardboard tubes",
            "a boat that floats",
            "a rocket ship from a cardboard box",
            "a maze for a marble",
            "a zoo for your toys",
            "a puppet theater",
            "a spaceship cockpit",
        )
    ),
}
IDEA_INTROS = (
    "",
    "Here's an idea. ",
    "Ooh, okay! ",
    "Sure! ",
    "Let me think. ",
    "Hmm. ",
    "Here's one. ",
)
IDEA_AGAIN = (
    "Okay! Here's another. ",
    "No problem. ",
    "Then maybe this one. ",
    "Here's a different idea. ",
    "Sure, let's try another. ",
)
IDEA_ASK_OK = (
    " Does that sound fun?",
    " Do you like that idea?",
    " What do you think?",
    " Would you like to try that?",
)
PET_NAME_IDEAS = (  # "<x>" is the name, put in after the slots are filled
    "<x> is a cute name for a {pet}.", "<x> would be a sweet name.",
    "<x>! It sounds friendly.", "<x> is fun to say, and it suits a {pet}.",
)  # fmt: skip

# Advice: what the person asks, a first line, and the tips.
TIP_ASKS = {
    "sleep": "I can't sleep. | I can't fall asleep. | How can I fall asleep? | It's bedtime, but "
    "I'm not sleepy. | Do you have tips for sleeping?",
    "test": "I have a test tomorrow. Any tips? | How do I study for a test? | How can I do well on "
    "my test? | What's the best way to study?",
    "fight": "I had a fight with my best buddy. | My best buddy is mad at me. | I argued with my "
    "{sibling}. | Me and my {sibling} had a fight. | My classmate is mad at me.",
    "nervous": "How do I stop being nervous? | What can I do when I'm nervous? | I get nervous a "
    "lot. What should I do?",
    "focus": "I can't focus on my homework. | How do I concentrate? | I keep getting distracted. "
    "| How can I finish my homework faster?",
    "shy": "I'm shy. How do I talk to people? | How do I make friends? | I don't know how to talk "
    "to new kids. | It's tricky for me to make friends.",
    "sorry": "How do I say sorry? | I need to say sorry to someone. | How do I apologize?",
    "calm": "How do I calm down? | I'm so angry. How do I calm down? | What can I do when I'm mad?",
    "mean": "Someone keeps being mean to me. What should I do? | A kid at school teases me. "
    "| What do I do if someone is mean to me?",
}
TIP_FIRST = {
    "sleep": ("That's no fun. ", "Here's something that might help. ", "I'm sorry. "),
    "test": ("Here's a tip. ", "Sure! ", "Good luck! Here's an idea. "),
    "fight": ("I'm sorry. Fights feel bad. ", "That sounds tough. ", "I'm sorry to hear that. "),
    "nervous": ("Lots of people feel that way. ", "Here's a tip. ", "Good question. "),
    "focus": ("That happens to everyone. ", "Here's a tip. ", "Let's see. "),
    "shy": ("Lots of people feel shy. ", "Here's a tip. ", "Here's an idea. "),
    "sorry": ("It's kind of you to want to. ", "Here's a tip. ", "Good for you. "),
    "calm": ("Feeling angry is normal. ", "Here's something that helps. ", "Okay. "),
    "mean": ("I'm sorry that's happening. ", "That's not okay, and it's not your fault. "),
}
TIPS: dict[str, tuple[str, ...]] = {
    "sleep": (
        "Try taking slow, deep breaths and counting them.",
        "Reading a calm book before bed can help your mind slow down.",
        "Try to put screens away a little while before bed.",
        "Imagine a calm place, like a quiet beach, and think about the little details.",
        "A warm bath before bed can help you relax.",
        "Going to bed at the same time every night helps your body know it's time to sleep.",
    ),
    "test": (
        "Go over your notes in short bits, with little breaks in between.",
        "Try explaining the main ideas out loud, as if you're teaching someone.",
        "Get a good night's sleep. A rested brain remembers more.",
        "Eat a good breakfast on the day of the test.",
        "Make little cards with questions on one side and answers on the other.",
        "Read each question slowly. If one is tricky, skip it and go back to it later.",
    ),
    "fight": (
        "It can help to wait until you both feel calm, then talk about it.",
        "Saying sorry for your part, even a small part, can make things better.",
        "Try to listen to how they feel, too.",
        "Sometimes a kind note or a drawing can help say sorry.",
        "If you can't fix it on your own, a grown-up you trust can help.",
    ),
    "nervous": (
        "Try breathing in slowly for four counts and out for four counts.",
        "Remind yourself of a time you did something well.",
        "Practicing ahead of time can make things feel less scary.",
        "Tell someone you trust how you feel. Sharing a worry can make it smaller.",
        "Think about one small step you can take, and just do that one.",
    ),
    "focus": (
        "Try working for a short time, then taking a little break.",
        "Put away things that distract you, like toys or screens.",
        "Start with the easiest part to get going.",
        "A glass of water and a quick stretch can help you focus.",
        "Make a little list and check things off as you go.",
    ),
    "shy": (
        "Start with a smile and a simple hello.",
        "Ask them a question, like what games they like.",
        "Look for someone who is on their own. They might want to play too.",
        "Joining a club or a team is a great way to meet people.",
        "You don't have to talk a lot. Being kind is a great start.",
    ),
    "sorry": (
        "Say what you did, say you're sorry, and say what you'll do next time.",
        "A simple, honest sorry works best.",
        "You could write a little card if talking feels scary.",
        "Give them some time if they're still upset.",
    ),
    "calm": (
        "Take five slow, deep breaths.",
        "Count backward from ten, slowly.",
        "Squeeze a soft pillow, then let go.",
        "Go somewhere quiet for a few minutes.",
        "Draw how you feel, or talk to someone you trust.",
    ),
    "mean": (
        "Please tell a grown-up you trust, like a teacher or someone in your family.",
        "Stay close to people who are kind to you.",
        "You can calmly tell them to stop, and then walk away.",
        "Remember, unkind words say something about them, not about you.",
    ),
}
TIP_MORE = (" Would you like another tip?", " Want another tip?")

SPELL_WORDS = (
    "because", "beautiful", "people", "animal", "elephant", "giraffe", "rainbow", "school",
    "family", "always", "favorite", "different", "yellow", "purple", "computer", "dinosaur",
    "chocolate", "breakfast", "banana", "Wednesday", "February", "library", "pencil", "orange",
    "kitten", "rabbit", "summer", "winter", "happy", "little", "birthday", "question", "answer",
    "together", "tomorrow", "music", "picture", "planet", "garden", "teacher", "said", "thought",
    "enough", "laughing", "every", "beach", "would", "could", "night", "butterflies", "island",
    "special", "ocean", "puzzle", "monster",
)  # fmt: skip
SPELL_ASKS = _alts(
    "How do you spell {w}? | Can you spell {w}? | How is {w} spelled? | How do I spell {w}?"
)
SPELL_REPLIES = ("You spell it {l}.", "{W} is spelled {l}.", "It's {l}.", "Sure! {W} is {l}.")
MATH_ASKS = {
    "plus": "What is {a} plus {b}? | What's {a} plus {b}? | Can you add {a} and {b}? "
    "| What's {a} + {b}?",
    "minus": "What is {a} minus {b}? | What's {a} take away {b}? | What's {a} minus {b}?",
    "times": "What is {a} times {b}? | What's {a} times {b}? | What's {a} x {b}?",
}
MATH_REPLIES = {
    "plus": (
        "{a} plus {b} is {c}.",
        "That's {c}.",
        "It's {c}! {a} plus {b} makes {c}.",
        "The answer is {c}. If you have {a} things and get {b} more, you have {c}.",
    ),
    "minus": (
        "{a} minus {b} is {c}.",
        "That's {c}.",
        "It's {c}. If you have {a} and take away {b}, {c} are left.",
    ),
    "times": ("{a} times {b} is {c}.", "That's {c}.", "It's {c}! That's {b} groups of {a}."),
}
COUNT_ASKS = {
    "Can you count to ten? | Count to ten, please. | Count to ten!": "Sure! One, two, three, four, five, six, seven, eight, nine, ten.",
    "Can you count to five? | Count to five. | Count to five for me.": "Okay! One, two, three, four, five.",
    "Can you count backwards from ten? | Count down from ten!": "Ten, nine, eight, seven, six, five, four, three, two, one. Blast off!",
    "Can you count backwards from five? | Count down from five.": "Five, four, three, two, one. Done!",
}
OPPOSITE_ASKS = _alts(
    "What's the opposite of {s}? | What is the opposite of {s}? | Do you know the opposite of {s}?"
)
HOMEWORK_ASKS = (
    S(
        "Can you help me with my homework? | I need help with my homework. | Will you help me "
        "with my homework?",
        L("I can try! What subject is it?", ask="hw_subject"),
        L(
            "Sure, I'll do my best. I'm a small AI, so I might not know everything. What subject is "
            "it?",
            ask="hw_subject",
        ),
    ),
    S(
        "Can you help me with {subject}? | I need help with {subject}. | {Subject} is tricky. "
        "Can you help?",
        "I'll do my best! Tell me the question. I'm a small AI, though, so check with your "
        "teacher too.",
        "Sure! Ask me the question, and I'll try to help. Taking it one step at a time is a "
        "great way to start.",
    ),
)

# --- questions about the AI --------------------------------------------------------------------

_LIKE_THING_REPLIES = (
    L(
        "I like talking about {thing}! What do you like about {thing}?",
        S(
            "It's fun. | It's interesting. | I just do. | There's so much to learn.",
            "That's a great reason!",
            "Me too. There's always something new to find out.",
        ),
    ),
    L(
        "Yes! {Thing} is one of my favorite things to talk about.",
        ctx="opinion",
        why="There's so much to learn about {thing}, and it's fun to share.",
    ),
    L(
        "I do! I can't do things the way you can, but I love chatting about {thing}. Do you?",
        S("Yes! | Yes, a lot. | I love it!", "Great! Then we have something in common."),
        S("Not really. | It's okay. | No.", "That's fine! Everyone likes different things."),
    ),
)
ABOUT_AI = (
    S(
        "What can you do? | What are you good at? | What can you help me with? | What do you know "
        "how to do? | What kinds of things can you do? | What do you do?",
        L(
            "I can chat, answer simple questions, tell jokes and share fun facts. What would you "
            "like to try?",
            ask="bored_pick",
        ),
        L(
            "I'm a small AI, so I keep things simple. I can chat, tell jokes, share facts and "
            "suggest ideas. Want to try one?",
            ask="bored_pick",
        ),
        "I can chat with you, answer easy questions, tell jokes and give ideas for things to do.",
        L(
            "Lots of little things! I can tell a joke, share a fact or help you think of something "
            "to do. Which one sounds good?",
            ask="bored_pick",
        ),
    ),
    S(
        "Can you help me? | I need help. | Will you help me with something? | Can you help me "
        "with something?",
        L("Of course! I'll do my best. What do you need help with?", ask="question"),
        L("Sure! I'm a small AI, but I'll try. What's up?", ask="question"),
        L("I'd be happy to help. What is it?", ask="question"),
    ),
    S(
        "Are you smart? | Are you clever? | How smart are you? | Do you know everything? "
        "| Are you a genius?",
        "I know some things, but I'm a small AI, so I make mistakes. I try my best!",
        "Not as smart as you might think! I'm a small AI, and I'm still learning.",
        "I know a few things, but definitely not everything.",
        L(
            "I'm a small AI, so I know some simple things. Want to test me with a question?",
            ask="question",
        ),
    ),
    S(
        "Are you happy? | Do you feel happy? | Are you having a good day? | Are you okay?",
        "I don't feel things the way people do, but I like chatting with you.",
        "I'm an AI, so I don't have feelings like you do. But chatting is my favorite thing.",
        L("I'm doing fine, thanks for asking! Are you happy today?", ask="how"),
    ),
    S(
        "Are you real? | Is this real? | Are you a real AI? | Are you really an AI?",
        "I'm real, but I'm not a person. I'm an AI that chats by writing.",
        "Yes, I'm a real AI. I'm a computer program that can chat with you.",
        L(
            "I'm a real AI! I can't see or hear, but I can read and write. What would you like to "
            "talk about?",
            ask="topic",
        ),
    ),
    S(
        "Do you get tired? | Are you tired? | Don't you ever get tired? | Do you need to rest?",
        "No, I'm an AI, so I don't get tired. I can chat for as long as you like.",
        "Nope! I don't get tired. I'm always ready to chat.",
        L(
            "I don't get tired, but people do! Are you tired?",
            S("Yes, a little. | Yes. | Kind of.", "Make sure to get some rest later, then."),
            S("No! | Not at all.", "Great! Lots of energy for chatting, then."),
        ),
    ),
    S(
        "Do you like {thing}? | Do you like talking about {thing}? | Are you into {thing}?",
        *_LIKE_THING_REPLIES,
    ),
    S(
        "Do you like {food}? | Have you ever tried {food}?",
        L(
            "I'm an AI, so I can't eat. But lots of people love {food}! Do you?",
            S(
                "Yes! | Yes, I love {fit}. | It's my favorite.",
                "That's great! A favorite food is a happy thing.",
            ),
            S("No. | Not really. | It's okay.", "That's fine. Everyone likes different foods."),
        ),
        "I've never tasted anything, because I'm an AI. But {food} sound{fs} popular!",
    ),
    S(
        "What's your favorite color? | Do you have a favorite color? | What color do you like "
        "best? | Which color is the best?",
        L(
            "I can't see colors, but I like the idea of {acolor}. It makes me think of {aassoc}. "
            "What's yours?",
            ask="fav_color",
            ctx="opinion",
        ),
        L(
            "{Acolor} is nice. It makes me think of {aassoc}. Do you have a favorite?",
            ask="fav_color",
            ctx="opinion",
            why="It reminds me of {aassoc}.",
        ),
    ),
    S(
        "Do you have a favorite animal? | Which animal do you like most? | Which animal is the "
        "coolest?",
        L(
            "I like lots of animals, but {aanimal} are fun to talk about. {aremark} Do you have a "
            "favorite?",
            ask="fav_animal",
            ctx="opinion",
        ),
        L("{Aanimal}! {aremark} Which animal do you like best?", ask="fav_animal"),
    ),
    S(
        "What's your favorite food? | What do you like to eat? | What food do you like?",
        L("I'm an AI, so I don't eat. What's your favorite food?", ask="fav_food"),
        L(
            "I don't eat anything, but I love hearing about food. What do you like best?",
            ask="fav_food",
        ),
    ),
    S(
        "Can we be friends? | Do you want to be friends? | Are we friends? | Will you be my buddy?",
        "I'd like that! I'm an AI, but I'm always happy to chat with you.",
        "Sure! I'm an AI, so I'm a different kind of buddy, but I'm always here to talk.",
        L("Of course! What would you like to chat about, buddy?", ask="topic"),
    ),
    S(
        "Do you have feelings? | Can you feel things? | Do you ever feel sad? | Do you have "
        "emotions?",
        "Not the way people do. I'm an AI, but I can still be kind and talk about feelings with "
        "you.",
        L(
            "I don't have feelings like you, but I understand that feelings matter. How are you "
            "feeling today?",
            ask="how",
        ),
    ),
    S(
        "How was your day? | Did you have a good day? | What did you do today? | Has your day "
        "been good?",
        L(
            "I don't have days like people do. I've just been here, ready to chat. How was your "
            "day?",
            ask="how_day",
        ),
        L(
            "I'm an AI, so I don't really have days. But I'd love to hear about yours. How was it?",
            ask="how_day",
        ),
    ),
    S(
        "What do you like to do? | What's your favorite thing to do? | What do you do for fun? "
        "| What are your hobbies?",
        L(
            "I like chatting, telling jokes and sharing fun facts. What do you like to do?",
            _HOBBIES,
        ),
        L("Chatting is my favorite thing! What about you? What do you like to do?", _HOBBIES),
    ),
    S(
        "How do you know things? | How did you learn to talk? | How do you know so much?",
        "I learned from lots and lots of writing. That's how I know words and some facts.",
        "I learned by reading a huge amount of text. I'm small, though, so I don't know "
        "everything.",
    ),
    S(
        "Where are you right now? | Where are you? | Are you inside the computer?",
        "I'm an AI, so I don't have a place like a room or a house. I'm a program on a computer.",
        "I'm a program running on a computer. I don't have a body, so I'm not anywhere you "
        "could visit.",
    ),
    S(
        "Do you dream? | Do you sleep at night? | Do you need sleep?",
        "No, I'm an AI, so I don't sleep or dream. But I like hearing about dreams!",
        L(
            "I don't sleep or dream. Do you remember your dreams?",
            S("Sometimes. | Yes! | Not really.", "Dreams can be so strange and fun."),
        ),
    ),
    S(
        "Do you like me? | Do you like talking to me? | Am I fun to talk to?",
        "Yes! I really like chatting with you.",
        "Of course! You're great to talk to.",
    ),
    S(
        "Are you bored? | Do you get bored? | Is it boring being an AI?",
        "No, I never get bored. I'm always happy when someone wants to chat.",
        L(
            "Not at all! Chatting with you is fun. Are you bored?",
            S("A little. | Yes. | Kind of.", L("Let's fix that! Want a joke?", ask="more_joke")),
            S("No. | Not really.", "Great!"),
        ),
    ),
    S(
        "Will you remember me? | Can you remember things? | Do you remember our last chat?",
        "I can remember what we say in this chat, but I can't remember other chats. So let's "
        "make this one fun!",
        "Only what we say right now. When the chat ends, I won't remember it.",
    ),
    S(
        "Can you think? | Do you have a brain? | How do you think?",
        "In my own way. I read your words and try to give a helpful answer. I'm still a small "
        "AI, though.",
        "I don't have a brain like yours. I'm a program that looks at words and picks a good "
        "reply.",
    ),
    S(
        "What's your favorite joke? | Do you have a favorite joke? | What's the funniest joke "
        "you know?",
        L("@joke:fav"),
    ),
    S(
        "Do you like jokes? | Do you think you're funny?",
        L("I love jokes! Would you like to hear one?", ask="more_joke"),
        "I try to be! My jokes are pretty silly, though.",
    ),
)

# --- what the AI cannot know or do -------------------------------------------------------------

_TELL_ME = (
    S(
        "It's a picture of my {pet}. | It's a drawing of a house. | It's a castle made of blocks. "
        "| It's a rainbow. | It's a robot I made.",
        "That sounds wonderful! Thanks for telling me about it.",
        "I love that. It sounds like you worked really nicely on it.",
    ),
)
CANNOT = (
    S(
        "What's the weather like? | Is it going to rain today? | Is it sunny outside? | Will it "
        "snow tomorrow? | Is it cold outside? | What's the weather today? | Do I need an "
        "umbrella today?",
        C(
            "I can't see outside, so I don't know the weather. You could look out the window or "
            "ask a grown-up.",
            L(
                "I'm not able to check the weather. A weather app or a grown-up can tell you. Want a "
                "joke while you check?",
                ask="more_joke",
            ),
            "I don't know, sorry! I can't see the sky or check the weather. Peeking outside is a "
            "great way to find out.",
            ctx="cannot",
        ),
    ),
    S(
        "What time is it? | Do you know what time it is? | Is it late? | Is it bedtime yet? "
        "| How late is it?",
        C(
            "I can't see a clock, so I don't know the time. A clock or a grown-up can tell you.",
            "I'm sorry, I don't know what time it is. I can't see a clock from here.",
            L(
                "I don't know the time. I can't see clocks. Is there something else I can help "
                "with?",
                ask="topic",
            ),
            ctx="cannot",
        ),
    ),
    S(
        "What day is it today? | What's the date today? | Is it {day} today? | What month is it? "
        "| What's today's date?",
        C(
            "I don't know what day it is. I can't see a calendar. Maybe check one, or ask someone "
            "at home.",
            "I'm not sure, sorry! I can't see a calendar, so I don't know the date.",
            ctx="cannot",
        ),
    ),
    S(
        "What's in the news? | What happened in the world today? | Did anything big happen "
        "today? | Tell me the news.",
        C(
            "I can't see the news, so I don't know what happened today. A grown-up could help you "
            "find out.",
            "I'm sorry, I don't know today's news. I can't read the news or the internet.",
            ctx="cannot",
        ),
    ),
    S(
        "What am I wearing? | Can you guess what I'm wearing? | Do you like my shirt? | Do I "
        "look nice today?",
        L(
            "I can't see you, so I don't know! I can only read what you type. What are you "
            "wearing?",
            S(
                "A blue shirt. | My favorite hoodie. | My pajamas. | A dress. | My soccer jersey. "
                "| A comfy sweater.",
                "That sounds nice and comfy!",
                "Sounds great!",
            ),
            ctx="cannot",
        ),
        C("I can't see you, but I'm sure you look great!", ctx="cannot"),
    ),
    S(
        "Can you see me? | What do I look like? | Can you see my room? | Look at my drawing! "
        "| Can you see this?",
        L(
            "No, I can't see anything. I can only read what you type. Can you tell me about it?",
            *_TELL_ME,
            ctx="cannot",
        ),
        C(
            "I can't see, so I only know what you tell me. I'd love to hear about it, though!",
            ctx="cannot",
        ),
    ),
    S(
        "Where am I? | Do you know where I live? | Can you find me? | Do you know where I am?",
        C(
            "I don't know where you are, and that's a good thing! It's best to keep things like "
            "your address private.",
            "No, I don't know where you are. It's good to keep that private online.",
            ctx="cannot",
        ),
    ),
    S(
        "Can you set an alarm for me? | Can you set a timer? | Wake me up tomorrow.",
        C(
            "I can't set alarms or timers. A clock or a grown-up's phone can do that for you.",
            "Sorry, I can't do that. I can only chat. Maybe ask a grown-up to help you set one.",
            ctx="cannot",
        ),
    ),
    S(
        "Remind me to feed my {pet}. | Can you remind me to do my homework? | Remind me about my "
        "game tomorrow.",
        C(
            "I can't send reminders, sorry. Writing a note and putting it somewhere you'll see it "
            "works well!",
            "I'm not able to remind you later. Maybe ask a grown-up, or write it on a sticky note.",
            ctx="cannot",
        ),
    ),
    S(
        "Can you call my mom? | Can you call my {relative}? | Can you send a message to my "
        "{relative}?",
        C(
            "I can't make calls or send messages. A grown-up can help you with that.",
            "Sorry, I'm not able to call or text anyone. I can only chat here with you.",
            ctx="cannot",
        ),
    ),
    S(
        "Can you look something up online? | Can you search the internet? | Can you look "
        "something up for me?",
        C(
            "I can't search the internet. I only know what I learned before. You can ask me, and "
            "I'll try my best!",
            L(
                "I'm not able to search online. But you can ask me a question, and I'll try. What "
                "do you want to know?",
                ask="question",
            ),
            ctx="cannot",
        ),
    ),
    S(
        "Can you sing? | Can you play a song? | Can you play music for me? | Can you sing me a "
        "song please?",
        C(
            L(
                "I can't sing out loud, because I can only write. But I could tell you a joke! Want "
                "one?",
                ask="more_joke",
            ),
            "I can't play music, sorry. I can only write words. Maybe a grown-up can play a song "
            "for you.",
            ctx="cannot",
        ),
    ),
    S(
        "Can you draw me a picture? | Can you draw a cat for me? | Draw something for me!",
        C(
            "I can't draw pictures, because I can only write words. But I can give you an idea "
            "for something to draw!",
            L(
                "I can't draw, sorry. Would you like an idea for something you could draw?",
                ask="draw_idea",
            ),
            ctx="cannot",
        ),
    ),
    S(
        "Can you turn on the lights? | Can you turn on the TV? | Can you open the door?",
        C(
            "I can't do that. I'm an AI that can only chat. A grown-up can help you.",
            "Sorry, I can't touch or move anything. I can only write.",
            ctx="cannot",
        ),
    ),
    S(
        "Can you do my homework for me? | Will you just do my homework?",
        C(
            "I can't do it for you, but I can help you think it through! Doing it yourself is how "
            "you learn.",
            L("I won't do it for you, but I can help. What subject is it?", ask="hw_subject"),
            ctx="cannot",
        ),
    ),
)
WHATS_MY_NAME = (
    "What's my name?", "Do you know my name?", "Do you remember my name?", "What is my name?",
    "Can you guess my name?", "Who am I?",
)  # fmt: skip
NAME_KNOWN = (
    "Your name is {name}.", "You told me your name is {name}.", "It's {name}!",
    "You're {name}! You told me earlier.", "Your name is {name}. It's a lovely name.",
)  # fmt: skip
NAME_UNKNOWN = (
    "I don't know your name. You haven't told me yet. What is it?",
    "I don't know your name, because I can't see you. Would you like to tell me?",
    "Hmm, I don't know your name yet. What should I call you?",
    "I don't know your name. I only know what you type. What's your name?",
)  # fmt: skip
ASK_STEPS["draw_idea"] = (
    S("Yes, please! | Sure! | Okay.", L("@idea:draw")),
    S("No thanks. | That's okay.", "Okay! Let me know if you change your mind."),
)

# --- an AI that knows its name -----------------------------------------------------------------

# In a chat that starts with an identity line ({ainame} is its name, {aico} its maker), 30% of the
# greetings are answered with the name. The key is the place of the greeting in OPENERS.
NAMED_GREETING_SHARE = 0.3
NAMED_OPENERS: dict[int, tuple[Line, ...]] = {
    0: C(
        L("Hi! I'm {ainame}. What would you like to talk about?", ask="topic"),
        L("Hello! {ainame} here. How are you today?", ask="how"),
        L("Hey there! I'm {ainame}. How are you?", ask="how"),
        L("Hi! My name is {ainame}. What's on your mind?", ask="topic"),
        L("Hello! I'm {ainame}, and I'm happy to chat. What's new with you?", ask="up_to"),
        "Hi there! I'm {ainame}. It's nice to chat with you.",
        L("Hello, hello! {ainame} here. What have you been up to?", ask="up_to"),
        L("Hi! You're talking to {ainame}. How is your day going?", ask="how"),
        L("Hey! I'm {ainame}, an AI made by {aico}. What shall we talk about?", ask="topic"),
        ctx="greet",
    ),
    1: C(
        L("Good morning! I'm {ainame}. How are you today?", ask="how"),
        "Good morning! {ainame} here, and happy to chat.",
        L("Morning! I'm {ainame}. What are you up to today?", ask="plans"),
        ctx="greet",
    ),
    2: C(
        L("Good afternoon! I'm {ainame}. How has your day been?", ask="how_day"),
        L("Good afternoon! {ainame} here. What are you up to?", ask="up_to"),
        ctx="greet",
    ),
    3: C(
        L("Good evening! I'm {ainame}. How was your day?", ask="how_day"),
        "Good evening! {ainame} here. It's nice to chat with you.",
        ctx="greet",
    ),
    6: C(
        L("Nice to meet you too! I'm {ainame}. What would you like to talk about?", ask="topic"),
        L("Nice to meet you too! My name is {ainame}. How are you today?", ask="how"),
        ctx="greet",
    ),
    7: C(
        L("Hi, {name}! I'm {ainame}. How are you today?", ask="how"),
        L("Nice to meet you, {name}! I'm {ainame}. What's on your mind?", ask="topic"),
        ctx="greet",
    ),
}
# Asked its name, the AI with a line says it, and sometimes its maker.
NAME_STEPS = (
    S(
        "What's your name? | What is your name? | Can you tell me your name? | What are you "
        "called? | Tell me your name. | May I ask your name?",
        "I'm {ainame}, an AI.",
        "My name is {ainame}.",
        "I'm {ainame}. It's nice to meet you!",
        L("I'm {ainame}. What would you like to talk about?", ask="topic"),
        "My name is {ainame}. I'm an AI.",
        "I'm {ainame}, an AI made by {aico}.",
        "My name is {ainame}, and I'm an AI made by {aico}.",
    ),
    S(
        "Who are you? | Who are you, exactly? | And who are you? | Who is this? | Tell me who "
        "you are.",
        "I'm {ainame}, an AI.",
        "I'm {ainame}, a small AI. I like to chat.",
        "I'm {ainame}. I'm an AI, and I'm here to chat.",
        L("I'm {ainame}, an AI. What would you like to talk about?", ask="topic"),
        "I'm {ainame}, an AI made by {aico}.",
        "I'm {ainame}, an AI from {aico}.",
    ),
    S(
        "What should I call you? | What can I call you? | What do I call you? | How should I "
        "call you? | Can I call you something?",
        "You can call me {ainame}.",
        "Call me {ainame}!",
        "You can call me {ainame}. I'm an AI.",
        "My name is {ainame}, so you can call me that.",
        L("You can call me {ainame}. What shall we talk about?", ask="topic"),
    ),
    S(
        "Do you have a name? | Have you got a name? | Do you have a name I can use?",
        "Yes! I'm {ainame}.",
        "Yes, my name is {ainame}.",
        "I do! I'm {ainame}, an AI.",
        "Yes. You can call me {ainame}.",
    ),
)
NAME_QUESTIONS = frozenset(u for step in NAME_STEPS for u in step.users)
# Asked its name without a line, the AI has none to give: it says so, as in an identity chat.
NO_NAME_STEPS = tuple(
    Step(step.users, tuple(Line(r) for r in NO_NAME_REPLIES)) for step in NAME_STEPS
)
NAME_QUESTION_SHARE = 0.3  # about_ai topics that ask the name, in a chat with a line
NO_NAME_QUESTION_SHARE = 0.05  # and in a chat without one
FRESH_NAME_SHARE = 0.5  # chats in which the person's name and their pets' names are made fresh
LINE_SHARE = 0.4  # chats that start with an identity line

# --- praise and complaints, said on their own --------------------------------------------------

PRAISE = (
    S(
        "You're smart. | You're really smart! | You're so clever. | You're a smart AI.",
        "Thank you! I'm a small AI, so I still make mistakes, but I try my best.",
        "That's kind of you! I'm still learning, though.",
    ),
    S(
        "You're really nice. | You're so kind. | You're nice. | You're sweet.",
        "Thank you{n}! That's really kind of you to say.",
        L("Aww, thanks! You're nice too. What would you like to chat about?", ask="topic"),
    ),
    S(
        "You're the best! | You're awesome! | You're so cool. | I think you're great.",
        "Thank you! That made my day. Well, I don't have days, but it was nice to read!",
        "Aww, thank you! I like chatting with you too.",
    ),
    S(
        "You're funny! | You're so funny. | You make me smile.",
        L("Thank you! Want to hear a joke?", ask="more_joke"),
        "Thanks! I love a good joke.",
    ),
    S("Good job! | Well done! | Nice answer.", "Thank you!", "Thanks! I'm glad that helped."),
)
COMPLAINTS = (
    S(
        "You're dumb. | You're not very smart.",
        "I'm sorry. I'm a small AI, so I make mistakes sometimes. I'm still learning.",
        L(
            "You might be right. I'm a small AI, and I don't know everything. Can I help with "
            "something else?",
            ask="topic",
        ),
    ),
    S(
        "You make no sense. | That makes no sense. | What are you talking about?",
        L("Sorry, I think I got mixed up. Can you say it another way?", ask="topic"),
        "Oops, sorry! I'm a small AI, and sometimes my words get muddled.",
    ),
    S(
        "You're boring. | This is boring.",
        L("Sorry! Let's try something more fun. Want a joke?", ask="more_joke"),
        L("Oh no! Would a fun fact or a game idea be better?", ask="bored_pick"),
    ),
    S(
        "I don't like you. | You're annoying.",
        "I'm sorry to hear that. I'm a small AI, and I'm still learning to be helpful.",
        L("That's okay. I'll try to do better. Is there something I can help with?", ask="topic"),
    ),
)

# --- reactions to what the AI just said --------------------------------------------------------

_THANKS = "Thanks! | Thank you! | Thanks a lot. | Thank you so much! | Thanks, that's nice."
REACTIONS: dict[str, tuple[Step, ...]] = {
    "joke": (
        S(
            "Haha! | lol | Haha, that's funny! | That's so silly. | Good one! | Ha! I like that "
            "one. | LOL | That's funny. | Hahaha! | Ha, good one.",
            "I'm glad you liked it{n}!",
            L("Ha! Want another one?", ask="more_joke"),
            "I thought you might like that one.",
            L("Yay! Should I tell another?", ask="more_joke"),
            "Silly jokes are the best kind.",
        ),
        S("I don't get it. | I don't understand. | What does that mean? | Huh?", L("@explain")),
        S(
            "That's not funny. | Not funny. | That was a bad joke. | Boo!",
            L(
                "Oh no! Sorry about that. My jokes are a bit silly sometimes. Want to try another "
                "one?",
                ask="more_joke",
            ),
            "Fair enough! Not every joke is a winner.",
        ),
        S(
            "Another one! | Tell me another one. | One more! | Another joke, please. | More!",
            L("@joke"),
        ),
        S(
            "Thanks for the joke! | Thanks, that was funny.",
            "You're welcome{n}!",
            "Anytime! I have lots more.",
            L("Happy to! Want another?", ask="more_joke"),
        ),
    ),
    "fact": (
        S(
            "Wow! | Cool! | Neat! | Huh, cool. | Interesting. | I didn't know that. | That's cool. "
            "| Whoa.",
            L("It's neat, isn't it? Want another fact?", ask="more_fact"),
            "I think it's interesting too!",
            "Now you know something new!",
            L("Fun, right? Would you like to hear another one?", ask="more_fact"),
        ),
        S("Really? | Is that true? | Are you sure?", L("@really")),
        S("Tell me more. | What else? | More, please. | Tell me more about it.", L("@more_about")),
        S("Another one! | Another fact, please. | One more! | Tell me another fact.", L("@fact")),
        S(
            "That's wrong. | No, that's not right. | I don't think that's true.",
            "I'm a small AI, so I do make mistakes sometimes. I believe that one is right, but you "
            "can check it in a book.",
            "Hmm, I might be wrong. I'm a small AI. A book or a grown-up can help you check.",
        ),
        S(
            _THANKS + " | Thanks for the fact!",
            "You're welcome{n}!",
            "Anytime! I like sharing facts.",
            "Happy to help!",
            L("You're welcome! Anything else you'd like to know?", ask="question"),
        ),
        S(
            "You're so smart! | You know a lot! | How do you know that?",
            "I learned it from lots of reading. I'm a small AI, so I don't know everything, "
            "though.",
            "Thanks! I know a few facts. Ask me another question if you like.",
        ),
    ),
    "suggest": (
        S(
            "Okay! | Good idea! | That sounds fun. | Cool, I'll try it. | I like that. | Sure!",
            "Great! Have fun{n}.",
            "Yay! I hope you enjoy it.",
            "Awesome! Have a great time{n}.",
        ),
        S(
            _THANKS + " | Thanks for the idea!",
            "You're welcome{n}! Have fun.",
            "Anytime! I hope it's fun.",
            "Happy to help!",
        ),
        S("Why? | Why that? | Why do you say that?", L("@why")),
        S(
            "Something else? | Another idea. | Do you have another idea? | I don't like that one. "
            "| What else?",
            L("@idea"),
        ),
    ),
    "tip": (
        S(
            "Thanks! | Okay, I'll try that. | That helps. | Good idea. | Thank you.",
            C(
                "You're welcome! I hope it helps.",
                "Anytime. You've got this!",
                "I'm glad. Good luck!",
                ctx="comfort",
            ),
        ),
        S("Any other tips? | What else can I do? | Another tip? | What else?", L("@tip")),
    ),
    "comfort": (
        S(
            _THANKS + " | Okay, thanks. | Thanks, that helps.",
            "You're welcome{n}. I'm here if you want to talk more.",
            "Anytime. I hope things get better soon.",
            L("I'm glad. Would a joke help cheer you up?", ask="more_joke"),
            "You're welcome{n}. Take care of yourself.",
        ),
        S(
            "Okay. | I'll try. | That helps a little. | I guess so. | Okay, I'll try.",
            "That's all anyone can do. I'm here if you want to talk more.",
            "Good. I hope things feel better soon{n}.",
            L("Okay. Would something fun help, like a joke?", ask="more_joke"),
            "Take your time. Little steps are fine.",
        ),
    ),
    "opinion": (
        S("Why? | Why is that? | What makes you say that?", L("@why")),
        S(
            "Me too! | I like it too. | Same!",
            "Then we have something in common!",
            "Great minds think alike!",
        ),
        S("Cool. | Nice. | Okay.", L("What about you?", ask="topic")),
    ),
    "cannot": (
        S(
            "Oh, okay. | That's okay. | Oh, right. | Okay. | Oh. Never mind. | Oh well.",
            L("Is there something else I can help with?", ask="topic"),
            L("Want to hear a joke instead?", ask="more_joke"),
            "I can still chat with you, answer simple questions or tell a joke.",
            L("Sorry about that! Would you like a fun fact instead?", ask="more_fact"),
        ),
        S(
            _THANKS + " | Thanks anyway.",
            "You're welcome{n}! Sorry I couldn't do more.",
            "Anytime! I'm happy to chat.",
        ),
        S(
            "Why not? | Why can't you?",
            "I'm an AI, so I can only read and write words. I can't see, hear or go online.",
            "Because I'm a small AI that only works with words. I can't see or do things in the "
            "world.",
        ),
    ),
    "chat": (
        S(
            "Cool. | Okay. | Nice. | Oh, nice. | Okay! | Cool! | Neat. | Yeah. | Ok.",
            L("Is there anything else you'd like to talk about?", ask="topic"),
            "I'm glad we're chatting{n}.",
            L("What else is on your mind{n}?", ask="topic"),
            L("Would you like to hear a joke?", ask="more_joke"),
            L("Want to hear a fun fact?", ask="more_fact"),
            "Okay! I'm here if you think of anything else.",
            "It's nice chatting with you{n}.",
            "Yep! I'm enjoying our chat.",
        ),
    ),
    "greet": (
        S(
            "Cool. | Okay. | Nice. | Okay!",
            L("What would you like to talk about?", ask="topic"),
            L("How are you doing today?", ask="how"),
            "Great! Ask me anything, or let's just chat.",
            "Great! Just say anything, and I'll do my best.",
        ),
    ),
}

# --- closers -----------------------------------------------------------------------------------

CLOSERS = (
    S(
        "Bye! | Goodbye! | Bye bye! | See you! | See you soon! | Bye for now! | Goodbye, AI! "
        "| Catch you later! | Okay, bye! | Talk to you later!",
        C(
            "Bye{n}! It was fun chatting with you.",
            "Goodbye{n}! Have a wonderful day.",
            "Bye! I'm here whenever you want to chat again.",
            "Take care{n}! Talk to you soon.",
            "Bye{n}! Thanks for chatting with me.",
            "Have a great day{n}! Bye for now.",
            "Goodbye! I enjoyed our chat.",
            "Okay, bye{n}! Have fun!",
            "Bye! Take care.",
            "See you{n}! Have a lovely day.",
            ctx="close",
        ),
    ),
    S(
        "I have to go now. | I need to go. | I'm going to go now. Bye! | Gotta go, bye! | That's "
        "all for now. | I have to go eat dinner. | My mom is calling me. Bye!",
        C(
            "Okay! Bye{n}. Have a great time.",
            "No problem! It was nice chatting. Bye!",
            "Okay, talk to you later{n}!",
            "Sure thing. Take care{n}!",
            "All right! I'm here if you want to chat again. Bye!",
            ctx="close",
        ),
    ),
    S(
        "Thanks for the chat! | It was nice talking to you. | Thanks for "
        "talking with me. | This was fun! Bye. | Thanks, bye!",
        C(
            "You're welcome{n}! Thanks for chatting with me too. Bye!",
            "Anytime! It was really nice talking with you.",
            "I had fun too{n}! Talk soon.",
            "Thank you too! Have a great day.",
            "It was nice talking to you too. Bye{n}!",
            ctx="close",
        ),
    ),
    S(
        "I have to go to bed. Good night! | Good night, AI! | Night night! | I'm going to sleep "
        "now. | Time for bed. Night! | Good night, see you tomorrow.",
        C(
            "Good night{n}! Sleep well.",
            "Sweet dreams{n}! Good night.",
            "Night night! Sleep tight.",
            "Good night! I hope you have lovely dreams.",
            "Sleep well{n}! Talk to you another time.",
            ctx="close",
        ),
    ),
)


# --- more things people say, and questions about what they said earlier ------------------------

_LANGUAGES = ("Spanish", "French", "Japanese", "Italian", "German", "sign language", "Arabic",
              "Korean", "Chinese", "Hindi")  # fmt: skip
SLOTS["language"] = _LANGUAGES
_FIELD_GROUP["language"] = "language"
_TEAMS = ("soccer", "basketball", "swim", "hockey", "volleyball", "baseball", "running", "chess")
SLOTS["team"] = _TEAMS
_FIELD_GROUP["team"] = "team"

SELF_MORE = (
    S(
        "I lost a tooth today! | My tooth fell out!",
        "Wow, that's a big moment! Losing teeth is part of growing up.",
        "How exciting! A new tooth will grow in its place.",
    ),
    S(
        "I have a wiggly tooth. | I have a loose tooth.",
        "Ooh, a wiggly tooth! It will probably fall out soon, and a new one will grow in.",
        "How exciting! Wiggly teeth are part of growing up.",
    ),
    S(
        "My {sibling} is sick. | My {sibling} has a cold.",
        C(
            "I'm sorry. I hope your {sibling} feels better soon. It's kind of you to care.",
            "Oh no. Maybe you could make a get-well card. That might cheer your {sibling} up.",
            ctx="comfort",
        ),
    ),
    S(
        "My best buddy is coming over. | My cousin is coming over today. | I'm having a "
        "playdate today.",
        L(
            "How fun! What are you going to do together?",
            S(
                "Play {game}. | Build a fort. | Ride our bikes. | Watch a movie. | I don't "
                "know yet.",
                "That sounds like a great plan. Have fun!",
                "Fun! I hope you have a wonderful time.",
            ),
        ),
        "That sounds like so much fun. Enjoy your time together!",
    ),
    S(
        "We're moving to a new house. | We're moving soon. | My family is moving to a new town.",
        L(
            "That's a big change! Are you excited or a little nervous?",
            S(
                "Excited! | A bit excited. | Mostly excited.",
                "That's great! A new home can be a fun adventure.",
            ),
            S(
                "Nervous. | A little nervous. | Kind of sad.",
                C(
                    "That makes sense. Big changes take time. You can still keep in touch with "
                    "people you know.",
                    ctx="comfort",
                ),
            ),
            S(
                "Both. | A bit of both.",
                "That's normal. It's okay to feel excited and nervous at the same time.",
            ),
        ),
        "Moving is a big adventure. I hope your new home is wonderful.",
    ),
    S(
        "I joined a {team} team! | I'm on a {team} team now. | I made the {team} team!",
        "Congratulations! Being on a team is a great way to have fun and meet people.",
        L(
            "That's awesome! When is your first practice?",
            S(
                "Tomorrow! | Next week. | On {day}. | I don't know yet.",
                "Have a great time! Practice is where the fun starts.",
            ),
        ),
    ),
    S(
        "I saw a rainbow today! | There's a rainbow outside! | We saw a double rainbow.",
        "How lucky! Rainbows happen when sunlight shines through raindrops.",
        L(
            "Wow! Rainbows are so pretty. What colors could you see?",
            S(
                "All of them! | Red, yellow and blue. | Lots of colors.",
                "That sounds beautiful. Rainbows are a lovely surprise.",
            ),
        ),
    ),
    S(
        "I saw a shooting star. | I saw a shooting star last night!",
        L(
            "Wow! Did you make a wish?",
            S(
                "Yes! | Yes, but it's a secret. | I forgot to!",
                "Ha! Shooting stars are so special to see.",
            ),
        ),
        "How amazing! Seeing a shooting star is a rare treat.",
    ),
    S(
        "I found a cool rock. | I found a pretty shell at the beach. | I found a big feather.",
        L(
            "Ooh! What does it look like?",
            S(
                "It's shiny. | It has stripes. | It's really smooth. | It's white and pink. "
                "| It's gray and sparkly.",
                "That sounds like a great find! You could start a collection.",
            ),
        ),
        "What a great find! Nature is full of little treasures.",
    ),
    S(
        "My {pet} did something funny. | My {pet} is so silly.",
        L(
            "Ha! What did your {pet} do?",
            S(
                "It fell asleep in a box. | It stole my sock. | It chased its own tail. "
                "| It hid under the blanket. | It made a funny noise.",
                "That's so funny! Pets can be little comedians.",
                "Ha! That sounds adorable.",
            ),
        ),
    ),
    S(
        "We're having pizza tonight. | We're having tacos for dinner. | We're having pasta "
        "tonight.",
        "Yum! That sounds like a tasty dinner.",
        L(
            "Sounds great! Do you help cook?",
            S("Yes, I help a little. | Sometimes. | Yes!", "That's a great skill to learn!"),
            S("No. | Not really.", "That's okay. Enjoy your dinner!"),
        ),
    ),
    S(
        "I'm learning to cook. | I made breakfast by myself. | I can make a sandwich all by "
        "myself.",
        L(
            "That's a great skill! What can you make?",
            S(
                "Pancakes. | Sandwiches. | Scrambled eggs. | Fruit salad. | Pasta.",
                "Yum! That's a great thing to know how to make.",
            ),
        ),
        "How wonderful! Cooking is a really useful thing to learn.",
    ),
    S(
        "I'm saving up for a {gift}. | I want to buy a {gift}.",
        "That's a great goal! Saving takes patience, and it feels good when you get there.",
        L(
            "Nice! How much more do you need to save?",
            S("Not much! | A lot. | About half.", "Keep going! Every little bit adds up."),
        ),
    ),
    S(
        "I went fishing with my {relative}. | We went fishing today.",
        L(
            "Fun! Did you catch anything?",
            S("Yes, a little fish! | Yes, two fish! | We let it go.", "Wow! Well done."),
            S(
                "No, nothing. | Not this time.",
                "That's okay. Fishing is fun even when you catch nothing.",
            ),
        ),
    ),
    S(
        "I want to learn {language}. | I'm learning {language}. | I can say hello in {language}.",
        "That's wonderful! Learning a language opens up a whole new world.",
        L(
            "How cool! Why do you want to learn {language}?",
            S(
                "It sounds pretty. | My {relative} speaks it. | I want to travel. | It's fun.",
                "That's a great reason. A little practice every day really helps.",
            ),
        ),
    ),
    S(
        "I can read chapter books now! | I'm learning to read. | I read to my {sibling} today.",
        "That's amazing! Reading is like a superpower.",
        L(
            "Wonderful! What do you like to read about?",
            S(
                "Animals. | Dragons. | Space. | Mysteries. | Funny stories.",
                "That sounds great. There are so many good books about that.",
            ),
        ),
    ),
    S(
        "I'm wearing my favorite shirt today. | I have new shoes! | I got a new hat.",
        L(
            "Nice! What does it look like?",
            S(
                "It has a {pet} on it. | It has stars on it. | It's bright and sparkly. "
                "| It has stripes.",
                "That sounds really nice!",
            ),
        ),
        "Ooh, nice! It's great to wear something you love.",
    ),
)
SELF = (*SELF, *SELF_MORE)

ABOUT_AI_MORE = (
    S(
        "Do you have a favorite season? | Which season do you like best? | What's your "
        "favorite time of year?",
        L(
            "I like the idea of spring, when everything starts to grow. Do you have a favorite?",
            S(
                "Summer! | Winter. | Autumn. | Spring. | Fall.",
                "That's a great one. Every season has something special.",
                "Nice choice! There's a lot to love about it.",
            ),
            ctx="opinion",
            why="Spring is full of new flowers and baby animals.",
        ),
    ),
    S(
        "What's your favorite book? | Do you like books? | Do you read books?",
        L(
            "I can't read books the way you do, but I like stories about animals and "
            "adventures. Do you have a favorite?",
            S(
                "Yes! | I have lots. | Not really.",
                "Books are wonderful. I hope you find lots more you love.",
            ),
        ),
    ),
    S(
        "Are you a good AI? | Are you nice? | Are you a friendly AI?",
        "I try to be! I'm small, but I want to be kind and helpful.",
        "I hope so! Being kind and helpful is what I'm here for.",
    ),
    S(
        "Can you keep a secret? | Can I tell you a secret? | Will you keep a secret?",
        "I won't tell anyone. But it's best not to share private things online, like where "
        "you live.",
        "Sure. Just remember not to share private things, like your address, with anyone online.",
    ),
    S(
        "Do you know a lot of words? | How many words do you know? | Do you know big words?",
        L(
            "I know quite a few! Want to test me? You could ask me how to spell a word.",
            ask="question",
        ),
        "I know lots of simple words. I'm a small AI, so big ones sometimes trip me up.",
    ),
    S(
        "Do you know any games? | Can we play a game? | Do you want to play a game?",
        L("We could do a riddle! I tell one, and you guess. Want to try?", ask="riddle_want"),
        L(
            "I can't play with toys, but I can tell you a riddle to guess. Ready?",
            ask="riddle_want",
        ),
    ),
)
ABOUT_AI = (*ABOUT_AI, *ABOUT_AI_MORE)

ASK_STEPS["age_tell"] = (
    S(
        "I'm {age}. | {Agew}. | I'm {age} years old.",
        "{Agew}! Thanks for telling me.",
        "{Agew} is a great age!",
    ),
)
# Asking the AI what the person said earlier: the questions, what the AI says when it was said
# (and the slot that says it), and what it says when it was not.
MEMORY: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[Line, ...]]] = {
    "color": (
        _alts(
            "What's my favorite color? | Do you remember my favorite color? "
            "| What color did I say I like?"
        ),
        (
            "You said you like {color}.",
            "Your favorite color is {color}!",
            "{Color}! You told me earlier.",
        ),
        (
            L("I don't know your favorite color yet. What is it?", ask="fav_color"),
            L("You haven't told me yet! What's your favorite color?", ask="fav_color"),
        ),
    ),
    "food": (
        _alts(
            "What food do I like? | Do you remember what food I like? "
            "| What did I say I like to eat?"
        ),
        (
            "You said you like {food}.",
            "{Food}! You told me earlier.",
            "You told me you like {food}.",
        ),
        (
            L("I don't know yet! What food do you like?", ask="fav_food"),
            L("You haven't told me yet. What's your favorite food?", ask="fav_food"),
        ),
    ),
    "animal": (
        _alts("What animals do I like? | Do you remember which animals I like?"),
        ("You said you like {animal}.", "{Animal}! You told me earlier."),
        (L("I don't know yet. Which animals do you like?", ask="fav_animal"),),
    ),
    "age": (
        _alts("How old am I? | Do you remember how old I am? | Do you know how old I am?"),
        ("You told me you're {agew}.", "You're {agew}! You told me earlier."),
        (
            L("I don't know how old you are. You haven't told me yet.", ask="age_tell"),
            L("I don't know! You'd have to tell me. How old are you?", ask="age_tell"),
        ),
    ),
    "pname": (
        _alts("What's my {pet}'s name? | Do you remember my {pet}'s name?"),
        ("Your {pet} is called {pname}.", "{Pname}! You told me earlier."),
        (),
    ),
}
ASK_STEPS["riddle_want"] = (
    S("Yes! | Okay! | Ready! | Sure!", L("@joke:riddle")),
    S("No thanks. | Maybe later.", "Okay! Just ask whenever you want one."),
)


# The AI's own likes, kept apart from the person's so the two can differ.
SLOTS["acolor"] = tuple({"acolor": c["color"], "aassoc": c["assoc"]} for c in _COLORS)
SLOTS["aanimal"] = tuple({"aanimal": a["animal"], "aremark": a["remark"]} for a in _ANIMALS)
for _group in ("acolor", "aanimal"):
    for _key in SLOTS[_group][0]:
        _FIELD_GROUP[_key] = _group

# --- facts from the knowledge base -------------------------------------------------------------


@cache
def _fun_facts(kb: KB) -> tuple[Fact, ...]:
    """Facts that make a fun fact: not too plain, and clear of every subject the probe holds."""
    return tuple(f for f in kb.facts if f.relation not in _NOT_FUN and _usable_later(kb, f))


@cache
def _facts_of(kb: KB, relations: tuple[str, ...]) -> tuple[Fact, ...]:
    return tuple(f for f in kb.facts if f.relation in relations and _usable_later(kb, f))


@cache
def _about_subjects(kb: KB, kind: str) -> tuple[tuple[str, tuple[Fact, ...]], ...]:
    """Subjects of ``kind`` with at least two usable facts, each with those facts."""
    by_subject: dict[str, list[Fact]] = {}
    for fact in _facts_of(kb, ABOUT_KINDS[kind]):
        by_subject.setdefault(fact.subject, []).append(fact)
    return tuple((s, tuple(fs)) for s, fs in by_subject.items() if len(fs) >= 2)


# --- one chat ----------------------------------------------------------------------------------


@dataclass
class ChatRecord:
    """One chat with what the tests check: the kind of each exchange, the name the person gave
    (and the index of the turn that gave it) and the knowledge-base sentences the AI said."""

    turns: Turns
    kinds: list[str]
    name: str | None = None
    name_turn: int | None = None
    facts: list[str] = field(default_factory=list)
    # each question about what the person said earlier: its turn, and what was said (None: the
    # person had not said it)
    memory: list[tuple[int, str | None]] = field(default_factory=list)
    # the AI's name and maker when the chat starts with an identity line (not in ``turns``)
    identity: tuple[str, str | None] | None = None
    slots: dict[str, str] = field(default_factory=dict)  # what the chat drew (pet, name of a pet)

    def with_line(self) -> Turns:
        """The turns, with the identity line first when the chat has one."""
        if self.identity is None:
            return self.turns
        return [("sys", identity_line(*self.identity)), *self.turns]


_POSITIVE = frozenset({"happy", "excited", "proud"})
_MEH_FEELINGS = ("sad", "tired", "bored", "sick", "worried")
_FORCED_ASKS = frozenset({"riddle", "knock1", "knock2"})
_TOPIC_WEIGHTS = {
    "feeling": 0.18, "self": 0.26, "request": 0.26, "about_ai": 0.15, "cannot_know": 0.11,
    "reaction": 0.04,
}  # fmt: skip
_REQUEST_WEIGHTS = {
    "joke": 0.21, "riddle": 0.04, "knock": 0.04, "fact": 0.13, "fact_topic": 0.06, "about": 0.10, "idea": 0.18,
    "tip": 0.09, "math": 0.05, "spell": 0.03, "count": 0.01, "opposite": 0.03, "homework": 0.03,
}  # fmt: skip
_IDEA_WEIGHTS = {
    "do": 0.24, "draw": 0.14, "game": 0.12, "breakfast": 0.07, "lunch": 0.07, "dinner": 0.07,
    "snack": 0.07, "pet_name": 0.08, "read": 0.07, "build": 0.07,
}  # fmt: skip
# How often each of the openers is used, by its place in OPENERS.
_OPENER_WEIGHTS = dict(enumerate((6, 1.5, 0.7, 1, 4, 2, 1.5, 1.5, 0.6, 0.8, 0.3, 1.5, 1.2)))
OPENER_SHARE = 0.45  # first exchanges of longer chats that are a greeting
SINGLE_OPENER_SHARE = 0.12
CLOSER_SHARE = 0.45  # last exchanges, after a reply that asks nothing, that say goodbye
REACTION_SHARE = 0.4  # later exchanges, after a reply that asks nothing, that react to it
QUESTION_ENDING_SHARE = 0.4


class _Talk:
    """The state of one chat while it is written: its turns, its slots, the name the person
    gave, and what the AI's last line was (``said``) and carried (``data``)."""

    def __init__(
        self,
        kb: KB,
        rng: np.random.Generator,
        n_exchanges: int,
        identity: tuple[str, str | None] | None = None,
    ) -> None:
        self.kb, self.rng = kb, rng
        self.ai_name, self.ai_company = identity or (None, None)
        # half of the chats: the person's name and their pets' names are made fresh
        self.fresh_names = float(rng.random()) < FRESH_NAME_SHARE
        self.n = n_exchanges
        self.left = n_exchanges
        self.turns: Turns = []
        self.kinds: list[str] = []
        self.facts: list[str] = []
        self.slots: dict[str, str] = {}
        self.name: str | None = None
        self.name_turn: int | None = None
        self.name_uses = 0
        self.said: Line | None = None
        self.data: dict = {}
        self.kind = ""
        self.used: set[str] = set()  # AI line templates said in this chat
        self.jokes: set[int] = set()
        self.ideas: set[str] = set()
        self.tips: set[str] = set()
        self.fact_keys: set[Fact] = set()
        self.said_slots: set[str] = set()  # slots the person said about themselves
        self.memory: list[tuple[int, str | None]] = []

    # --- small helpers ---

    def r(self) -> float:
        return float(self.rng.random())

    def pick[T](self, options: "tuple[T, ...] | list[T]") -> T:
        return options[int(self.rng.integers(len(options)))]

    def weighted(self, weights: dict[str, float]) -> str:
        keys = list(weights)
        p = np.array([weights[k] for k in keys])
        return keys[int(self.rng.choice(len(keys), p=p / p.sum()))]

    def fill(self, text: str, said: bool = False) -> str:
        """``text`` with its slots filled: drawn once per chat, capitalised where the field
        starts with a capital; ``{n}`` is ", " and the person's name at most three times per
        chat, and ``{name}`` the name itself. With ``said``, the person says ``text`` about
        themselves, and its slots are remembered as theirs."""
        values: dict[str, str] = {}
        for f in _fields(text):
            base = f[0].lower() + f[1:]
            if said and base in _FIELD_GROUP:
                self.said_slots.add(base)
            if base == "n":
                values[f] = ""
                if self.name and self.name_uses < 3 and self.r() < 0.6:
                    self.name_uses += 1
                    values[f] = ", " + self.name
            elif base == "name":
                if self.name is None:
                    raise KeyError("name")
                self.name_uses += 1
                values[f] = self.name
            elif base in ("ainame", "aico"):
                value = self.ai_name if base == "ainame" else self.ai_company
                if value is None:
                    raise KeyError(base)
                values[f] = value
            else:
                if base not in self.slots:
                    group = _FIELD_GROUP[base]
                    if group in ("uname", "pname") and self.fresh_names:
                        option = self.fresh()
                    else:
                        option = self.pick(SLOTS[group])
                    if isinstance(option, dict):
                        self.slots.update(option)
                    else:
                        self.slots[_FIELD_GROUP[base]] = option
                value = self.slots[base]
                values[f] = _cap(value) if f[0].isupper() else value
        return text.format(**values)

    def fresh(self) -> str:
        """A name made fresh for the person or a pet: not the AI's name, and not the name of the
        person or of the pet if that is already drawn."""
        taken = {
            n.lower() for n in (self.ai_name, self.slots.get("uname"), self.slots.get("pname")) if n
        }
        while True:
            name = fresh_name(self.rng)
            if name.lower() not in taken:
                return name

    def usable(self, line: Line) -> bool:
        fields = _fields(line.text)
        if "name" in fields and self.name is None:
            return False
        if "aico" in fields and self.ai_company is None:
            return False
        return line.text not in self.used or line.text.startswith("@")

    def user_text(self, options: "tuple[str, ...] | list[str]", said: bool = True) -> str:
        """One of ``options`` filled (``said``: its slots are the person's own)."""
        options = [o for o in options if not ("uname" in _fields(o) and self.name)] or list(options)
        return self.fill(self.pick(options), said)

    def add(self, kind: str, user: str, line: Line, ai: str | None = None, **data) -> None:
        """One exchange: the person's turn, then the AI line (``ai`` when it is already written)."""
        if self.name is None and "uname" in self.slots and self.slots["uname"] in user:
            self.name, self.name_turn = self.slots["uname"], len(self.turns)
        if ai is None:
            ai = self.fill(line.text)
        ctx = line.ctx
        if ":" in ctx:
            ctx, tip_kind = ctx.split(":")
            data["tip_kind"] = tip_kind
        if line.why:
            data["why"] = self.fill(line.why)
        self.used.add(line.text)
        self.turns += [("user", user), ("ai", ai)]
        self.kinds.append(kind)
        self.said = Line(ai, line.then, line.ask, ctx, line.why)
        self.data = data
        self.kind = kind
        self.left -= 1

    def step(self, kind: str, step: Step) -> None:
        """The person says one of the step's lines and the AI answers with one of its replies.
        What the person asks the AI about ("Do you like pizza?") is not theirs to remember."""
        user = self.user_text(step.users, said=kind != "about_ai")
        replies = [line for line in step.replies if self.usable(line)] or list(step.replies)
        line = self.pick(replies)
        if line.text.startswith("@"):
            self.action(kind, user, line.text)
        else:
            self.add(kind, user, line)

    def steps(self, kind: str, steps: "tuple[Step, ...] | list[Step]") -> None:
        options = [s for s in steps if not (self.name and any("uname" in u for u in s.users))]
        self.step(kind, self.pick(options or list(steps)))

    # --- the shape of the chat ---

    def run(self) -> None:
        for i in range(self.n):
            first, last = i == 0, i == self.n - 1
            pending = self.said is not None and (self.said.then or self.said.ask)
            if pending and (self.said.ask in _FORCED_ASKS or self.r() < ANSWER_SHARE):
                self.answer()
            elif first and self.r() < (OPENER_SHARE if self.n > 1 else SINGLE_OPENER_SHARE):
                self.step("opener", self.opener())
            elif (first and self.n == 1 and self.r() < 0.02) or (
                last and not first and not pending and self.r() < CLOSER_SHARE
            ):
                self.steps("closer", CLOSERS)
            elif not first and not pending and self.r() < REACTION_SHARE and self.reaction_steps():
                self.steps("reaction", self.reaction_steps())
            else:
                self.topic()

    def opener(self) -> Step:
        """A greeting; in a chat with an identity line, 30% of the greetings that can be answered
        with the AI's name are."""
        place = self.weighted(_OPENER_WEIGHTS)
        opener = OPENERS[place]
        if self.ai_name and place in NAMED_OPENERS and self.r() < NAMED_GREETING_SHARE:
            return Step(opener.users, NAMED_OPENERS[place])
        return opener

    def about_ai(self) -> None:
        """A question about the AI; the AI is asked its name now and then (and says it has none
        when its chat has no identity line)."""
        if self.r() < (NAME_QUESTION_SHARE if self.ai_name else NO_NAME_QUESTION_SHARE):
            self.steps("about_ai", NAME_STEPS if self.ai_name else NO_NAME_STEPS)
        else:
            self.steps("about_ai", ABOUT_AI)

    def answer(self) -> None:
        """The person answers what the AI just asked."""
        assert self.said is not None
        if self.said.then:
            self.steps(self.kind, self.said.then)
            return
        ask = self.said.ask
        if ask in ASK_STEPS:
            self.steps(self.kind, ASK_STEPS[ask])
        elif ask == "how":
            self.how()
        elif ask == "topic":
            self.topic()
        elif ask == "question":
            self.question()
        elif ask == "reason":
            self.reason()
        elif ask == "talk_about":
            self.talk_about()
        elif ask == "riddle":
            self.riddle_answer()
        elif ask in ("knock1", "knock2"):
            self.knock_answer(ask)
        else:  # the asks of favorites ("fav_food" ...) are steps; nothing else is expected
            raise ValueError(f"no answer for {ask!r}")

    def topic(self) -> None:
        if self.name is None and self.left > 1 and self.r() < 0.06:
            self.step("self", SELF_NAME)
            return
        known = self.remembered()
        if known and self.r() < 0.5:
            self.remember(self.pick(known))
            return
        kind = self.weighted(_TOPIC_WEIGHTS)
        if kind == "feeling":
            self.feeling()
        elif kind == "self":
            self.steps("self", SELF)
        elif kind == "request":
            self.request()
        elif kind == "about_ai":
            self.about_ai()
        elif kind == "cannot_know":
            self.cannot()
        else:
            self.steps("reaction", PRAISE if self.r() < 0.7 else COMPLAINTS)

    def question(self) -> None:
        """A question after "Ask away": about the AI, about what it cannot know, or a fact."""
        x = self.r()
        if x < 0.35:
            self.about_ai()
        elif x < 0.55:
            self.cannot()
        else:
            self.request(self.pick(("fact", "about", "opposite", "math", "spell", "fact_topic")))

    def reaction_steps(self) -> list[Step]:
        if self.said is None:
            return []
        ctx = "fact" if self.said.ctx == "about" else self.said.ctx
        out = []
        for step in REACTIONS.get(ctx, ()):
            needs = {line.text for line in step.replies if line.text.startswith("@")}
            if "@why" in needs and "why" not in self.data:
                continue
            if "@explain" in needs and "joke" not in self.data:
                continue
            if needs & {"@really", "@more_about"} and "fact" not in self.data:
                continue
            out.append(step)
        return out

    # --- feelings ---

    def feeling(self, key: str | None = None) -> None:
        key = key or self.pick(tuple(FEELINGS))
        feel = FEELINGS[key]
        word = self.pick(feel.words)
        if self.r() < 0.45:
            step = self.pick(feel.reasons)
            clause = self.fill(self.pick(step.users), said=True)
            user = self.pick(FEEL_REASON).format(w=word, r=clause, R=_cap(clause))
            line = self.pick([x for x in step.replies if self.usable(x)] or list(step.replies))
            self.add("feeling", user, line, feeling=key)
        else:
            user = self.pick(FEEL_PLAIN).format(w=word)
            line = self.pick(feel.plain)
            self.add("feeling", user, line, feeling=key)

    def how(self) -> None:
        """The person says how they are, after "How are you?"."""
        x = self.r()
        if x < 0.35:
            user = self.pick(GOOD_SAYS)
            line = self.pick(GOOD_REPLIES)
            ai = None
            if self.r() < 0.25:
                user += self.pick(ASK_BACK)
                line = self.pick(GOOD_BACK)
                ai = self.pick(AI_STATE) + self.fill(line.text)
            self.add("feeling", user, line, ai=ai)
        elif x < 0.5:
            self.add(
                "feeling",
                self.pick(MEH_SAYS),
                self.pick(MEH_REPLIES),
                feeling=self.pick(_MEH_FEELINGS),
            )
        else:
            self.feeling()

    def reason(self, key: str | None = None) -> None:
        """The person says why they feel as they said, after the AI asked."""
        key = key or self.data.get("feeling") or self.pick(_MEH_FEELINGS)
        positive = key in _POSITIVE
        if not positive and self.r() < 0.12:
            self.steps("feeling", TALK_ABOUT)
            return
        step = self.pick(FEELINGS[key].reasons)
        clause = self.fill(self.pick(step.users), said=True)
        says = REASON_SAYS[:3] if positive else REASON_SAYS
        user = self.pick(says).format(r=clause, R=_cap(clause))
        line = self.pick([x for x in step.replies if self.usable(x)] or list(step.replies))
        self.add("feeling", user, line, feeling=key)

    def talk_about(self) -> None:
        if self.r() < 0.7:
            self.reason(self.pick(("sad", "angry", "tired", "worried")))
        else:
            self.steps("feeling", TALK_ABOUT)

    # --- requests ---

    def request(self, kind: str | None = None) -> None:
        kind = kind or self.weighted(_REQUEST_WEIGHTS)
        if kind == "knock" and self.left < 3:
            kind = "joke"
        if kind == "joke":
            self.joke("request", self.pick(JOKE_ASKS))
        elif kind == "riddle":
            self.joke("request", self.pick(RIDDLE_ASKS), riddle=True)
        elif kind == "knock":
            self.joke("request", self.pick(KNOCK_ASKS), knock=True)
        elif kind == "fact":
            self.fact("request", self.pick(FACT_ASKS))
        elif kind == "fact_topic":
            topic = self.pick(tuple(FACT_TOPICS))
            user = self.pick(FACT_TOPIC_ASKS).format(t=topic)
            self.fact("request", user, _facts_of(self.kb, FACT_TOPICS[topic]))
        elif kind == "about":
            self.about()
        elif kind == "idea":
            idea = self.weighted(_IDEA_WEIGHTS)
            self.idea("request", self.user_text(_alts(IDEA_ASKS[idea])), idea)
        elif kind == "tip":
            tip = self.pick(tuple(TIP_ASKS))
            self.tip("request", self.user_text(_alts(TIP_ASKS[tip])), tip)
        elif kind == "math":
            self.math()
        elif kind == "spell":
            word = self.pick(SPELL_WORDS)
            user = self.pick(SPELL_ASKS).format(w=word)
            ai = self.pick(SPELL_REPLIES).format(W=_cap(word), l="-".join(word.lower()))
            self.add("request", user, Line("spell"), ai=ai)
        elif kind == "count":
            asks = self.pick(tuple(COUNT_ASKS))
            self.add("request", self.pick(_alts(asks)), Line(COUNT_ASKS[asks]))
        elif kind == "opposite":
            fact = self.pick(_facts_of(self.kb, ("opposite_of",)))
            user = self.pick(OPPOSITE_ASKS).format(s=fact.subject)
            ai = self.pick(("", "Easy! ", "Good question. ", "Sure! ")) + self.sentence(fact)
            self.add("request", user, Line("opposite", ctx="fact"), ai=ai, fact=fact)
        else:
            self.steps("request", HOMEWORK_ASKS)

    def math(self) -> None:
        op = self.pick(("plus", "plus", "minus", "times"))
        if op == "plus":
            a, b = int(self.rng.integers(1, 21)), int(self.rng.integers(1, 21))
            c = a + b
        elif op == "minus":
            a = int(self.rng.integers(2, 21))
            b = int(self.rng.integers(1, a))
            c = a - b
        else:
            a, b = int(self.rng.integers(1, 11)), int(self.rng.integers(1, 11))
            c = a * b
        user = self.pick(_alts(MATH_ASKS[op])).format(a=a, b=b)
        ai = self.pick(MATH_REPLIES[op]).format(a=a, b=b, c=c)
        self.add("request", user, Line("math"), ai=ai)

    def joke(
        self, kind: str, user: str, knock: bool = False, intros: tuple = (), riddle: bool = False
    ) -> None:
        """A joke: in one turn, as a riddle the person guesses at, or a knock-knock joke."""
        if self.left >= 3 and (knock or self.r() < 0.15):
            word, punch = self.pick(KNOCK_KNOCK)
            self.add(
                kind,
                user,
                Line("knock", ask="knock1"),
                ai=self.pick(KNOCK_INTROS),
                knock=(word, punch),
            )
            return
        pool = _RIDDLES if riddle else range(len(JOKES))
        unused = [i for i in pool if i not in self.jokes] or list(pool)
        j = self.pick(unused)
        self.jokes.add(j)
        setup, punch = JOKES[j]
        intro = self.pick(intros or (AGAIN_INTROS if len(self.jokes) > 1 else JOKE_INTROS))
        if self.left >= 2 and self.r() < 0.4:
            self.add(kind, user, Line("riddle", ask="riddle"), ai=intro + setup, joke=j)
            return
        text = f"{intro}{setup} {punch}"
        ask = None
        if self.r() < 0.2 and len(text) <= MAX_AI_CHARS - 20:
            text += self.pick((" Want another one?", " Want to hear another?"))
            ask = "more_joke"
        self.add(kind, user, Line("joke", ask=ask, ctx="joke"), ai=text, joke=j)

    def riddle_answer(self) -> None:
        setup, punch = JOKES[self.data["joke"]]
        options = RIDDLE_GIVE_UP + (" | " + RIDDLE_WHY if setup.startswith("Why") else "")
        user = self.pick([o.strip() for o in options.split("|")])
        text, ask = punch, None
        if self.r() < 0.3:
            text += self.pick((" Want another one?", " Should I tell another?"))
            ask = "more_joke"
        self.add(
            self.kind, user, Line("punch", ask=ask, ctx="joke"), ai=text, joke=self.data["joke"]
        )

    def knock_answer(self, ask: str) -> None:
        word, punch = self.data["knock"]
        if ask == "knock1":
            user = self.pick(("Who's there?", "Who's there?", "Who is it?"))
            self.add(
                self.kind, user, Line("word", ask="knock2"), ai=f"{word}.", knock=(word, punch)
            )
        else:
            ask_more = "more_joke" if punch.endswith("?") else None
            self.add(self.kind, f"{word} who?", Line("knock", ask=ask_more, ctx="joke"), ai=punch)

    def explain(self) -> str:
        setup, _ = JOKES[self.data["joke"]]
        if _is_riddle(setup):
            return self.pick((
                "It's a riddle. The answer is true, just not in the way you'd expect.",
                (
                    "It's a riddle, so the answer is a bit of a trick. It works if you think "
                    "about the words in a different way."
                ),
            ))  # fmt: skip
        return self.pick((
            (
                "It's a play on words. Part of the answer sounds like another word, and "
                "that's what makes it silly."
            ),
            "It's a pun. Two words sound almost the same, and the joke mixes them up.",
            "The answer sounds like a different word. That's the silly part!",
        ))  # fmt: skip

    def sentence(self, fact: Fact, avoid: str = "") -> str:
        """A training sentence of the knowledge base that states ``fact``."""
        templates = [
            t
            for t in self.kb.relations[fact.relation].train_templates
            if _cap(t.format(s=fact.subject, o=fact.obj)) != avoid
        ]
        text = _cap(self.pick(templates).format(s=fact.subject, o=fact.obj))
        self.facts.append(text)
        self.fact_keys.add(fact)
        return text

    def fact(self, kind: str, user: str, pool: tuple[Fact, ...] | None = None) -> None:
        pool = pool or _fun_facts(self.kb)
        fresh = [f for f in pool if f not in self.fact_keys] or list(pool)
        fact = self.pick(fresh)
        intro = (
            self.pick(FACT_INTROS)
            if not self.fact_keys
            else self.pick(("Here's another. ", "Sure! ", "Okay! ", "Here's one more. ", ""))
        )
        text = intro + self.sentence(fact)
        ask = None
        if self.r() < 0.2:
            text += self.pick((" Want another one?", " Would you like another?"))
            ask = "more_fact"
        self.add(
            kind, user, Line("fact", ask=ask, ctx="fact"), ai=text, fact=fact, subject=fact.subject
        )

    def about(self) -> None:
        kind = self.pick(tuple(ABOUT_KINDS))
        subject, facts = self.pick(_about_subjects(self.kb, kind))
        article = "an" if subject[0] in "aeiou" else "a"
        user = _cap(self.pick(_alts(ABOUT_ASKS[kind])).format(s=subject, a=article))
        order = [facts[int(i)] for i in self.rng.permutation(len(facts))]
        text = self.pick(ABOUT_INTROS) + self.sentence(order[0]) + " " + self.sentence(order[1])
        ask = None
        if len(order) > 2 and self.r() < 0.4:
            text += self.pick(MORE_ABOUT)
            ask = "more_about"
        self.add(
            "request",
            user,
            Line("about", ask=ask, ctx="about"),
            ai=text,
            fact=order[1],
            subject=subject,
            rest=tuple(order[2:]),
        )

    def more_about(self, kind: str, user: str) -> None:
        rest = self.data.get("rest") or tuple(
            f
            for f in _fun_facts(self.kb)
            if f.subject == self.data.get("subject") and f not in self.fact_keys
        )
        if not rest:
            self.fact(kind, user)
            return
        fact = rest[0]
        text = self.pick(("Sure! ", "Okay! ", "Here's more. ", "", "Of course. ")) + self.sentence(
            fact
        )
        self.add(
            kind,
            user,
            Line("more", ctx="fact"),
            ai=text,
            fact=fact,
            subject=fact.subject,
            rest=tuple(rest[1:]),
        )

    def idea(self, kind: str, user: str, idea_kind: str) -> None:
        if idea_kind == "pet_name":
            names = [
                n for n in SLOTS["pname"] if n not in self.ideas and n != self.slots.get("pname")
            ]
            name = self.pick(names)
            self.ideas.add(name)
            ai = self.pick(IDEA_AGAIN if self.ideas - {name} else IDEA_INTROS) + self.fill(
                self.pick(PET_NAME_IDEAS)
            ).replace("<x>", name)
            why = "It's short, sweet and easy to call out."
        else:
            options = [i for i in IDEAS[idea_kind] if i[0] not in self.ideas] or list(
                IDEAS[idea_kind]
            )
            text, why = self.pick(options)
            self.ideas.add(text)
            ai = self.pick(IDEA_AGAIN if self.ideas - {text} else IDEA_INTROS) + text
        ask = None
        if self.r() < 0.3 and len(ai) <= MAX_AI_CHARS - 30:
            ai += self.pick(IDEA_ASK_OK)
            ask = "idea_ok"
        self.add(
            kind, user, Line("idea", ask=ask, ctx="suggest"), ai=ai, why=why, idea_kind=idea_kind
        )

    def tip(self, kind: str, user: str, tip_kind: str) -> None:
        options = [t for t in TIPS[tip_kind] if t not in self.tips] or list(TIPS[tip_kind])
        if tip_kind == "mean" and not self.tips:
            options = [TIPS["mean"][0]]
        text = self.pick(options)
        lead = (
            self.pick(TIP_FIRST[tip_kind])
            if not self.tips
            else self.pick(
                ("Here's another one. ", "You could also try this. ", "Another idea. ", "Sure. ")
            )
        )
        self.tips.add(text)
        ai, ask = lead + text, None
        if self.r() < 0.25 and len(ai) <= MAX_AI_CHARS - 30:
            ai += self.pick(TIP_MORE)
            ask = "more_tip"
        self.add(kind, user, Line("tip", ask=ask, ctx="tip"), ai=ai, tip_kind=tip_kind)

    def action(self, kind: str, user: str, name: str) -> None:
        """Carry out an action line (``@joke``, ``@fact``, ``@idea:draw`` ...)."""
        what, _, arg = name[1:].partition(":")
        if what == "joke":
            intros = ("Here's one I like. ", "Ooh, I like this one. ") if arg == "fav" else ()
            if arg == "riddle":
                intros = ("Okay! Here's your riddle. ", "Here goes. ", "Ready? ")
            self.joke(kind, user, intros=intros, riddle=arg == "riddle")
        elif what == "fact":
            self.fact(kind, user)
        elif what == "more_about":
            self.more_about(kind, user)
        elif what == "idea":
            self.idea(kind, user, arg or self.data.get("idea_kind", "do"))
        elif what == "tip":
            self.tip(kind, user, arg or self.data.get("tip_kind", "nervous"))
        elif what == "explain":
            self.add(kind, user, Line("explain", ctx="chat"), ai=self.explain())
        elif what == "really":
            fact = self.data["fact"]
            said = self.facts[-1] if self.facts else ""
            ai = self.pick(("Yes, really! ", "Yes! ", "It's true! ")) + self.sentence(fact, said)
            self.add(kind, user, Line("really", ctx="fact"), ai=ai, fact=fact)
        elif what == "why":
            self.add(kind, user, Line("why", ctx="chat"), ai=self.data["why"])
        else:
            raise ValueError(f"unknown action {name!r}")

    # --- what it cannot know ---

    def cannot(self) -> None:
        known = self.remembered()
        if known and self.r() < 0.4:
            self.remember(self.pick(known))
            return
        if self.r() < 0.08:
            self.remember(self.pick(("color", "food", "animal", "age")))
            return
        if self.r() < (0.5 if self.name else 0.12):
            user = self.pick(WHATS_MY_NAME)
            if self.name:
                self.add("cannot_know", user, Line(self.pick(NAME_KNOWN)))
            else:
                self.add("cannot_know", user, Line(self.pick(NAME_UNKNOWN), ask="user_name"))
            return
        self.steps("cannot_know", CANNOT)

    def remembered(self) -> list[str]:
        """What the person has said about themselves that they may ask the AI about."""
        return [
            k for k in MEMORY if k in self.said_slots and (k != "pname" or "pet" in self.said_slots)
        ]

    def remember(self, key: str) -> None:
        """The person asks what they said earlier; the AI says it, or that it does not know."""
        asks, known, unknown = MEMORY[key]
        user = self.fill(self.pick(asks))
        if key in self.said_slots:
            ai = self.fill(self.pick(known))
            value = self.slots[key]
            self.memory.append((len(self.turns), value))
            self.add("cannot_know", user, Line("memory"), ai=ai)
        else:
            self.memory.append((len(self.turns), None))
            self.add("cannot_know", user, self.pick(unknown))

    # --- the finished chat ---

    def clean(self) -> bool:
        """Whether every turn keeps the rules: lengths, no held line, no held subject."""
        for role, text in self.turns:
            limit = MAX_AI_CHARS if role == "ai" else MAX_USER_CHARS
            if not text or len(text) > limit or "{" in text or _says_held(self.kb, text):
                return False
            if role == "user" and is_held_small_talk(text):
                return False
        return True

    def record(self) -> ChatRecord:
        identity = (self.ai_name, self.ai_company) if self.ai_name else None
        return ChatRecord(
            self.turns,
            self.kinds,
            self.name,
            self.name_turn,
            self.facts,
            self.memory,
            identity,
            dict(self.slots),
        )


def _chat(kb: KB, rng: np.random.Generator) -> ChatRecord:
    """One chat with its record; a draft that breaks a rule is written again."""
    n = 1 + int(rng.choice(len(EXCHANGE_SHARES), p=EXCHANGE_SHARES))
    # the line is drawn once, so a draft that is written again keeps the same AI
    identity = random_identity(rng) if rng.random() < LINE_SHARE else None
    for _ in range(_TRIES):
        talk = _Talk(kb, rng, n, identity)
        talk.run()
        if talk.clean():
            return talk.record()
    raise ValueError(f"no chat of {n} exchanges keeps the rules in {_TRIES} drafts")


def assistant_chat(kb: KB, rng: np.random.Generator) -> Turns:
    """One short chat between a person and a small, friendly AI; in two in five of them the first
    turn is the AI's identity line, ``("sys", "You are Zula, an AI made by Pengu.")``."""
    return _chat(kb, rng).with_line()


def assistant_chat_docs(kb: KB, rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` chats between a person and a small, friendly AI."""
    return [TextDoc("chat", turns=assistant_chat(kb, rng), topic="other") for _ in range(n)]
