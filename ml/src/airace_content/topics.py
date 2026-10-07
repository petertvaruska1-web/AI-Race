"""Topic tags: which of the ``TOPICS`` a document is about, by counting keywords.

A document's topic is the topic with the most keyword hits (the earlier topic wins a tie); a
document with no hits is "other". Keywords are lowercase words; a word also matches through its
possessive, plural and verb endings ("cat's", "foxes", "berries", "smiled", "baking"). Every
keyword belongs to exactly one topic.
"""

from airace_content.textproc import words
from airace_ml.data.corpus import TOPICS

_OTHER = TOPICS.index("other")


def _keywords(text: str) -> frozenset[str]:
    return frozenset(text.split())


TOPIC_KEYWORDS: dict[str, frozenset[str]] = {
    "animals": _keywords(
        """
        animal animals pet zoo cat dog puppy kitten bird fish cow horse pony pig sheep lamb goat
        duck chicken hen rooster rabbit bunny mouse rat lion tiger bear cub elephant monkey ape
        gorilla frog toad snake lizard turtle butterfly bee ant spider insect bug beetle fox wolf
        deer squirrel owl eagle hawk parrot penguin whale dolphin shark seal crab octopus zebra
        giraffe hippo camel kangaroo panda koala donkey crocodile alligator dinosaur hedgehog bat
        worm snail paw tail fur feather wing beak hoof mammal reptile amphibian herd flock
        """
    ),
    "food": _keywords(
        """
        food eat ate eating hungry thirsty delicious tasty yummy taste cook cooking bake baking
        kitchen recipe oven meal breakfast lunch dinner supper snack dessert apple banana orange
        grape lemon peach pear plum cherry strawberry blueberry berry melon watermelon mango
        pineapple fruit vegetable carrot potato tomato onion corn bean pea lettuce cucumber
        broccoli mushroom bread toast sandwich pizza burger pasta noodle rice soup salad cheese
        butter milk egg yogurt cream cake cookie pie candy chocolate sugar honey jam cereal
        pancake waffle juice tea coffee salt pepper flour sauce meat beef pork sausage bacon nut
        popcorn lollipop cupcake muffin donut doughnut feast spice grocery
        """
    ),
    "nature": _keywords(
        """
        nature tree flower rose daisy tulip grass leaf leaves branch root seed plant bush forest
        wood woods jungle garden field meadow river lake pond stream sea ocean beach shore wave
        sand rock stone mountain hill valley cave canyon waterfall volcano island desert cliff
        sun sunny sunshine sunset sunrise sky cloud rain rainy snow snowy ice wind windy storm
        thunder lightning rainbow fog mist weather season spring summer autumn winter mud puddle
        dew frost breeze environment ecosystem color colour
        """
    ),
    "family": _keywords(
        """
        family mother mom mommy mama father dad daddy papa parent parents brother sister sibling
        grandmother grandma grandfather grandpa granny nana aunt uncle cousin son daughter child
        children kid kids baby babies toddler twin husband wife married wedding birthday friend
        friends friendship neighbor neighbour buddy playmate together hug kiss cuddle bedtime
        lullaby gift present party relative relatives guest
        """
    ),
    "school": _keywords(
        """
        school teacher class classroom student pupil lesson homework book read reading write
        writing pencil pen paper desk backpack notebook test exam quiz grade learn learning study
        spelling alphabet letter math maths count counting number subtract subtraction addition
        multiply divide history geography library recess principal college university lecture
        chalk blackboard whiteboard ruler eraser crayon draw drawing paint painting education
        textbook essay word sentence grammar vocabulary question answer calendar monday tuesday
        wednesday thursday friday saturday sunday january february april june july august
        september october november december shape triangle square circle rectangle pentagon
        hexagon octagon oval opposite antonym
        """
    ),
    "science": _keywords(
        """
        science scientist experiment atom molecule element chemical chemistry physics biology
        energy force gravity magnet magnetic electricity electric electron proton neutron oxygen
        hydrogen helium nitrogen carbon iron gold silver copper zinc sodium calcium metal gas
        liquid solid mineral temperature heat boil melt freeze evaporate planet star moon earth
        space galaxy universe orbit solar comet asteroid meteor telescope microscope mercury
        venus mars jupiter saturn uranus neptune pluto astronaut rocket cell dna gene bacteria
        virus fossil evolution species theory research laboratory lab discover discovery measure
        weight mass volume speed light sound periodic symbol
        """
    ),
    "sports": _keywords(
        """
        sport game team player coach goal score scored win won winner lose race racing runner
        swim swimming swimmer soccer football basketball baseball tennis golf hockey volleyball
        rugby cricket badminton bowling skate skating ski skiing surf surfing gym exercise
        workout fitness match tournament league champion championship medal trophy referee umpire
        stadium arena kick throw pitch dribble racket racquet net olympic olympics athlete
        marathon sprint wrestling boxing karate judo gymnastics cheer fan practice training ball
        """
    ),
    "technology": _keywords(
        """
        technology computer laptop tablet phone smartphone internet website web online email app
        software hardware program programming code coding data database network server
        screen keyboard monitor printer camera video robot robotic machine engine motor device
        digital electronic battery charger wire cable circuit chip processor memory algorithm
        artificial intelligence drone satellite television tv radio headphones speaker password
        invent invention inventor engineer engineering gadget car truck bus train plane airplane
        aeroplane jet helicopter boat ship bicycle bike motorcycle vehicle subway tram taxi
        tractor van ferry canoe kayak sailboat yacht tugboat submarine locomotive balloon glider
        blimp scooter
        """
    ),
    "places": _keywords(
        """
        place city town village country capital continent europe asia africa america north south
        east west map travel trip journey street road avenue highway bridge tunnel house home
        building castle palace tower church temple mosque cathedral museum hospital airport
        station market store shop mall restaurant cafe hotel office factory bank park farm barn
        neighborhood neighbourhood downtown border nation kingdom empire playground france paris
        germany berlin italy rome spain madrid england london britain ireland dublin scotland
        russia moscow china beijing japan tokyo india delhi korea seoul egypt cairo brazil canada
        ottawa mexico australia canberra argentina chile peru kenya nigeria morocco rabat greece
        athens sweden norway finland denmark poland portugal lisbon vietnam thailand indonesia
        iran iraq israel pakistan ukraine kyiv
        """
    ),
    "feelings": _keywords(
        """
        feel felt feeling emotion happy happiness sad sadness unhappy angry anger mad afraid
        scared fear frightened brave courage proud pride lonely excited exciting excitement
        nervous worried worry anxious surprised surprise upset cry cried crying tear laugh
        laughed laughing giggle smile smiled smiling joy joyful glad cheerful love loved hate
        kindness gentle shy jealous grumpy calm tired sleepy bored curious sorry hope hopeful
        wish wished disappointed embarrassed grateful thankful cozy comfort comfortable relieved
        hurt mood
        """
    ),
}


def _build_lookup() -> dict[str, int]:
    lookup: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lookup:
                raise ValueError(
                    f"keyword {keyword!r} is in both {TOPICS[lookup[keyword]]!r} and {topic!r}"
                )
            lookup[keyword] = TOPICS.index(topic)
    return lookup


_TOPIC_OF_KEYWORD = _build_lookup()
_MIN_STEM = 3  # a stem left by stripping an ending must be at least this long
# After "ed" or "ing" the stem must be longer, or "being" would be "bee" and "cared" would be "car".
_MIN_VERB_STEM = 4
_ES_AFTER = ("s", "x", "z", "ch", "sh", "o")  # "foxes", "lunches", "potatoes", but not "cares"
_NOT_INFLECTED = frozenset({"bearing"})  # looks like the verb form of the keyword "bear"


def _forms(word: str) -> list[str]:
    """The word itself, then the stems left by its plausible endings, most specific first.

    A possessive ("cat's") is the word without "'s". Plural endings are "ies", "es" (after a
    sibilant) and "s"; verb endings are "ed" and "ing", the stem with and without its final "e".
    """
    word = word.removesuffix("'s")
    forms = [word]
    if word in _NOT_INFLECTED:
        return forms
    if word.endswith("ies"):
        forms.append(word[:-3] + "y")
    if word.endswith("es") and word[:-2].endswith(_ES_AFTER):
        forms.append(word[:-2])
    if word.endswith("s"):
        forms.append(word[:-1])
    if word.endswith("ed"):
        forms += [stem for stem in (word[:-2], word[:-1]) if len(stem) >= _MIN_VERB_STEM]
    if word.endswith("ing"):
        forms += [stem for stem in (word[:-3], word[:-3] + "e") if len(stem) >= _MIN_VERB_STEM]
    return forms


def _topic_of_word(word: str) -> int | None:
    for i, form in enumerate(_forms(word)):
        if (i == 0 or len(form) >= _MIN_STEM) and form in _TOPIC_OF_KEYWORD:
            return _TOPIC_OF_KEYWORD[form]
    return None


def tag_topic(text: str) -> int:
    """Index into ``TOPICS`` of the topic with the most keyword hits in ``text``.

    The earlier topic wins a tie; text without any keyword is "other".
    """
    hits = [0] * len(TOPICS)
    for word in words(text):
        topic = _topic_of_word(word)
        if topic is not None:
            hits[topic] += 1
    best = max(hits)
    return hits.index(best) if best > 0 else _OTHER
