import random

prefixes = [
    "one", "red", "blue", "green", "golden", "ancient", "brave", "the", "this", "that",
    "mighty", "shadow", "swift", "silent", "bright", "dark", "lone", "wild", "wise", "bold",
    "deep", "last", "lost", "pale", "sage", "true", "false", "dead", "calm", "dual", "pure",
    "raw", "vast", "ever", "new", "old", "free", "high", "low", "near", "far", "open",
    "strong", "weak", "young", "late", "early", "great", "small", "wide", "dread", "foul",
    "grim", "harsh", "keen", "meek", "mild", "rude", "seek-the", "chase", "dance", "dream",
    "fall", "flow", "fly", "gaze", "grow", "know", "make", "move", "ride",
    "rise", "see", "take", "walk", "watch", "become-the", "speak", "run", "chasing",
    "whisper", "conquer", "wander", "embrace", "forge", "guide", "ignite",
    "look-to-the", "reach-for-the", "call-of-the", "path-of-the", "gift-of-the",
    "shadow-of", "eye-of", "song-of", "voice-of", "spirit-of", "wind-of", "embrace-of",
    "guardian-of", "light-of", "knight-of", "crown-of", "whisper-of", "edge-of", "way-of",
    "sword-of", "heart-of", "star-of", "fire-of", "frost-of", "storm-of", "truth-of-the",
    "throne-of", "flame-of", "dawn-of", "journey-of", "soul-of", "scepter-of", "rise-of",
    "march-of", "grace-of", "destiny-of", "fate-of", "tale-of", "power-of-the",
    "echo-of", "fury-of", "wrath-of", "shadow-of-the-dawn", "flame-of-the-night",
    "bringer-of", "harbinger-of", "shield-of", "vessel-of", "blade-of", "protector-of",
    "ward-of", "soul-of-the", "keeper-of", "echo-of-the", "bound-by", "born-of", "fated-to",
    "guided-by", "cursed-by", "blessed-by", "whisper-of-the", "dream-of", "cry-of-the",
    "song-of-the", "spirit-of-the", "child-of-the", "servant-of-the", "master-of-the",
    "rise-of-the", "fire-in-the", "power-within-the", "sign-of", "prophecy-of-the",
    "seeker-of-the", "wanderer-of-the", "tamer-of", "ruler-of-the", "victor-of-the",
    "fire-in-the-heart", "wanderer-between", "night-of-the", "in-the-shadow-of", "lost-to-the",
    "spark-of", "wisdom-of", "sight-of-the", "phantoms-of",
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", 
    "between-the", "along-the", "above-the", "beneath-the", "beyond-the", "by-the", "of-the",
    "bound-to", "three-red", "five-blue", "eternal-sun", "hidden-light", "dark-moons-of",
    "burning", "tears-of", "dreams-of", "whispers-of", "steps-of-the", "among",
    "tales-of-the", "echoes-of-the", "visions-of", "secrets-of-the", "tears-of-the",
    "cries-of-the", "legends-of", "mysteries-of", "stories-of", "memories-of",
    "in-the-light-of", "voices-of", "winds-of", "scents-of", "fires-of",
    "faces-of-the", "graces-of", "destinies-of", "fates-of", "gifts-of", "wonders-of",
    "flames-of-the", "paths-of-the", "spirits-of-the", "blades-of", "storms-of", "shadows-of",
    "march-of-the", "signs-of", "legacies-of", "crystals-of", "wisdoms-of",
    "call-of-the-ancient", "seeds-of", "stones-of", "dreams-of-the", "fires-of-the",
    "secrets-of", "ghosts-of", "ruins-of", "murmurs-of", "waves-of",
    "children-of", "lights-of-the", "songs-of-the", "powers-of-the", "prophecies-of",
    "legends-from", "keepers-of", "sentinels-of", "signs-of-the", "symbols-of"
]

nouns = [
    "mountain", "river", "ocean", "forest", "desert", "glacier", "volcano", "canyon", "valley",
    "meadow", "prairie", "tundra", "island", "reef", "cave", "spring", "stream", "lake", "sea",
    "cliff", "peak", "summit", "horizon", "shore", "beach", "dune", "oasis", "marsh", "swamp",
    "star", "sun", "moon", "planet", "comet", "asteroid", "meteor", "galaxy", "nebula",
    "constellation", "cosmos", "universe", "void", "aurora", "light", "shadow", "darkness",
    "storm", "thunder", "lightning", "rain", "snow", "hail", "wind", "breeze", "tornado",
    "hurricane", "cyclone", "blizzard", "frost", "ice", "flame", "fire", "spark", "ember",
    "tree", "flower", "rose", "lily", "oak", "pine", "willow", "vine", "leaf",
    "branch", "root", "seed", "bloom", "blossom", "garden", "grove", "hedge",
    "wolf", "eagle", "hawk", "owl", "raven", "crow", "swan", "lion", "tiger", "bear",
    "whale", "dolphin", "shark", "serpent", "viper", "cobra", "fox", "deer", "horse",
    "dragon", "phoenix", "griffin", "unicorn", "pegasus", "hydra", "sphinx", "kraken",
    "wyrm", "drake", "basilisk", "chimera", "manticore", "leviathan", "behemoth",
    "knight", "warrior", "mage", "wizard", "witch", "sorcerer", "king", "queen", "prince",
    "princess", "lord", "lady", "sage", "monk", "priest", "hunter", "seeker", "wanderer",
    "nomad", "guardian", "sentinel", "watcher", "keeper", "master",
    "sword", "blade", "dagger", "spear", "axe", "bow", "arrow", "shield",
    "hammer", "mace", "staff", "wand", "orb", "scepter", "flail", "scythe",
    "crystal", "gem", "jewel", "ruby", "emerald", "sapphire", "diamond", "pearl",
    "crown", "ring", "amulet", "pendant", "bracelet", "relic", "artifact",
    "book", "scroll", "tome", "grimoire", "map", "compass", "key", "lock", "chain",
    "mirror", "lens", "prism", "vessel", "chalice", "grail", "bell",
    "castle", "tower", "temple", "shrine", "sanctuary", "haven", "refuge", "fortress",
    "palace", "cathedral", "citadel", "keep", "gate", "bridge", "arch", "door", "portal",
    "dream", "nightmare", "vision", "illusion", "memory", "thought", "mind", "soul",
    "spirit", "heart", "destiny", "fate", "fortune", "chance", "luck", "hope", "wish",
    "prayer", "blessing", "curse", "prophecy", "omen", "sign", "truth", "lie",
    "secret", "mystery", "riddle", "puzzle", "enigma", "paradox", "wisdom", "knowledge",
    "moment", "instant", "eternity", "infinity", "beginning", "ending", "dawn", "dusk",
    "twilight", "midnight", "abyss", "nexus", "crossroads", "path", "way", "road",
    "journey", "quest", "adventure", "voyage"
]

def num_to_name(num: int) -> str:
    return f"{prefixes[num % 256]}-{nouns[num // 256 % 256]}"

def name_to_num(name: str) -> int:
    adj, noun = name.rsplit("-", 1)
    return prefixes.index(adj) + nouns.index(noun) * 256

def gen_name() -> str:
    return num_to_name(random.randint(0, 65535))

if __name__ == "__main__":
    for i in range(256):
        n = random.randint(0, 65535)
        print(f"{n:3} -> {num_to_name(n)}")