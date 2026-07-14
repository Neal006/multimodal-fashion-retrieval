"""Query -> {pairs:[(color,garment)], garments, environment, style, clauses}."""
import re

MULTI_COLORS = ["light blue", "dark blue", "navy blue", "bright yellow", "dark green",
                "light green", "off white", "dark red", "light pink"]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
          "black", "white", "grey", "gray", "beige", "navy", "maroon", "olive",
          "teal", "cream", "tan", "khaki", "gold", "silver", "burgundy", "mustard"]

COLOR_FAMILY = {"navy": "blue", "navy blue": "blue", "light blue": "blue", "dark blue": "blue",
                "teal": "blue", "maroon": "red", "burgundy": "red", "dark red": "red",
                "bright yellow": "yellow", "gold": "yellow", "mustard": "yellow",
                "grey": "gray", "silver": "gray", "cream": "white", "off white": "white",
                "beige": "brown", "tan": "brown", "khaki": "brown", "olive": "green",
                "dark green": "green", "light green": "green", "light pink": "pink"}

GARMENTS = {
    "shirt": "shirt", "button-down": "shirt", "button down": "shirt", "dress shirt": "shirt",
    "blouse": "shirt", "t-shirt": "tshirt", "tshirt": "tshirt", "tee": "tshirt",
    "hoodie": "hoodie", "sweatshirt": "hoodie", "sweater": "sweater", "cardigan": "sweater",
    "pullover": "sweater", "blazer": "blazer", "suit jacket": "blazer", "suit": "suit",
    "raincoat": "raincoat", "rain coat": "raincoat", "rain jacket": "raincoat",
    "jacket": "jacket", "coat": "coat", "overcoat": "coat", "trench coat": "coat",
    "tie": "tie", "necktie": "tie", "pants": "pants", "trousers": "pants",
    "slacks": "pants", "chinos": "pants", "jeans": "jeans", "denim": "jeans",
    "shorts": "shorts", "dress": "dress", "gown": "dress", "skirt": "skirt",
    "scarf": "scarf", "hat": "hat", "cap": "hat", "beanie": "hat", "vest": "vest",
    "shoes": "shoes", "sneakers": "shoes", "boots": "shoes", "heels": "shoes",
}

ENVS = {"office": ["office", "workplace", "meeting", "conference", "desk"],
        "street": ["street", "city", "urban", "sidewalk", "downtown", "crosswalk"],
        "park": ["park", "bench", "garden", "trail", "lawn"],
        "home": ["home", "house", "couch", "sofa", "living room", "bedroom", "kitchen"]}
STYLES = {"formal": ["formal", "professional", "business", "dressy", "elegant"],
          "casual": ["casual", "weekend", "relaxed", "everyday", "streetwear"],
          "sporty": ["sporty", "athletic", "gym", "workout"]}

ENV_CLAUSE = {"office": "inside a modern office", "street": "on a city street",
              "park": "in a park outdoors", "home": "at home indoors"}

def family(c): return COLOR_FAMILY.get(c, c)

def _first_hit(text, table):
    for canon, words in table.items():
        if any(w in text for w in words):
            return canon
    return None

def parse(query: str) -> dict:
    q = re.sub(r"[^\w\s-]", " ", query.lower())
    for mc in MULTI_COLORS:
        q = q.replace(mc, mc.replace(" ", "_"))
    for gsurf in sorted(GARMENTS, key=len, reverse=True):
        if " " in gsurf:
            q = q.replace(gsurf, gsurf.replace(" ", "_"))
    toks = q.split()

    pairs, bare, used = [], [], set()
    for i, t in enumerate(toks):
        c = t.replace("_", " ")
        if c in COLORS or c in MULTI_COLORS:
            for j in range(i + 1, min(i + 4, len(toks))):   # color binds within 3 tokens
                g = GARMENTS.get(toks[j].replace("_", " "))
                if g:
                    pairs.append((c, g)); used.add(j); break
    for i, t in enumerate(toks):
        g = GARMENTS.get(t.replace("_", " "))
        if g and i not in used and g not in [p[1] for p in pairs]:
            bare.append(g)

    env = _first_hit(q.replace("_", " "), ENVS)
    style = _first_hit(q.replace("_", " "), STYLES)

    clauses = [f"a photo of a person wearing a {c} {g}" for c, g in pairs]
    clauses += [f"a person wearing a {g}" for g in bare]
    if env:   clauses.append(f"a photo taken {ENV_CLAUSE[env]}")
    if style: clauses.append(f"a person in {style} clothes")
    return {"pairs": pairs, "garments": bare, "environment": env,
            "style": style, "clauses": clauses[:5]}


def _demo():
    r = parse("a red tie and a white shirt in a formal setting")
    assert ("red", "tie") in r["pairs"] and ("white", "shirt") in r["pairs"], r["pairs"]
    assert r["style"] == "formal", r
    r = parse("A person in a bright yellow raincoat")
    assert ("bright yellow", "raincoat") in r["pairs"], r["pairs"]
    r = parse("Someone wearing a blue shirt sitting on a park bench")
    assert ("blue", "shirt") in r["pairs"] and r["environment"] == "park", r
    r = parse("Casual weekend outfit for a city walk")
    assert r["style"] == "casual" and r["environment"] == "street", r
    r = parse("Professional business attire inside a modern office")
    assert r["style"] == "formal" and r["environment"] == "office", r
    print("parser OK")


if __name__ == "__main__":
    _demo()
