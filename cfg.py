import json
import time
import gc
from pathlib import Path
from string import Template
from llama_cpp import Llama
from llama_cpp.llama import LlamaGrammar
from nltk.corpus import wordnet as wn

# === Load 1 synset from file ===
with open("imagenet_synsets.json", "r") as f:
    synsets = json.load(f)

synset_id, data = next(iter(synsets.items()))
synset = wn.synset_from_pos_and_offset(synset_id[-1], int(synset_id[:8]))
name = synset.lemmas()[0].name().replace('_', ' ')
description = data["description"]
print(f"🔍 Synset ID: {synset_id} — {description}")

# === Load prompt templates ===
prompt_templates = {}
for template_file in Path("prompts").glob("*.txt"):
    with open(template_file) as f:
        prompt_templates[template_file.stem] = Template(f.read())

# === Define context-free grammar (CFG) ===
COMMONSENSE_GBNF = r"""
ws ::= ([ \t\n] ws)?

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
     "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\""


digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

one-or-two-digits ::= digit | digit digit

zero-to-four-digits ::= "" | digit | digit digit | digit digit digit | digit digit digit digit

material ::= "wood" | "stone" | "clay" | "sand" | "soil" | "mud" | "bone" | "leather" | "cotton" | "wool" | "silk" | "hemp" | "bamboo" | "straw" |
             "glass" | "plastic" | "rubber" | "paper" | "cardboard" | "fabric" | "ceramic" | "concrete" | "asphalt" | "foam" | "fiberglass" |
             "iron" | "steel" | "copper" | "aluminum" | "gold" | "silver" | "brass" | "bronze" | "lead" | "tin" | "titanium" | "nickel" |
             "nylon" | "polyester" | "acrylic" | "kevlar" | "teflon" | "PVC" | "carbon fiber" | "silicone" |
             "laminate" | "composite" | "alloy" | "bioplastic" | "graphene" | "resin" | "adhesive" |
             "material" | "substance" | "element" | "compound" | "stuff" | "matter" | "All types of material"
material-list ::= "[" material ("," ws material)* "]"
             
location ::= "kitchen" | "bathroom" | "bedroom" | "classroom" | "hallway" | "office" | "laboratory" | "warehouse" | "basement" | "attic" |
             "city" | "town" | "village" | "street" | "building" | "store" | "factory" | "hospital" | "school" | "airport" | "station" |
             "parking lot" | "museum" | "theater" | "mall" | "restaurant" | "library" | "gym" |
             "forest" | "desert" | "mountain" | "river" | "ocean" | "beach" | "valley" | "cave" | "island" | "swamp" |
             "jungle" | "lake" | "waterfall" | "hill" | "meadow" | "volcano" |
             "door" | "wall" | "floor" | "ceiling" | "corner" | "roof" | "balcony" | "window" | "tunnel" | "stairway" |
             "room" | "place" | "environment" | "area" | "site" | "location" | "region" | "zone" | "field" | "All types of location"
location-list ::= "[" location ("," ws location)* "]"

shape ::= "circle" | "square" | "triangle" | "rectangle" | "oval" | "pentagon" | "hexagon" | "octagon" | "diamond" | "star" | "heart" |
          "cube" | "sphere" | "cone" | "cylinder" | "pyramid" | "dome" | "prism" |
          "blob" | "spiral" | "wave" | "coil" | "jagged" | "curved" | "flat" | "round" | "pointed" | "irregular" |
          "line" | "edge" | "arc" | "frame" | "corner" | "cross" | "ring" |
          "pattern" | "shape" | "form" | "Not applicable"
shape-list ::= "[" shape ("," ws shape)* "]"

color ::= "red" | "blue" | "yellow" | "green" | "orange" | "purple" |
          "black" | "white" | "gray" | "brown" | "beige" | "cream" |
          "pink" | "turquoise" | "cyan" | "magenta" | "teal" | "maroon" | "navy" | "lime" | "gold" | "silver" |
          "light blue" | "dark green" | "pale yellow" | "bright red" | "olive" | "indigo" | "lavender" | "coral" | "peach" | "mint" |
          "transparent" | "multicolor" | "colorless" | "Not applicable"
color-list ::= "[" color ("," ws color)* "]"

function ::= "sitting" | "standing" | "sleeping" | "eating" | "drinking" | "walking" | "running" | "playing" | "reading" | "writing" |
             "cutting" | "heating" | "cooling" | "storing" | "carrying" | "protecting" | "cleaning" | "building" | "fixing" | "measuring" |
             "holding" | "containing" | "covering" | "sealing" | "organizing" |
             "watching" | "listening" | "displaying" | "signaling" | "recording" | "lighting" |
             "charging" | "connecting" | "computing" | "processing" | "detecting" | "communicating" |
             "decoration" | "entertainment" | "transportation" | "education" | "comfort" | "safety" | "Not applicable"
function-list ::= "[" function ("," ws function)* "]"
             
size ::= "tiny" | "small" | "medium" | "large" | "huge" | "gigantic" |
         "microscopic" | "massive" | "enormous" | "minuscule" |
         "variable" | "unknown" | "Not applicable"

state-of-matter ::= "solid" | "liquid" | "gas" | "plasma" | "gel" | "foam" | "aerosol" | "vapor" |
                    "unknown" | "Not applicable"

durability ::= "fragile" | "soft" | "flexible" | "rigid" | "durable" | "indestructible" | "Not applicable"

common-context ::= "home" | "kitchen" | "outdoors" | "school" | "hospital" | "office" | "factory" | "laboratory" | "space" | "vehicle" | "urban" | "rural" | "Not applicable"
common-context-list ::= "[" common-context ("," ws common-context)* "]"

temperature-sensitivity ::= "heat-sensitive" | "cold-resistant" | "temperature-neutral" | "flammable" | "non-flammable" | "Not applicable"

range ::= zero-to-four-digits "-" zero-to-four-digits

weight-kg ::= range " kg"
size-cm ::= range " cm"


entity ::= (
  "{\n" ws
    "\"name\": " string "," ws
    "\"material\": " material-list "," ws
    "\"location\": " location-list "," ws
    "\"shape\": " shape-list "," ws
    "\"color\": " color-list "," ws
    "\"used for\": " function-list "," ws
    "\"size\": \"" size " \"," ws
    "\"size (cm)\": \"" size-cm " \"," ws
    "\"weight (kg)\": \"" weight-kg "\"," ws
    "\"state of matter\": " state-of-matter "," ws
    "\"durability\": " durability "," ws
    "\"common-context\": " common-context "," ws
    "\"temperature-sensitivity\": " temperature-sensitivity "\n" ws
  "}"
)


root ::= "[\n  " entity (",\n" ws entity)* ws "]"
"""

grammar = LlamaGrammar.from_string(COMMONSENSE_GBNF, verbose=False)

# === Available models ===
MODELS = {
    "llama-2-13b": "/home/user/repos/local_llm/models/llama-2-13b.Q4_K_M.gguf",
    "openorca-13b": "/home/user/repos/local_llm/models/openorca-13b.Q4_K_M.gguf",
    "wizardlm-13b": "/home/user/repos/local_llm/models/wizardlm-13b-v1.2.Q4_K_M.gguf",
    "openorca-platypus2-13b": "/home/user/repos/local_llm/models/openorca-platypus2-13b.Q4_K_M.gguf"
}

# === Loop through templates and models ===
for template_name, template in prompt_templates.items():
    prompt = template.safe_substitute(name=name, description=description)
    print(f"\n🧠 Template: {template_name}\n---\n{prompt}\n")

    for model_name, model_path in MODELS.items():
        print(f"🚀 Running model: {model_name}...")

        # Load the model
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=16,
            n_gpu_layers=0,
            verbose=False
        )

        try:
            output = llm(
                prompt,
                max_tokens=200,
                grammar=grammar  # 👈 constrained decoding
            )
            text = output["choices"][0]["text"].strip()
            print(f"✅ Output from {model_name}:\n{text}\n{'-'*60}")

        except Exception as e:
            print(f"❌ Error with model {model_name}: {e}")

        # Clean up
        del llm
        gc.collect()
        time.sleep(1)
