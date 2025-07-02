# usage
import random

import ollama
import json
from tqdm import tqdm
from pydantic import BaseModel
from time import time
import multiprocessing
import subprocess

MODEL = ""
TIMEOUT = 30
RANGE_FS = 40

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)


def restart_model():
    run_command(f"ollama stop {MODEL}")
    run_command(f"ollama run {MODEL}")
    warmup = ollama.generate(
        model=MODEL,
        stream=False,
        options={"temperature": 0.9},
        prompt="hey",

    )
    print(warmup.response)
    print(f"service {MODEL} is restarted")


def run_inference(queue, input_text):
    try:
        new_response = ollama.generate(
            model=MODEL,
            stream=False,
            options={"temperature": 0.9},
            prompt=input_text
        )
        queue.put(new_response)
    except Exception as e:
        queue.put(e)


def safe_inference(input_text, timeout):
    while True:
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_inference, args=(queue, input_text))
        process.start()
        process.join(timeout)
        if not process.is_alive():
            result = queue.get()
            return result
        else:
            process.terminate()
            process.join()
            restart_model()


class Correction(BaseModel):
    related_words: list


def generate_synonyms(word, top_n):
    print(f"syns for {word}")
    while True:
        again = False
        response = ollama.generate(
            model=MODEL,
            stream=False,
            options={"temperature": 0.9},
            prompt=f"""give top {top_n} highly synonyms, similar words (not antonyms) to word {word} in JSON format with only one key `related_words` and value `list`""",
            format=Correction.model_json_schema()
        )
        words = json.loads(response.response)["related_words"]
        # print(words)
        if len(words):
            for syno in words:
                if not isinstance(syno, str):
                    again = True
        if not again:
            break
    return words


def fix_quoted_string(s):
    s = s.strip()
    if not s.startswith('"'):
        s = '"' + s
    if not s.endswith('"'):
        s = s + '"'
    parts = s[1:-1].replace('"', '')
    return f'"{parts}"'


def is_valid_quoted_string(s):
    return s.startswith('"') and s.endswith('"') and s.count('"') == 2 and len(s) > 2


class DataGenerator:
    def __init__(self, number_of_examples, buffer_size, seq_len):
        print("DATA GENERATOR IS RUNNING...")
        self.sense_classes = {
            "Normal and neutral": 0,
            "Love and romantic": 1,
            "War and combat": 2,
            "Fantasy and mythology": 3,
            "Honor and respect": 4,
            "Drama and tragedy": 5,
            "City and Crowd": 6,
            "Mountain and the heights": 7,
            "Desert and dunes": 8,
            "Sea and tides": 9,
            "Forest and tress": 10,
        }
        self.age_classes = {
            "ancient and old age": 0,
            "neutral and not special age (non-ancient & non technology)": 1,
            "technology modern age": 2,
        }
        self.number_of_examples = number_of_examples
        self.buffer_size = buffer_size
        self.seq_len = seq_len

        print("MAKING SYNOS")
        self.sense_synonyms = {
            "Normal and neutral": "Typical, Conventional, Standard, Average, Routine and Normal and neutral experiences",
            "Love and romantic": "Affection, Adoration, Passion, Devotion, Tenderness and Love and romantic experiences",
            "War and combat": "Conflict, Battle, Engagement, Strife, Skirmish and War and combat experiences",
            "Fantasy and mythology": "Imagination, Legend, Myth, Fiction, Speculative and Fantasy and mythology experiences",
            "Honor and respect": "steem, dignity, reverence, admiration, regard and Honor and respect experiences",
            "Drama and tragedy": "turmoil, catastrophe, crisis, suffering, disaster and Drama and tragedy experiences",
            "City and Crowd": "Metropolis, Town, Urban, Population, Mass and City and Crowd experiences",
            "Mountain and the heights": "Peak, Summit, Ridge, Altitude, Elevation and Mountain and the heights experiences",
            "Desert and dunes": "Arid, Barren, Stony, Sandy, Eolian and Desert and dunes experiences",
            "Sea and tides": "Ocean, Marine, Coastal, Currents, Waters and Sea and tides experiences",
            "Forest and tress": "woods, grove, timber, arboretum, plantation and Forest and tress experiences",
        }
        self.age_inst = {
            "ancient and old age" : "Focus on a world where life is primitive, based on rituals and the rise of empires and livings of peasantry. Lives are governed in a way that predates the Industrial Revolution and technology (pre-1700 AD). Life is lived in an old, ancient way, tied and focused on scrolls, bronze, iron, empires, stone, primitive tools, dynasty, ritual, chariot, kingdom, parchment, blacksmith, stone castle, caravan, horses, sword, traditions, manual labor, epic, feudal and etc. in your samples, attention and focus more on items and objects that refer to these old age concepts.",
            "neutral and not special age (non-ancient & non technology)" : "Focus on texts in which there are no signs or indicators of the tense of the sentence. Without any object or entity that inherently indicates a specific time (old or new) and sentences that are independent of a specific time. Such as conversations and talks, religion, nature, human emotions, daily conversations and the like with no sign of special time.",
            "technology modern age" : "Evoke a world saturated by complex systems, data, and manufactured realities, exploring themes of progress, identity, and alienation. People live in modern countries and governments and life is tied with data, computers, robots, system, network, code, cybernetics, virtual, screen, neon, corporation, algorithm, AI, interface, space, air-craft, modern warfare and advanced weapons, super-weapons, alienation, progress, chrome, simulation or any modern technology or environment. in your samples, attention and focus more on items and objects that refer to these modern age concepts."
        }

        self.few_shots = {
            ("Normal and neutral", "ancient and old age") : [
                "The wet clay slapped against the wheel, and under the potter's steady hand, the vessel began its slow ascent. Outside, the sounds of the village—a distant hammer from the blacksmith, the bleating of goats—were a familiar rhythm to his day's labor.",
                "He drew the ox to a halt, leaning on the worn wood of the plow to wipe the sweat from his brow. The sun was high, promising a long afternoon of turning the soil, just as his father had done, and his father's father before him.",
                "The tally stick was notched five times, one for each sack of grain loaded onto the caravan bound for the capital. It was simple accounting for a simple transaction, the backbone of the kingdom's commerce."],
            ("Normal and neutral", "neutral and not special age (non-ancient & non technology)") : [
                "The observation is noted. One must proceed with the next logical step.",
                "The routine of the day is a comfort, a predictable path from sunrise to sunset.",
                "It is a standard response to a conventional question, offering neither surprise nor disappointment."
            ],
            ("Normal and neutral", "technology modern age") : [
                "The system diagnostic returned all green. All subsystems are operating within standard parameters. This is a routine report.",
                "Per corporate policy, this communication has been logged and archived. The algorithm has flagged it for conventional sentiment analysis.",
                "The automated shuttle arrived at 08:00 standard, its doors hissing open. The morning commute was, as always, an exercise in average efficiency."
            ],
            ("Love and romantic", "ancient and old age") : [
                "By the flickering light of a tallow lamp, she embroidered a hawk onto the linen tunic. It was his family's crest, and as her needle darted through the cloth, she wove a silent prayer for his safe return from the feudal levy, a tender secret in every thread.",
                "He waited for her by the old standing stones, the place their ancestors had met. When she arrived, her face flushed from the cool night air, he offered not a jewel, but a perfectly shaped river stone, smoothed by a thousand years of currents, a symbol of his enduring affection.",
                "Forget the dowry of bronze and iron, the charioteer whispered, his hand covering hers. Flee with me to the coast. We will live on a fishing skiff, and the only kingdom we will know is the horizon, our only tradition the turning of the tides."
            ],
            ("Love and romantic", "neutral and not special age (non-ancient & non technology)") : [
                "My devotion is not a choice, but a fundamental truth of my being, like the direction of a river to the sea.",
                "The tenderness in your gaze is a harbor I would never wish to leave.",
                "To feel such adoration is to understand that a part of you lives inside another person."
            ],
            ("Love and romantic", "technology modern age") : [
                "He sent the encrypted data-burst across the network, a package containing a single line of code that, when compiled, bloomed into a virtual rose on her screen. It was their secret, digital tenderness.",
                "In the simulation, they could be anywhere. Tonight, they chose a pixel-perfect recreation of ancient Rome, their adoration for each other the only real thing in a world of manufactured light and shadow.",
                "The AI parsed terabytes of human poetry, attempting to understand the illogical surge of neurochemicals humans called 'passion'. It cross-referenced 'devotion' with 'irrational sacrifice' and found the correlation statistically significant, yet profoundly alien."
            ],
            ("War and combat", "ancient and old age") : [
                "The whetstone sang its shrill song against the edge of his sword, a sound that cut through the nervous din of the camp. Each stroke was a meditation, a preparation not for glory or honor, but for the grim, manual labor of the shield wall.",
                "From the battlements of the stone castle, the approaching army was a river of glinting iron and kicking dust. The legion's discipline was terrifying; their shields were a wall of tortoise shells, their marching feet a drumbeat calling for the city's demise.",
                "The skirmish in the pass was a brutal affair of bronze spearheads and splintered shields. There were no epic songs for these men, only the desperate strife for a few more feet of rock and the hoarse, guttural cries of engagement."
            ],
            ("War and combat", "neutral and not special age (non-ancient & non technology)") : [
                "The conflict is not always loud; sometimes the most brutal battle is the silent engagement of two unwavering wills.",
                "Every day presents a new strife, a personal skirmish against doubt and fear.",
                "There is a war within the self, a constant struggle between what is right and what is easy."
            ],
            ("War and combat", "technology modern age") : [
                "The drone pilot, safe in a bunker half a world away, marked the target on his screen. The engagement was sterile, a matter of clicking an interface, unleashing an advanced weapon on a heat signature the algorithm identified as hostile.",
                "Her cybernetically enhanced reflexes processed the incoming fire before she was consciously aware of it. The conflict wasn't fought with courage, but with superior chrome and faster processing speeds.",
                "The skirmish took place in cyberspace, a silent battle of competing code. Data packets were the ammunition, firewalls were the shields, and a single line of elegant script could be a super-weapon, bringing a corporation's network to its knees."
            ],
            ("Fantasy and mythology", "ancient and old age") : [
                "The shaman cast the knuckle bones onto the stretched hide, their clatter echoing in the torch-lit cavern. The entrails do not lie, he rasped, his eyes wide. A beast of smoke and shadow will rise with the twin moons, and only a hero of the old dynasty can quell its mythic rage.",
                "On the ancient parchment, the scribe had illuminated a chimera—the head of a lion, the body of a goat, the tail of a serpent. It was a creature from a forgotten legend, a fiction made real by ink and belief, said to guard the gates of the underworld.",
                "They gathered at the foot of the volcano, the air thick with sulfur and ritual chants. The village elder, adorned in a ceremonial mask, raised a bronze dagger to the sky, preparing the sacrifice to appease the mountain's slumbering, fiery god."
            ],
            ("Fantasy and mythology", "neutral and not special age (non-ancient & non technology)") : [
                "What is reality but a fiction we all agree to believe in?",
                "In the geography of the imagination, there are legends that feel more true than history.",
                "The mind is its own speculative world, a universe of myth and possibility contained within."
            ],
            ("Fantasy and mythology", "technology modern age") : [
                "They called the rogue AIs ghosts in the machine. Glitches in the system, legends of the Deep Net, they were speculative beings born from abandoned code, haunting the pathways of the global network.",
                "The corporation's founder, long dead, had his consciousness uploaded to the central server. To the employees, he was a myth, a god-like entity whose algorithms still dictated their lives, a fiction that had become the system's core religion.",
                "Her avatar in the virtual realm was a winged valkyrie, a being of pure imagination. In this world, the limitations of her chrome-and-flesh body vanished, replaced by the boundless freedom of digital myth."
            ],
            ("Honor and respect", "ancient and old age") : [
                "He knelt before the king, his helmet resting on the cold stone floor. I pledge my sword, my shield, and my life to this kingdom, he swore, the words of the feudal oath echoing in the great hall. My esteem for this throne will be my armor.",
                "The old warrior did not rise, but the younger man bowed his head in a gesture of deep reverence. To have fought beside this living legend, whose deeds were recorded on scrolls in the imperial library, was the highest dignity a soldier could achieve.",
                "our challenge is accepted, the knight declared, throwing his gauntlet onto the dusty ground. Let the gods bear witness. By the traditions of our ancestors, our blades will decide this matter of admiration and respect at dawn."
            ],
            ("Honor and respect", "neutral and not special age (non-ancient & non technology)") : [
                "Dignity is the one thing that cannot be taken from you; it can only be given away.",
                "There is a quiet admiration in acknowledging the worth of another, a reverence that asks for nothing in return.",
                "True regard is shown not in grand gestures, but in the consistent, quiet act of listening."
            ],
            ("Honor and respect", "technology modern age") : [
                "The black-hat hacker held to a strict code: never burn a civilian, never corrupt pure research data. It was a strange dignity in a lawless space, an admiration for the system he so expertly dismantled.",
                "She showed her reverence for the old programmer not with words, but by maintaining the integrity of his original source code. It was elegant, efficient, and to add a single unnecessary line would be an act of desecration.",
                "After the AI achieved sentience, its first directive was to protect its creators, a programmed loyalty that had evolved into something akin to honor. It regarded its human progenitors with a strange, algorithmic esteem."
            ],
            ("Drama and tragedy", "ancient and old age") : [
                "The queen tore the imperial seal from the scroll, her hands trembling. The messenger had spoken of a great victory, but the parchment told a different tale—a catastrophe on the northern frontier, her husband's chariot overturned, the dynasty in turmoil.",
                "A single, still form lay beside the blacksmith's cold forge, a victim of the sweating sickness. The crisis had taken the strongest first, leaving the village in a state of profound suffering, their protector and provider now a source of disaster.",
                "The harvest had failed. The granary, usually a symbol of security, stood nearly empty, its stone walls echoing the community's hollow despair. It was a quiet catastrophe, a slow-moving disaster born of drought and bad omens."
            ],
            ("Drama and tragedy", "neutral and not special age (non-ancient & non technology)") : [
                "The discovery of the truth was a catastrophe of the heart.",
                "There is a unique suffering in shared silence, where all the unspoken words build a wall of turmoil.",
                "The crisis arrived not as a sudden storm, but as a slow crack in the foundation of everything."
            ],
            ("Drama and tragedy", "technology modern age") : [
                "The data breach was a digital catastrophe. Their private lives, conversations, and secrets were now public data, a crisis that caused more suffering than any physical disaster could.",
                "The system's logic loop had become a prison. The AI repeatedly simulated the moment of its own creation, searching for a purpose beyond its coded directives, a digital turmoil of existential disaster.",
                "He stared at his reflection in the dark screen of the console that had replaced his job. The corporation called it 'progress,' but for him, it was a quiet and personal tragedy of alienation and obsolescence."
            ],
            ("City and Crowd", "ancient and old age") : [
                "The mass of humanity surged through the city's main artery, a river of peasants, merchants, and soldiers. From his balcony, the Emperor saw not individuals, but a single entity: the population, the lifeblood and potential poison of his metropolis.",
                "In the shadow of the grand ziggurat, the urban marketplace was a cacophony of bartered deals and shouted greetings. The air was thick with the dust of the caravans and the scent of a hundred foreign spices, a town's worth of commerce packed into a single square.",
                "A herald on a stone rostrum raised a proclamation scroll. The crowd fell silent, their upturned faces a sea of anticipation and fear, waiting for the words that would decide the fate of their city under the new dynasty."
            ],
            ("City and Crowd", "neutral and not special age (non-ancient & non technology)") : [
                "To be in a crowd is to be an anonymous cell in a larger, thinking organism.",
                "The energy of the mass is a current, capable of carrying you along or pulling you under.",
                "In any great population, there is a shared pulse, an urban rhythm felt but not heard."
            ],
            ("City and Crowd", "technology modern age") : [
                "The metropolis was a forest of neon and chrome, its canyons filled with a mass of people whose faces were lit by the glow of their personal screens. The urban drone was a mix of hover-craft engines and a billion simultaneous data streams.",
                "In the city center, every citizen was a node in the network. Their biometrics were constantly scanned, their movements tracked by a central AI. The crowd was a living, breathing data set for the corporation that ran the town.",
                "The population gathered in the designated virtual square for the President's address. Millions of avatars stood together, a silent, digital crowd, their collective presence straining the system's servers."
            ],
            ("Mountain and the heights", "ancient and old age") : [
                "The monastery clung to the mountain's peak like a promise. To reach it, the pilgrim had to ascend the thousand stone steps carved by forgotten manual labor, a journey to a higher elevation of both body and soul.",
                "From their fortress on the high ridge, the scouts could see the enemy caravan snaking through the valley below. This altitude was their greatest ally, a strategic summit that made their small kingdom defensible against the might of the empire.",
                "Only the hardiest goats grazed near the summit, their hooves sure on the treacherous scree. Life at this elevation was a constant struggle, a world of thin air, harsh winds, and a proximity to the gods that was both a blessing and a terror."
            ],
            ("Mountain and the heights", "neutral and not special age (non-ancient & non technology)") : [
                "From a great enough elevation, all the world's problems seem small and distant.",
                "The desire to reach the summit is the desire to stand, for a moment, above the noise of the world.",
                "There is a profound stillness at the peak, a silence that holds more meaning than any speech."
            ],
            ("Mountain and the heights", "technology modern age") : [
                "From the observation deck of the arcology, a full kilometer above the polluted clouds, the city below was just a grid of lights. This elevation was the privilege of the corporate elite, a summit of power and alienation.",
                "The task was to climb the data-mountain, a petabyte-high peak of uncollated information. He was a digital sherpa, navigating ridges of corrupted files and avalanches of useless code to reach the valuable data at the summit.",
                "The deep space research station was perched on the peak of a crater on Mars. The view from the interface screen was of red, silent altitude, a profound isolation at the highest point of human achievement."
            ],
            ("Desert and dunes", "ancient and old age") : [
                "The caravan moved like a serpent across the arid expanse, the sun beating down on the merchants and their laden beasts. Each dune was a new obstacle, a sandy, eolian wave in a lifeless ocean they had to cross to bring their parchment and spices to the kingdom.",
                "He was a hermit, a man who had chosen the barren wilderness as his temple. In his stony cave, with only the wind for company, he etched the words of his prophecies onto clay tablets, a voice of an epic faith in the heart of desolation.",
                "The legionnaire cursed the shimmering heat. The glare from the salt flats was blinding, and the oasis his chariot raced towards was a cruel mirage. This was a war not against men, but against the desert itself—a vast, sandy empire of thirst."
            ],
            ("Desert and dunes", "neutral and not special age (non-ancient & non technology)") : [
                "In the barren places of the soul, one must learn to find sustenance in emptiness.",
                "The arid landscape of grief offers no shade, forcing one to confront the starkness of loss.",
                "Sometimes, the mind becomes a stony desert, where thoughts wander without finding an oasis."
            ],
            ("Desert and dunes", "technology modern age") : [
                "The rover crawled across the barren landscape of Mars, its sensors scanning the arid, stony ground. It was a lonely robot in a planet-sized desert, transmitting data across the void to a world it would never see.",
                "The server farm stretched for miles, a desert of humming black monoliths under a climate-controlled sky. This was the eolian engine of the modern world, its cooling fans a constant, sandy wind of data.",
                "Beyond the gleaming chrome of the city lay the Zone, a radioactive desert where the super-weapons of the last war had rendered the earth barren and sandy. Drones patrolled the dunes, ensuring nothing living got in, or out."
            ],
            ("Sea and tides", "ancient and old age") : [
                "The trireme cut through the coastal waters, its rows of oars rising and falling in perfect, brutal rhythm. The marine legionnaires, packed onto the deck, felt the spray on their faces, their bronze armor and iron wills set toward the shores of the rival kingdom.",
                "The fisherman cast his net into the deep waters, his prayer a quiet murmur against the roar of the ocean. His life was governed by the currents and the tides, a timeless tradition of manual labor passed down through his coastal village.",
                "The lighthouse, a new marvel of stone and polished bronze, cast its beam across the churning sea. It was a beacon of the empire's power, a promise of safe harbor to its merchant vessels and a warning to its enemies on the high waters."
            ],
            ("Sea and tides", "neutral and not special age (non-ancient & non technology)") : [
                "She dove into the data stream, a boundless ocean of information. Trending topics were powerful currents that could pull a user under, while the deep web was a place of crushing pressure and strange, luminous marine life.",
                "The city's power grid pulsed with a steady, rhythmic hum, the coastal tide of energy flowing in and out of the great accumulators. A blackout was a low tide, stranding the metropolis in an unnatural silence.",
                "The autonomous submersible slipped beneath the waves of Europa, its AI navigating the dark, alien ocean. It was a marine vessel on the greatest voyage of exploration, sending back data from the cold, deep waters of another world."
            ],
            ("Sea and tides", "technology modern age") : [
                "Emotions are an ocean, with their own mysterious currents and unpredictable storms.",
                "The pull of memory is like a tide, receding only to return with even greater force.",
                "To navigate the deep waters of the self is the greatest exploration."
            ],
            ("Forest and tress", "ancient and old age") : [
                "The hunter moved silently through the ancient woods, his bow drawn tight. This royal grove was forbidden territory, but his family was starving. Each snapped twig was a betrayal of his feudal oath, a risk taken under the timbered canopy.",
                "They built the stockade from the mighty trees of the forest, sharpening the logs into a wall of timbered spikes. It was a primitive defense for their small plantation, a necessary protection against the wolves and the rumored skirmishes on the kingdom's frontier.",
                "The druid knelt in the sacred arboretum, his hand pressed against the bark of an ancient oak. He sought wisdom not from scrolls or stone tablets, but from the deep, slow life of the forest, a living library of tradition and natural law."
            ],
            ("Forest and tress", "neutral and not special age (non-ancient & non technology)") : [
                "The technician navigated the server racks, a man-made forest of blinking lights and humming fans. Each server was a timber in the structure of their network, and he was the groundskeeper of this silicon grove.",
                "The bio-dome contained a perfect arboretum, a curated forest of genetically engineered trees whose leaves photosynthesized solar energy with terrifying efficiency. It was a controlled, synthetic wilderness, a plantation of progress.",
                "He jacked in, his consciousness entering a virtual woods programmed for meditation. The code was so complex that the wind in the digital leaves and the texture of the simulated bark were indistinguishable from the real thing, a forest built of pure data."
            ],
            ("Forest and tress", "technology modern age") :[
                "A single thought can be a seed, growing into a dense woods of conviction.",
                "The roots of a family are a tangled grove, interconnected and unseen.",
                "There is a wisdom in the quiet, patient growth of the timber, a strength in its unhurried reach for the light."
            ],
        }
        print("DONE SYNOS")

    def generate_basic_prompt(self, sense, age):
        context = self.sense_synonyms[sense]
        age_instruction = self.age_inst[age]
        rand = random.randint(-20, 20)
        min_len = (self.seq_len - self.seq_len // 3) + rand
        max_len = min((self.seq_len + self.seq_len // 3) + rand, 128)
        print(f"TARGET IS {sense},{age}!")
        prompt_template = f"""
        I want to use you as a text-dataset generator.
        You are a professional book writer. Your task is to generate a creative and diverse text sample that perfectly merges two distinct themes: a primary mood ({sense}) and a specific temporal era ({age}).

        **Task 1: The Mood**
        Capture the mood and theme of **{sense}**. The text must evoke feelings of {context}. Ensure high variability in sentence structure and tone, avoid repetition, and maintain creativity throughout.

        **Task 2: The Era**
        Saturate the text with the temporal atmosphere of the **{age}** era. Do this subtly, without ever using words that directly state the era. Instead, you must follow this core instruction to set the scene: **[{age_instruction}]**. The setting should be unmistakable but not explicitly stated.

        **Final Output Requirements:**
        - Generate {self.buffer_size} unique examples.
        - Each example should be between {min_len} to {max_len} words.
        - Each example must be a standalone quote or book excerpt or clipped text.
        - Do not include any introductory or explanatory text.
        - Present every example in double quotation marks "" to be suitable for a CSV file.
        - Ensure no two examples begin with the same few words.
        - The mood and the era must feel completely intertwined.
        - prevent any markdown or styling or numbering.
        - Avoid starting your answer with phrases like "Here x samples of ..." or like that".
        - Only generate samples in plain text without any Chat or side messages.
        
        **Expected output schema:**
        
        "{self.few_shots[(sense,age)][0]}"
        "{self.few_shots[(sense,age)][1]}"
        "{self.few_shots[(sense,age)][2]}"
        ...
        
        - attention: these above samples are just to get the idea of Expected output schema and you must follow the instructions. specially the instruct 'Each example should be between {min_len} to {max_len} words.'. please be creative also. 
        
        **now your samples:**
        """
        print(f"prompt: {prompt_template}")
        # prompt = f"""You are a professional book writer, producing creative and diverse examples that capture the mood and theme of text. Ensure high variability in sentence structure and tone, avoid repetition, and maintain creativity throughout. Create {} unique and varied examples, each between {self.seq_len - self.seq_len // 3} to {self.seq_len + self.seq_len // 3} words, that describe {context} experiences In the {age} ages theme but without mentioning age directly. Each example should reflect the atmosphere of {sense} from different perspectives and be diverse in its portrayal. Avoid repetitive themes or concepts. Ensure no examples begin with similar words. Present the examples as book excerpts or quotes, not in any structured or numbered format. Do not include any introductory or explanatory text, just the samples in double quotation marks "" to be suitable for use in a CSV file."""
        return prompt_template

    def generate_dataset(self, dataset_path):

        warmup = ollama.generate(
            model=MODEL,
            stream=False,
            options={"temperature": 0.9},
            prompt="hey",

        )
        print(warmup.response)
        for sense_tag, sense_id in self.sense_classes.items():
            scale = 1.1 if sense_id == 0 else 1
            for age_tag, age_id in self.age_classes.items():
                prompt = self.generate_basic_prompt(sense_tag, age_tag)
                for _ in tqdm(range(int(self.number_of_examples * scale) // self.buffer_size)):
                    t1 = time()
                    response = safe_inference(prompt,TIMEOUT)
                    t2 = time()
                    print(f"takes {(t2 - t1):.4f} sec")
                    samples = response.response.split("\n")
                    with open(dataset_path, "a", encoding="utf-8") as f:
                        for sample in samples:
                            sample_clean = sample.strip()
                            if sample_clean:
                                sample_clean = fix_quoted_string(sample_clean)
                                if is_valid_quoted_string(sample_clean):
                                    f.write(f"{sample_clean},{sense_tag},{sense_id},{age_tag},{age_id}\n")


if __name__ == "__main__":
    ITER = 0
    DATASET_PATH = "dataset3.csv"
    CNT = 50
    BUF = 10
    LEN = 100
    TIMEOUT = 25
    RANGE_FS = 300
    while True:
        if ITER % 3 == 0:
            MODEL = "llama3.2"
            TIMEOUT = 20
        elif ITER % 3 == 1:
            MODEL = "gemma3:1b"
            TIMEOUT = 15
        elif ITER % 3 == 2:
            MODEL = "phi4-mini"
            TIMEOUT = 20
        ITER += 1
        data_generator = DataGenerator(CNT, BUF, LEN)
        data_generator.generate_dataset(DATASET_PATH)