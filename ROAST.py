import logging
import math
import string
from typing import List, Tuple, Dict, Any, Optional
import ollama
import json
import ast
import re
# from haystack import component, Document, Answer
# from haystack.components.readers import ExtractiveReader
from abc import ABC, abstractmethod
from thefuzz import fuzz
import matplotlib.pyplot as plt
import torch
import time
import gc


class ROASTDriver(ABC):
    """
    Abstract Base Class defining the standard interface for a ROAST-style instance extractor.

    This class establishes a contract for any implementation that aims to find specific
    instances of an abstract concept within a given text. It ensures that different
    approaches (e.g., extractive vs. generative) can be used interchangeably
    by downstream components that rely on this common interface.
    """

    @staticmethod
    def get_extractor(
        driver_type: str, model_name_or_path: str, score_threshold: float
    ):
        if driver_type == "llm":
            return ExpertInstanceGenerator(model_name_or_path)
        elif driver_type == "nn":
            return ExpertInstanceExtractor(
                model_name_or_path, score_threshold=score_threshold
            )
        else:
            raise ValueError(
                f"Invalid driver type specified: {driver_type}. Choose 'nn' or 'llm'."
            )

    @abstractmethod
    def extract(
        self,
        context: str,
        abstract_concept: str,
        explanation_of_abstract: Optional[str] = None,
    ) -> List[Tuple[str, float, int, int]]:
        """
        The core method for performing instance extraction.

        Implementations of this method should take a context and a concept, and return
        a list of found instances with their associated metadata.

        :param context: The text to search within.
        :param abstract_concept: The abstract concept to find instances of (e.g., "car", "programming language").
        :param explanation_of_abstract: An optional, more detailed explanation of the abstract concept to aid extraction.
        :return: A list of tuples. Each tuple represents a found instance and contains:
                 - (str): The extracted instance string.
                 - (float): A confidence score for the extraction (between 0.0 and 1.0).
                 - (int): The starting character offset of the instance in the original context.
                 - (int): The ending character offset of the instance in the original context.
        """
        pass

#
# @component
# class QuestionGenerator:
#     """
#     Generates a set of varied, natural-language questions to guide the
#     ExtractiveReader, improving its ability to find diverse instances.
#     """
#
#     @component.output_types(questions=List[str])
#     def run(
#         self, abstract_concept: str, explanation: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """
#         Generates questions based on the abstract concept.
#         :param abstract_concept: The concept to find instances of.
#         :param explanation: An optional explanation of the abstract concept.
#         :return: A dictionary containing a list of questions.
#         """
#         if not isinstance(abstract_concept, str) or not abstract_concept:
#             logging.warning("QuestionGenerator received an invalid abstract_concept.")
#             return {"questions": []}
#
#         # Generate a plural form for more natural-sounding questions.
#         if (
#             abstract_concept.endswith("y")
#             and len(abstract_concept) > 1
#             and abstract_concept[-2] not in "aeiou"
#         ):
#             plural_concept = f"{abstract_concept[:-1]}ies"
#         elif abstract_concept.endswith(("s", "x", "z", "ch", "sh")):
#             plural_concept = f"{abstract_concept}es"
#         else:
#             plural_concept = f"{abstract_concept}s"
#
#         # Create a context-setting prefix if an explanation is provided.
#         prefix = ""
#         if explanation:
#             prefix = f"A {abstract_concept} {explanation}. "
#
#         # A set of diverse questions to improve the model's recall.
#         questions = [
#             f"{prefix}. Which instances of {abstract_concept} are mentioned in the text?",
#             f"{prefix}. Which {plural_concept} are described in the passage?",
#             # f"{prefix}. What specific {plural_concept} are listed in the document?",
#             # f"{prefix}. Identify the names of the {plural_concept} in the text.",
#         ]
#         return {"questions": questions}
#
#
# @component
# class AnswerFilter:
#     """
#     Filters and refines a list of Haystack Answers using score normalization
#     and span containment to find the best candidate answers.
#     """
#
#     @staticmethod
#     def _is_overlapping(answer1: Answer, answer2: Answer) -> bool:
#         """
#         Checks if one answer's span is fully contained within the other.
#         This is a static method as its logic does not depend on the state
#         of an AnswerFilter instance.
#         """
#         # Ensure both answers have valid document_offset attributes to compare.
#         if not all(
#             hasattr(ans, "document_offset") and ans.document_offset
#             for ans in [answer1, answer2]
#         ):
#             return False
#         start1, end1 = answer1.document_offset.start, answer1.document_offset.end
#         start2, end2 = answer2.document_offset.start, answer2.document_offset.end
#
#         return (start1 >= start2 and end1 <= end2) or (
#             start2 >= start1 and end2 <= end1
#         )
#
#     @component.output_types(filtered_answers=List[Answer])
#     def run(self, answers: List[Answer]) -> Dict[str, Any]:
#         """
#         Filters answers using score normalization and span containment.
#         :param answers: A list of Answer objects from the reader.
#         :return: A dictionary containing the filtered list of answers.
#         """
#         # Calculate a normalized score to penalize overly long answers.
#         for ans_item in answers:
#             if ans_item.data:
#                 # Add a small constant to length to avoid division by zero or log(1) issues.
#                 ans_item.meta["normalized_score"] = ans_item.score / math.log(
#                     len(ans_item.data) + 1.1
#                 )
#             else:
#                 ans_item.meta["normalized_score"] = 0
#
#         # Sort by the new normalized score to prioritize concise, high-confidence answers.
#         sorted_answers = sorted(
#             answers, key=lambda x: x.meta.get("normalized_score", 0), reverse=True
#         )
#
#         # Filter out overlapping answers, keeping the one with the higher normalized score.
#         final_answers: List[Answer] = []
#         for candidate_answer in sorted_answers:
#             if candidate_answer.data is None:
#                 continue
#             if not any(
#                 self._is_overlapping(candidate_answer, kept_answer)
#                 for kept_answer in final_answers
#             ):
#                 final_answers.append(candidate_answer)
#         return {"filtered_answers": final_answers}
#
#
# class ExpertInstanceExtractor(ROASTDriver):
#     """
#     Orchestrates Haystack components and applies advanced heuristic filtering
#     to perform highly accurate instance extraction.
#     """
#
#     def __init__(
#         self,
#         model_name_or_path: str,
#         device: Optional[str] = None,
#         reader_top_k: int = 20,
#         score_threshold: float = 0.0,
#     ):
#         self.q_gen = QuestionGenerator()
#         self.reader = ExtractiveReader(
#             model=model_name_or_path, device=device, top_k=reader_top_k, no_answer=True
#         )
#         self.filter = AnswerFilter()
#         self.reader.warm_up()
#         self.score_threshold = score_threshold
#
#         # An expanded set of "function words" to better identify descriptive phrases.
#         self.FUNCTION_WORDS = {
#             "a",
#             "an",
#             "the",
#             "in",
#             "on",
#             "of",
#             "for",
#             "to",
#             "with",
#             "by",
#             "at",
#             "is",
#             "are",
#             "was",
#             "were",
#             "be",
#             "been",
#             "being",
#             "have",
#             "has",
#             "had",
#             "do",
#             "does",
#             "did",
#             "will",
#             "would",
#             "should",
#             "can",
#             "could",
#             "may",
#             "might",
#             "must",
#             "one",
#             "two",
#             "three",
#             "four",
#             "five",
#             "six",
#             "seven",
#             "eight",
#             "nine",
#             "ten",
#             "some",
#             "any",
#             "all",
#             "several",
#             "many",
#             "few",
#             "other",
#             "another",
#             "various",
#             "its",
#             "their",
#             "my",
#             "your",
#             "his",
#             "her",
#             "first",
#             "second",
#             "third",
#             "last",
#             "next",
#             "former",
#             "latter",
#             "main",
#             "largest",
#             "smallest",
#             "older",
#             "newer",
#             "red",
#             "green",
#             "blue",
#             "named",
#             "called",
#             "known",
#             "described",
#             "including",
#             "such",
#             "as",
#             "and",
#             "or",
#             "but",
#             "performance-critical",
#             "sections",
#         }
#         logging.info("ExpertInstanceExtractor components are initialized and ready.")
#
#     def _is_valid_instance(self, span_to_check: str, abstract_concept: str) -> bool:
#         """
#         A final, intelligent validation gate to ensure the answer is a clean entity.
#         :param span_to_check: The cleaned answer string.
#         :param abstract_concept: The original concept being searched for.
#         :return: True if the span is a valid instance, False otherwise.
#         """
#         words = span_to_check.lower().split()
#
#         # Rule 1: Must not be excessively long (e.g., more than 3 words).
#         if len(words) > 3:
#             return False
#
#         # Rule 2: Must not be an "echo" of the abstract concept.
#         abstract_words = set(abstract_concept.lower().split())
#         if abstract_words.issubset(set(words)):
#             return False
#
#         # Rule 3: Must contain at least one "substantive" word.
#         if not any(word not in self.FUNCTION_WORDS for word in words):
#             return False
#
#         return True
#
#     def extract(
#         self,
#         context: str,
#         abstract_concept: str,
#         explanation_of_abstract: Optional[str] = None,
#     ) -> List[Tuple[str, float, int, int]]:
#         """
#         Runs the full extraction and filtering pipeline.
#         :param context: The text to search within.
#         :param abstract_concept: The abstract concept to find instances of.
#         :param explanation_of_abstract: An optional explanation of the abstract concept.
#         :return: A list of tuples, each containing (instance, score, start_offset, end_offset).
#         """
#         if not context or not abstract_concept:
#             return []
#         docs = [Document(content=context)]
#         questions = self.q_gen.run(
#             abstract_concept=abstract_concept, explanation=explanation_of_abstract
#         )["questions"]
#
#         # Step 1: Gather all possible answers from the reader for all questions.
#         all_raw_answers: List[Answer] = []
#         for query_text in questions:
#             try:
#                 reader_result = self.reader.run(query=query_text, documents=docs)
#                 all_raw_answers.extend(reader_result.get("answers", []))
#             except Exception as e:
#                 logging.error(f"Error running reader for question '{query_text}': {e}")
#                 continue
#
#         # Step 2: Apply the score threshold early for efficiency.
#         thresholded_answers = [
#             ans for ans in all_raw_answers if ans.score >= self.score_threshold
#         ]
#
#         # Step 3: Run the initial filtering component.
#         filter_result = self.filter.run(answers=thresholded_answers)
#         candidate_answers = filter_result["filtered_answers"]
#
#         # Step 4: Add a 'is_proper' flag to metadata for ranking.
#         for ans_item in candidate_answers:
#             if ans_item.data and ans_item.data[0].isupper():
#                 ans_item.meta["is_proper"] = True
#             else:
#                 ans_item.meta["is_proper"] = False
#
#         # Step 5: Re-rank candidates, prioritizing proper nouns, then by original score.
#         ranked_candidates = sorted(
#             candidate_answers,
#             key=lambda x: (x.meta.get("is_proper", False), x.score),
#             reverse=True,
#         )
#
#         # Step 6: Final processing loop with validation and de-duplication.
#         seen_strings = set()
#         results: List[Tuple[str, float, int, int]] = []
#         for current_ans in ranked_candidates:
#             if current_ans.data is None:
#                 continue
#
#             clean_span = current_ans.data.strip(string.punctuation + string.whitespace)
#
#             if self._is_valid_instance(clean_span, abstract_concept):
#                 if clean_span and clean_span.lower() not in seen_strings:
#                     # Get start and end from the .document_offset attribute, handling None.
#                     start_pos = (
#                         current_ans.document_offset.start
#                         if current_ans.document_offset
#                         else -1
#                     )
#                     end_pos = (
#                         current_ans.document_offset.end
#                         if current_ans.document_offset
#                         else -1
#                     )
#                     results.append(
#                         (clean_span, round(current_ans.score, 4), start_pos, end_pos)
#                     )
#                     seen_strings.add(clean_span.lower())
#
#         return results
#

# 1. Defines the core role and behavior for the model.
SYSTEM_BEHAVIOR_PROMPT = """
# System:
**Role**: You are an instance detector of a Special abstract.
**Behave**: Please find the instances ("answers") of "abstract" in the "context". the output must be a `list` of instances strings you found. Don't add any explanation and just return the list. if instance is not found, return empty list [].
**Output**: the output must be a list of instances you found. if instance is not found, return empty list [].
"""

# 2. Provides the few-shot examples to guide the model's output format.
FEW_SHOT_EXAMPLES_PROMPT = """
# samples:
**example 1**:

- "context": "In the ancient valley, a creature with obsidian scales and burning eyes guarded the gate. The villagers called it Narthul. Another dragon, a smaller but faster one, patrolled the skies. This second dragon was known as Ignis. Unlike Narthul, Ignis had shimmering silver scales."
- "abstract": "dragon"
- "explanation_of_abstract": "is a flying special and fire breath creature"
- "question": "A {dragon} {is a flying special and fire breath creature}. Which instances of {dragon} are described in the passage?"
+ "answers": ["Narthul", "Ignis"]

---
**example 2**:

- "context": "Queen Denis is flying with her balck dragon, the Drago and travel the seven kingdom at an afternoon!"
- "abstract": "car"
- "explanation_of_abstract": "is a transport vehicle in roads."
- "question": "A {car} {is a transport vehicle in roads}. Which instances of {car} are described in the passage?"
+ "answers": []

---
**example 3**:

- "context": "The royal messenger rode his white stallion across the kingdom. Beside him, a brown mare carried letters and supplies. These horses were bred for speed and endurance."
- "abstract": "horse"
- "explanation_of_abstract": "is an animal used for riding or transport"
- "question": "A {horse} {is an animal used for riding or transport}. Which instances of {horse} are described in the passage?"
+ "answers": ["white stallion", "brown mare"]

---
**example 4**:

- "context": "Under the starry sky, a sleek Tesla Model S sped down the empty highway. Moments later, a rusty Ford pickup rumbled past in the opposite direction."
- "abstract": "car"
- "explanation_of_abstract": "is a transport vehicle in roads."
- "question": "A {car} {is a transport vehicle in roads}. Which instances of {car} are described in the passage?"
+ "answers": ["Tesla Model S", "Ford pickup"]

---
**example 5**:

- "context": "He strode across the battlefield, gripping a weapon shaped like a crescent moon, gleaming under sunlight, whispering tales of forgotten wars. he draw his strange sword and draw it on a thief throat."
- "abstract": "curved sword"
- "explanation_of_abstract": "a curved sword is a bladed weapon with an arched edge, designed for swift, slicing attacks."
- "question": "An {curved sword} {is a curved sword is a bladed weapon with an arched edge, designed for swift, slicing attacks.}. Which instances of {curved sword} are described in the passage?"
+ "answers": ["a weapon shaped like a crescent moon", "his strange sword"]


---
**example 6**:

- "context": "At the zoo, the keeper introduced two animals: Zuri, the clever chimpanzee, and Bobo, a lazy orangutan who preferred sleeping to climbing trees."
- "abstract": "ape"
- "explanation_of_abstract": "is a kind of intelligent primate without a tail"
- "question": "An {ape} {is a kind of intelligent primate without a tail}. Which instances of {ape} are described in the passage?"
+ "answers": ["Zuri", "Bobo"]

---
"""

# 3. A template for the specific problem instance to be solved by the model.
PROBLEM_TEMPLATE = """
# problem:

- "context": "{context}"
- "abstract": "{abstract_concept}"
- "explanation_of_abstract": "{explanation_text}"
- "question": "{question}"
+ "answers" : 
"""

# 4. The prompt used for the self-correction step if parsing fails.
SELF_CORRECTION_PROMPT = """
The following text contains a Python list, but it is formatted incorrectly or has extra text.
Your task is to extract ONLY the valid JSON list from this text.
Do not add explanations. Do not change the items in the list.
Return an empty list `[]` if there are no items.

Original text:
---
{messy_output}
---

Corrected JSON list:
"""


class ExpertInstanceGenerator(ROASTDriver):
    """
    Orchestrates a generative model via Ollama to perform instance extraction.
    This version uses prompt templates defined outside the class for clarity
    and includes robust parsing with a self-correction mechanism.
    """

    def __init__(self, model_name: str = "llama3", max_retries: int = 3):
        self.model_name = model_name
        self.client = ollama.Client()
        self.max_retries = max_retries
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info(
            f"ExpertInstanceGenerator initialized with Ollama model: {self.model_name}"
        )

    def _build_prompt(
        self, context: str, abstract_concept: str, explanation: Optional[str]
    ) -> str:
        """
        Builds the complete prompt from the global templates.
        """
        explanation_text = explanation if explanation else "a concept"
        question = f"A {{{abstract_concept}}} {{{explanation_text}}}. Which instances of {{{abstract_concept}}} are described in the passage?"

        # Format the problem part using the template
        problem_part = PROBLEM_TEMPLATE.format(
            context=context,
            abstract_concept=abstract_concept,
            explanation_text=explanation_text,
            question=question,
        )

        return f"{SYSTEM_BEHAVIOR_PROMPT}{FEW_SHOT_EXAMPLES_PROMPT}{problem_part}"

    def _self_correct_output(self, messy_output: str) -> str:
        """
        Asks the LLM to clean its own messy output using the self-correction prompt template.
        """
        logging.info("Attempting self-correction with a cleaning prompt...")
        prompt = SELF_CORRECTION_PROMPT.format(messy_output=messy_output)
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.0},
            )
            corrected_text = response.get("response", "")
            logging.info(f"Self-correction returned: {corrected_text}")
            return corrected_text
        except Exception as e:
            logging.error(f"Self-correction LLM call failed: {e}")
            return ""

    def _find_positions(self, context: str, span: str) -> Tuple[int, int]:
        start = context.find(span)
        if start != -1:
            end = start + len(span)
            return start, end
        return -1, -1

    def _find_fuzzy_positions(
        self, context: str, span: str, threshold: int = 85
    ) -> Tuple[str, int, int]:
        """
        Finds the best fuzzy match for a span within the context.

        Instead of an exact match, it finds the word in the context with the
        highest similarity score to the span.

        :param context: The original text.
        :param span: The generated instance string to find a match for.
        :param threshold: The minimum similarity score (0-100) to consider a match.
        :return: A tuple of (best_match_word, start, end), or (None, -1, -1) if no match is found.
        """
        context_words = context.split()
        best_match_score = -1
        best_match_word = None

        # Find the word in the context that is most similar to the generated span
        for word in context_words:
            # We clean the word from common punctuation for a better match
            cleaned_word = re.sub(r"[.,!?;:]$", "", word)
            score = fuzz.ratio(span, cleaned_word)
            if score > best_match_score:
                best_match_score = score
                best_match_word = cleaned_word

        # If the best match found is above our threshold, find its position
        if best_match_score >= threshold:
            start = context.find(best_match_word)
            if start != -1:
                end = start + len(best_match_word)
                return best_match_word, start, end

        return "", -1, -1

    def _extract_list_from_text(self, raw_text: str) -> str:
        clean_text = raw_text.strip().replace("`", "")
        if clean_text.lower().startswith("json"):
            clean_text = clean_text[4:].strip()
        match = re.search(r"\[.*\]", clean_text, re.DOTALL)
        if match:
            return match.group(0)
        return clean_text

    def _parse_response(
        self, response_text: str, context: str
    ) -> Optional[List[Tuple[str, float, int, int]]]:
        list_text = self._extract_list_from_text(response_text)
        parsed_list = None
        try:
            parsed = json.loads(list_text)
            if isinstance(parsed, list):
                parsed_list = parsed
        except json.JSONDecodeError:
            pass
        if parsed_list is None:
            try:
                parsed = ast.literal_eval(list_text)
                if isinstance(parsed, list):
                    parsed_list = parsed
            except (ValueError, SyntaxError):
                logging.warning(
                    f"Failed to parse text as a list with both json and ast: {list_text}"
                )
                return None
        if parsed_list is None:
            return None
        # --- MODIFIED PART ---
        results = []
        for item in parsed_list:
            generated_instance = str(item)
            start, end = self._find_positions(context, generated_instance)
            if start == -1 or end == -1:
                logging.warning(
                    f"Could not find instance '{generated_instance}' in the original context."
                )
                # Use the new fuzzy finding method
                found_instance, start, end = self._find_fuzzy_positions(
                    context, generated_instance, threshold=60
                )
                if found_instance:
                    results.append((found_instance, 1.0, start, end))
                else:
                    logging.warning(
                        f"Could not find a confident fuzzy match for '{generated_instance}' in the context."
                    )
            else:
                results.append((generated_instance, 1.0, start, end))
        return results

    def extract(
        self,
        context: str,
        abstract_concept: str,
        explanation_of_abstract: Optional[str] = None,
    ) -> List[Tuple[str, float, int, int]]:
        if not context or not abstract_concept:
            return []
        prompt = self._build_prompt(context, abstract_concept, explanation_of_abstract)
        for attempt in range(self.max_retries):
            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=False,
                    options={"temperature": 0.0},
                )
                raw_response_text = response.get("response", "")
                parsed_results = self._parse_response(raw_response_text, context)
                if parsed_results is not None:
                    return parsed_results
                logging.warning(
                    f"Initial parsing failed on attempt {attempt + 1}. Raw response: '{raw_response_text}'"
                )
                corrected_text = self._self_correct_output(raw_response_text)
                if corrected_text:
                    corrected_results = self._parse_response(corrected_text, context)
                    if corrected_results is not None:
                        return corrected_results
                logging.warning(
                    f"Self-correction attempt {attempt + 1} also failed to yield a parsable list."
                )
            except Exception as e:
                logging.error(
                    f"Ollama API call failed on attempt {attempt + 1}/{self.max_retries}: {e}"
                )
        logging.error(
            f"Failed to get a valid, parsable response after {self.max_retries} attempts and self-correction."
        )
        return []


TEST_CASES = [
    # 0 instances
    {
        "id": 1,
        "context": "The starship hummed quietly as it drifted through the interstellar void, light-years from any known system. On the bridge, the crew monitored the long-range sensors, searching for any sign of habitable worlds. The journey had been long and uneventful, a slow crawl through the cosmic dark. Below deck, the hydroponics bay provided a splash of green, a reminder of the home they had left behind. Their mission was one of pure exploration, to chart the unknown and expand the boundaries of human knowledge. The captain gazed at the star-dusted viewscreen, a silent testament to the vastness of the universe and their small place within it. The silence was profound, broken only by the ships gentle thrum.",
        "abstract": "alien species",
        "explanation_of_abstract": "is a life form that does not originate from Earth.",
        "instances": [],
    },
    {
        "id": 2,
        "context": "The library was a sanctuary of silence and the smell of old paper. Sunlight streamed through the tall, arched windows, illuminating dust motes dancing in the air. Shelves stretched from floor to ceiling, packed with books bound in leather and cloth. Each volume was a gateway to another world, another time, another mind. A young student sat at a heavy oak table, poring over a manuscript about ancient cartography. He traced the faded lines of a map depicting lands that no longer existed, seas that had long since dried up. The world outside, with its noise and haste, seemed a distant dream. Here, only the rustle of turning pages broke the stillness.",
        "abstract": "magic spell",
        "explanation_of_abstract": "is a mystical incantation or formula that produces a supernatural effect.",
        "instances": [],
    },
    # 1 instance
    {
        "id": 3,
        "context": "The city of Aethelgard was carved from the heart of a mountain, its stone towers reaching for the cavernous ceiling. A single, massive gate was the only entrance, guarded by warriors in silver armor. Deep within the city, the forges burned day and night, crafting the finest steel in the kingdom. Merchants from distant lands traveled treacherous paths to trade for these legendary blades. The citys ruler, a stern but fair dwarf king, oversaw all from his throne of unpolished granite. His people were known for their resilience and craftsmanship, qualities reflected in the unyielding stone of their home. Their isolation bred a fierce independence, making them wary of outsiders but loyal to their own.",
        "abstract": "city",
        "explanation_of_abstract": "is a large and permanent human settlement with complex systems of infrastructure and governance.",
        "instances": ["Aethelgard"],
    },
    {
        "id": 4,
        "context": "The rebellion needed a symbol, a spark to ignite the flames of revolution across the oppressed districts. They found it in the Firehawk, a modified freighter ship that could outrun any patrol cruiser. It became a legend, a ghost in the shipping lanes, striking at military convoys and disappearing without a trace. Its captain, a mysterious figure known only as Kael, was the heart of the resistance. The ship itself was as much a character in the conflict as any person, its scorched hull and roaring engines a promise of freedom. For the downtrodden, the sight of the Firehawk streaking across the sky was a sign that hope was not yet lost, that the fight would continue.",
        "abstract": "starship",
        "explanation_of_abstract": "is a spacecraft designed for interstellar travel.",
        "instances": ["Firehawk"],
    },
    # 2 instances
    {
        "id": 5,
        "context": "In the ancient valley, a creature with obsidian scales and burning eyes guarded the gate. The villagers called it Narthul, a terror of the peaks. This fearsome dragon had slumbered for centuries, awakened by the greed of miners digging too deep. Another of its kind, a smaller but faster beast known as Ignis, patrolled the skies above the southern coast. Unlike the mountain-dweller, Ignis was a creature of the sea and storm, its scales shimmering like wet jewels. Sailors spoke of it in hushed tones, a beautiful but deadly omen. The two had never met, each a sovereign ruler of its own domain, their legends woven separately into the tapestry of the land.",
        "abstract": "dragon",
        "explanation_of_abstract": "is a large, serpentine legendary creature that appears in the folklore of many cultures worldwide.",
        "instances": ["Narthul", "Ignis"],
    },
    {
        "id": 6,
        "context": "The Galactic Federation relied on two primary classes of autonomous vessels for deep space operations. The Goliath was a massive harvester, designed to mine asteroids and comets for vital resources. These slow-moving titans were crewed by only a handful of technicians, as most of their functions were automated. For defense and patrol, the Federation deployed the Viper, a swift and agile interceptor. These smaller ships were purely robotic, controlled by a central AI network from the capital planet. While the Goliath was built for endurance and capacity, the Viper was designed for speed and combat, a perfect pairing of industrial might and military precision that secured the Federations borders.",
        "abstract": "spacecraft",
        "explanation_of_abstract": "is a vehicle or machine designed to fly in outer space.",
        "instances": ["Goliath", "Viper"],
    },
    # 3 instances
    {
        "id": 7,
        "context": "The alchemists workshop was a chaotic mess of bubbling beakers and arcane charts. On a central pedestal sat his three greatest achievements. The first was the Philosophers Stone, a ruby-red gem that pulsed with a faint inner light, said to be capable of transmuting lead into gold. Beside it was a vial containing the Elixir of Life, a shimmering silver liquid that promised to cure any disease and grant extended youth. His final creation was a small, unassuming wooden box known as the Aetherium, which he claimed could capture and store memories. These three artifacts represented the pinnacle of his art, the culmination of a lifetime dedicated to unlocking the hidden secrets of the material world.",
        "abstract": "magical artifact",
        "explanation_of_abstract": "is an object that possesses supernatural or magical properties.",
        "instances": ["Philosophers Stone", "Elixir of Life", "Aetherium"],
    },
    {
        "id": 8,
        "context": "The history of the digital frontier was written by a handful of legendary programs. The first was Archon, a simple but elegant search algorithm that laid the groundwork for all that followed. Then came Nexus, a complex networking protocol that allowed disparate systems to communicate seamlessly, creating the first true global network. The most famous, however, was Morpheus, a virtual reality interface so immersive that users could forget they were in a simulation. These three pieces of software were the pillars upon which the modern information age was built. Each was a revolution in its own right, pushing the boundaries of what was thought possible and forever changing how humanity interacted with data.",
        "abstract": "software program",
        "explanation_of_abstract": "is a set of instructions or data that tells a computer how to work.",
        "instances": ["Archon", "Nexus", "Morpheus"],
    },
    # ... continue for 92 more samples ...
    {
        "id": 9,
        "context": "The ancient Greeks told tales of many heroes, each with their own legendary feats. Heracles, known for his incredible strength, completed twelve impossible labors, from slaying the Nemean Lion to capturing Cerberus. Another celebrated figure was Odysseus, the clever king of Ithaca, whose ten-year journey home from the Trojan War was filled with perilous encounters with mythical creatures and wrathful gods. His cunning, rather than his strength, was his greatest asset. Then there was Achilles, the greatest warrior of the Achaean army, whose only vulnerability was his heel. His rage and prowess in battle were central to the epic of the Iliad, a mortal man who fought like a god.",
        "abstract": "mythological hero",
        "explanation_of_abstract": "is a person of great strength and courage, celebrated for their bold exploits in ancient legends.",
        "instances": ["Heracles", "Odysseus", "Achilles"],
    },
    {
        "id": 10,
        "context": "The museums new exhibit focused on three of the most influential inventions of the industrial revolution. The centerpiece was a fully restored Spinning Jenny, a multi-spindle spinning frame that revolutionized the textile industry by drastically increasing cloth production. Next to it stood a miniature, working model of the Steam Engine, the powerhouse that drove factories, locomotives, and ships, fundamentally changing transportation and manufacturing. The final display showcased the Telegraph, a device that allowed for the first time in history, instantaneous communication across vast distances. These innovations were the catalysts for a new era of technological advancement and societal change, laying the groundwork for the modern world we live in today.",
        "abstract": "invention",
        "explanation_of_abstract": "is a unique or novel device, method, composition or process.",
        "instances": ["Spinning Jenny", "Steam Engine", "Telegraph"],
    },
    {
        "id": 11,
        "context": "The intelligence agency operated with a series of highly classified espionage programs. Project Chimera was their most ambitious, a deep-cover operation to infiltrate foreign governments using agents with surgically altered appearances. For domestic surveillance, they relied on Operation Echelon, a massive data-mining system that could intercept and analyze millions of communications simultaneously. Their most secret initiative, however, was Stargate, a theoretical research division exploring the possibility of psychic warfare and remote viewing. These programs, while controversial, were deemed essential for national security by the agencys directors. They operated in the shadows, their successes and failures unknown to the public they were sworn to protect, a hidden layer of power.",
        "abstract": "espionage program",
        "explanation_of_abstract": "is a covert operation designed to obtain secret information.",
        "instances": ["Project Chimera", "Operation Echelon", "Stargate"],
    },
    {
        "id": 12,
        "context": "The ecosystem of the Crimson Forest was unlike any other on the planet. Its dominant predator was the Shadow Cat, a sleek, black feline with nocturnal habits and the ability to move in complete silence. It hunted a variety of prey, but its favorite was the Glimmerwing, a large, herbivorous insect with bioluminescent wings that it used for mating displays. Both species relied on the Sunpetal flower, a plant that tracked the sun and provided a high-energy nectar. The intricate relationship between these three organisms was a perfect example of co-evolution. The forests health depended on the balance between them, a delicate dance of hunter, hunted, and the life-giving flora that sustained them both.",
        "abstract": "organism",
        "explanation_of_abstract": "is an individual animal, plant, or single-celled life form.",
        "instances": ["Shadow Cat", "Glimmerwing", "Sunpetal"],
    },
    {
        "id": 13,
        "context": "The companys success was built on two key software products that dominated the market. The first, QuantumLeap, was a financial modeling tool so powerful it could predict market fluctuations with uncanny accuracy. It was the gold standard for investment banks and hedge funds worldwide. Their second flagship product was ConnectSphere, a social networking platform that prioritized user privacy and data security, a direct response to the controversies plaguing its competitors. While QuantumLeap was their primary revenue generator, ConnectSphere was their public face, earning them a reputation as an ethical and trustworthy tech giant. Together, they formed a symbiotic relationship, with the profits from one funding the growth and innovation of the other.",
        "abstract": "software product",
        "explanation_of_abstract": "is a commercially produced piece of software.",
        "instances": ["QuantumLeap", "ConnectSphere"],
    },
    {
        "id": 14,
        "context": "The ancient empire was guarded by two legendary legions, each with its own distinct history and reputation. The Tenth Legion, also known as the Iron Gryphons, were masters of siege warfare, renowned for their engineering skills and unbreakable defensive formations. They were the immovable object against which countless armies had shattered. In contrast, the Third Legion, or the Crimson Blades, were the empires swift sword. They specialized in rapid assaults and flanking maneuvers, often winning battles before their opponents had time to mount a proper defense. While the Iron Gryphons were celebrated for their discipline, the Crimson Blades were feared for their ferocity, two sides of the same imperial coin that kept the peace for centuries.",
        "abstract": "military unit",
        "explanation_of_abstract": "is a component of an armed force.",
        "instances": ["Tenth Legion", "Third Legion"],
    },
    {
        "id": 15,
        "context": "The grand library of Alexandria was said to have housed the greatest collection of knowledge in the ancient world. Within its scrolls, one could find the lost plays of Sophocles, detailed astronomical charts from Babylonian observers, and the complete histories of the pharaohs. It was more than a collection of books; it was a center for scholarship, attracting the greatest minds from across the Mediterranean and beyond. Scholars debated philosophy, calculated the circumference of the Earth, and translated works from dozens of languages. Its eventual destruction, a gradual decline culminating in a final, tragic fire, represented an incalculable loss for civilization, a dark age brought on by the loss of accumulated wisdom and learning.",
        "abstract": "historical location",
        "explanation_of_abstract": "is a place of significance in history.",
        "instances": ["Alexandria"],
    },
    {
        "id": 16,
        "context": "The colonization of Mars was a monumental undertaking, requiring vehicles of unprecedented design. The Ares 7 was the primary transport, a massive interplanetary vessel capable of carrying a crew of six and enough supplies for the two-year journey. Once on the surface, the astronauts relied on the Beagle, a pressurized rover designed for long-range exploration and geological surveys. This rugged vehicle served as their mobile base and laboratory. The final piece of the puzzle was the Icarus, a small, autonomous drone used for aerial mapping and scouting hazardous terrain. These three machines were the workhorses of the early Martian settlements, their robust designs ensuring the survival of the first off-world humans.",
        "abstract": "vehicle",
        "explanation_of_abstract": "is a machine that transports people or cargo.",
        "instances": ["Ares 7", "Beagle", "Icarus"],
    },
    {
        "id": 17,
        "context": "The old mariner spoke of the terrors of the deep, the colossal beasts that lurked beneath the waves. He claimed to have seen the Kraken, a squid of such immense size that its tentacles could pull an entire galleon to the ocean floor. He also told stories of the Leviathan, a great sea serpent whose scales were as impenetrable as shields and whose passing churned the sea into a maelstrom. These tales were dismissed by most as the drunken ramblings of a lonely old man. But for the young boy listening wide-eyed on the docks, they were a thrilling glimpse into a world of adventure and mystery, a world far removed from his quiet fishing village.",
        "abstract": "sea monster",
        "explanation_of_abstract": "is a large, monstrous creature believed to inhabit the sea.",
        "instances": ["Kraken", "Leviathan"],
    },
    {
        "id": 18,
        "context": "The deep space probe Voyager 3 drifted silently through the Oort cloud, its mission long since completed. It carried a golden record, a message in a bottle from a species on a small blue planet. The record contained sounds and images selected to portray the diversity of life and culture on Earth. It was a greeting card to the cosmos, a hopeful gesture sent out into the vast, unknown darkness. The probe itself was a relic of a bygone era of exploration, its power source nearly depleted. Yet it continued its journey, a silent ambassador carrying a message of peace, destined to wander the galaxy for eons, a testament to its creators curiosity and dreams.",
        "abstract": "space probe",
        "explanation_of_abstract": "is an uncrewed spacecraft that explores outer space.",
        "instances": ["Voyager 3"],
    },
    {
        "id": 19,
        "context": "The political landscape was dominated by two opposing ideologies. The Collectivists believed that the needs of the state superseded the rights of the individual, advocating for central planning and the abolition of private property. Their symbol was a gear, representing each persons role in the larger social machine. On the other side were the Libertarians, who championed individual freedom and minimal government intervention. They argued that a free market and personal responsibility were the keys to prosperity. Their symbol was a torch, signifying the light of individual liberty. The clash between these two philosophies defined the era, leading to decades of heated debate, political maneuvering, and social unrest across the nation.",
        "abstract": "political ideology",
        "explanation_of_abstract": "is a coherent set of ideas and beliefs that directs ones goals, expectations, and actions.",
        "instances": ["Collectivists", "Libertarians"],
    },
    {
        "id": 20,
        "context": "The patients treatment plan was complex, involving several next-generation pharmaceuticals. The primary drug was Omnizole, a targeted therapy that inhibited the growth of cancer cells without harming healthy tissue. This was supplemented by Immunoboost, a medication designed to enhance the bodys own natural defenses, helping the immune system identify and destroy the malignant cells. The final component was Neurocalm, a powerful anxiolytic to manage the psychological stress of the illness. This three-pronged approach represented the cutting edge of oncological medicine, a personalized strategy that attacked the disease from multiple angles while supporting the patients overall well-being. The hope was that this combination would lead to a full and lasting remission.",
        "abstract": "pharmaceutical drug",
        "explanation_of_abstract": "is a chemical substance used to treat, cure, prevent, or diagnose a disease.",
        "instances": ["Omnizole", "Immunoboost", "Neurocalm"],
    },
    {
        "id": 21,
        "context": "The detective surveyed the crime scene, a high-tech laboratory that had been ransacked. The company, a leader in robotics, had been working on a new prototype. According to the lead scientist, the missing unit, codenamed Prometheus, was their most advanced creation yet. It featured a revolutionary neural network that allowed it to learn and adapt in ways that mimicked human intuition. The scientist feared it could have been stolen by a rival corporation, a major setback for their research and a potential threat if its technology was misused. The detective knew this wasnt a simple case of industrial espionage; the stakes felt much higher. The search for the missing android had just begun.",
        "abstract": "android",
        "explanation_of_abstract": "is a robot with a human appearance.",
        "instances": ["Prometheus"],
    },
    {
        "id": 22,
        "context": "The old book described a pantheon of forgotten gods who once ruled the primordial world. Aethel, the Sky-Father, was the king of the gods, who painted the dawn and dusk with his own hands. His consort was Gaea, the Earth-Mother, from whom all living things were born. Their rebellious son, Pyros, was the god of fire and chaos, who stole the sun and brought about the first winter. These deities were not worshipped with temples and priests, but were seen in the natural world—the storm clouds, the fertile soil, the destructive wildfire. Their stories were cautionary tales, reminders of the power and unpredictability of nature itself, a force to be respected.",
        "abstract": "deity",
        "explanation_of_abstract": "is a god or goddess in a polytheistic religion.",
        "instances": ["Aethel", "Gaea", "Pyros"],
    },
    {
        "id": 23,
        "context": "The galactic stock market was in turmoil after the collapse of two major interstellar corporations. Cygnus Freight, once the largest shipping company in the sector, declared bankruptcy after losing its primary shipping lane to pirates. The second, Stellar Mining, defaulted on its loans when its largest asteroid mine was found to be depleted, rendering the company worthless overnight. The ripple effect was catastrophic, with smaller businesses failing and investors losing fortunes. The Galactic Commerce Commission called an emergency session to halt trading and prevent a complete market meltdown. It was a stark reminder of the volatility of the space-based economy, where fortunes could be made and lost in the blink of an eye.",
        "abstract": "corporation",
        "explanation_of_abstract": "is a company or group of people authorized to act as a single entity.",
        "instances": ["Cygnus Freight", "Stellar Mining"],
    },
    {
        "id": 24,
        "context": "The fantasy novel featured a rich tapestry of different races, each with their own unique culture. The Elves of Silverwood were graceful and wise, living in harmony with nature for thousands of years. In the mountain strongholds lived the Dwarves, master craftsmen and fierce warriors, who valued gold and honor above all else. The vast plains were home to the nomadic tribes of Orcs, a proud and shamanistic people often misunderstood by the other races. The interactions between these three groups drove the main plot, a story of ancient grudges, unlikely alliances, and the struggle to find common ground in a world filled with prejudice and conflict.",
        "abstract": "fantasy race",
        "explanation_of_abstract": "is a fictional sentient species found in fantasy literature or games.",
        "instances": ["Elves", "Dwarves", "Orcs"],
    },
    {
        "id": 25,
        "context": "The historian was studying the rise and fall of ancient empires. She was particularly fascinated by the Roman Empire, with its vast infrastructure, legal systems, and military might that dominated the Mediterranean for centuries. She contrasted its trajectory with that of the Persian Empire, a rival power in the East known for its tolerance of different cultures and its efficient bureaucracy under rulers like Cyrus the Great. Both were superpowers of their time, their conflicts and interactions shaping the course of Western and Middle Eastern history. Understanding their strengths and weaknesses, she believed, held valuable lessons for the political dynamics of the modern world, proving that history often repeats itself.",
        "abstract": "empire",
        "explanation_of_abstract": "is an extensive group of states or countries under a single supreme authority.",
        "instances": ["Roman Empire", "Persian Empire"],
    },
    {
        "id": 26,
        "context": "The botanist was on an expedition in the Amazon rainforest, searching for undiscovered plant species with medicinal properties. After weeks of searching, she found a rare orchid, which the local tribe called the Moonpetal, believed to have powerful healing abilities. She carefully documented its location and took a small sample for analysis. Her work was crucial, as the rainforest was under constant threat from deforestation. Every species lost was a potential cure that vanished forever. The expedition was a race against time, a quest to unlock the secrets of the jungle before they were silenced by the roar of chainsaws and the march of progress.",
        "abstract": "plant species",
        "explanation_of_abstract": "is a group of living organisms consisting of similar individuals capable of exchanging genes or interbreeding.",
        "instances": ["Moonpetal"],
    },
    {
        "id": 27,
        "context": "The art gallery was hosting a retrospective on the works of two pioneering abstract painters. The first section was dedicated to Kandinsky, whose use of color and geometric shapes was revolutionary, aiming to evoke an emotional response rather than depict a visual reality. The second half of the gallery featured the works of Pollock, famous for his drip technique, where he poured or splashed paint onto a horizontal canvas. His energetic and seemingly chaotic compositions challenged conventional notions of control and artistry. Though their methods were vastly different, both artists were instrumental in breaking away from traditional representation and paving the way for modern art, forever changing the landscape of painting.",
        "abstract": "artist",
        "explanation_of_abstract": "is a person who creates paintings or drawings as a profession or hobby.",
        "instances": ["Kandinsky", "Pollock"],
    },
    {
        "id": 28,
        "context": "The video game world was abuzz with the announcement of a new title from a legendary developer. The game, titled Cyberia, was set in a dystopian future where players had to navigate a sprawling, neon-lit metropolis. It promised a branching narrative and unprecedented player freedom. The hype was immense, fueled by the developers track record of creating immersive and critically acclaimed experiences. Fans eagerly dissected every screenshot and trailer, speculating about the story and gameplay mechanics. In an industry often criticized for its lack of originality, this release was seen as a beacon of creativity, a potential masterpiece that could redefine the open-world genre for a new generation of gamers.",
        "abstract": "video game",
        "explanation_of_abstract": "is an electronic game that involves interaction with a user interface to generate visual feedback on a video device.",
        "instances": ["Cyberia"],
    },
    {
        "id": 29,
        "context": "The lecture on particle physics covered the fundamental building blocks of the universe. The professor explained the nature of the Quark, a tiny elementary particle that combines to form protons and neutrons. He then discussed the Lepton, a family of particles that includes the electron and the elusive neutrino. Finally, he touched upon the Boson, a category of force-carrying particles, famously including the Higgs boson, which gives other particles their mass. The concepts were mind-bending, a journey into a subatomic realm governed by the strange laws of quantum mechanics. For the students, it was a glimpse into the very fabric of reality, a world far removed from everyday experience.",
        "abstract": "elementary particle",
        "explanation_of_abstract": "is a subatomic particle with no substructure, thus not composed of other particles.",
        "instances": ["Quark", "Lepton", "Boson"],
    },
    {
        "id": 30,
        "context": "The documentary explored the great rivers of the world, lifelines for civilizations throughout history. It began with the Nile, whose predictable annual floods sustained ancient Egyptian agriculture, allowing a great civilization to flourish in the desert. The film then traveled to South America to showcase the Amazon, the largest river by discharge volume, a vast, teeming ecosystem that is home to an incredible diversity of life. Finally, it journeyed to Asia to follow the course of the Ganges, a river considered sacred by millions of Hindus, who perform religious rituals in its waters. Each river had its own unique story, a powerful testament to the vital role of waterways in shaping human culture and the natural world.",
        "abstract": "river",
        "explanation_of_abstract": "is a large natural stream of water flowing in a channel to the sea, a lake, or another river.",
        "instances": ["Nile", "Amazon", "Ganges"],
    },
    {
        "id": 31,
        "context": "The old sailors map was covered in warnings and illustrations of mythical islands. One was Avalon, a legendary isle from Arthurian legend, said to be a place of magic where King Arthur was taken after his final battle. Another location marked on the map was Atlantis, the fabled city described by Plato, which supposedly sank into the ocean in a single day and night of misfortune. He also had a crude drawing of El Dorado, the lost city of gold, which drove countless conquistadors to their doom in the jungles of South America. These places, though likely fictional, represented the enduring human fascination with lost worlds and hidden treasures, the dream of discovering paradise on Earth.",
        "abstract": "mythical location",
        "explanation_of_abstract": "is a place that exists only in mythology or fiction.",
        "instances": ["Avalon", "Atlantis", "El Dorado"],
    },
    {
        "id": 32,
        "context": "The field of artificial intelligence has seen the development of several landmark systems. One of the earliest was Deep Blue, a chess-playing computer developed by IBM that famously defeated world champion Garry Kasparov in 1997. More recently, Googles AlphaGo mastered the ancient game of Go, a feat once thought impossible for a machine due to the games intuitive nature. Both systems demonstrated the power of machine learning and specialized algorithms. They were not general intelligences, but highly optimized problem-solvers that could surpass human ability in a specific domain. Their victories were milestones that sparked both excitement and debate about the future of AI and its potential impact on society.",
        "abstract": "artificial intelligence",
        "explanation_of_abstract": "is the theory and development of computer systems able to perform tasks that normally require human intelligence.",
        "instances": ["Deep Blue", "AlphaGo"],
    },
    {
        "id": 33,
        "context": "The fantasy world was governed by a council of ancient, powerful beings. The most revered was Bahamut, the Platinum Dragon, a god of justice and protection, worshipped by paladins and good-aligned creatures. His eternal rival was Tiamat, the Chromatic Dragon, a five-headed goddess of greed and vengeance, who commanded legions of evil dragons and cultists. The two represented the timeless struggle between good and evil. Their conflict was not fought with armies, but through their mortal followers and subtle manipulations of fate. The balance of power between them was delicate, and any shift could plunge the world into an age of light or an era of darkness, depending on which deity gained the upper hand.",
        "abstract": "deity",
        "explanation_of_abstract": "is a god or goddess in a polytheistic religion.",
        "instances": ["Bahamut", "Tiamat"],
    },
    {
        "id": 34,
        "context": "The astronomy class was studying the planets of our solar system. The teacher first described Jupiter, the largest planet, a gas giant with a famous Great Red Spot, which is a storm larger than Earth. She then moved on to Saturn, known for its spectacular ring system, composed of ice and rock particles. Finally, she discussed Mars, the red planet, which has captured human imagination for centuries with the possibility of past or present life. The students were captivated by the diversity of these worlds, from the immense scale of the gas giants to the rocky, cratered surface of our planetary neighbor. Each planet was a unique destination, a world with its own distinct character and mysteries waiting to be explored.",
        "abstract": "planet",
        "explanation_of_abstract": "is a celestial body moving in an elliptical orbit around a star.",
        "instances": ["Jupiter", "Saturn", "Mars"],
    },
    {
        "id": 35,
        "context": "The special forces team was equipped with the latest military hardware. Their standard issue assault rifle was the SCAR-H, a reliable and versatile weapon effective at medium range. For close-quarters combat, they carried the MP7, a compact submachine gun with a high rate of fire. Each squad also had a designated marksman who used the M110, a semi-automatic sniper rifle capable of engaging targets at great distances. The effectiveness of the team depended on their mastery of these tools and their ability to choose the right weapon for the situation. Their gear was an extension of themselves, finely tuned instruments of war that gave them a critical edge in the field of battle.",
        "abstract": "firearm",
        "explanation_of_abstract": "is a rifle, pistol, or other portable gun.",
        "instances": ["SCAR-H", "MP7", "M110"],
    },
    {
        "id": 36,
        "context": "The wizards grimoire contained instructions for summoning powerful elemental spirits. The first was Salamander, a being of pure fire that appeared as a lizard wreathed in flames, capable of incinerating anything it touched. The second was Undine, a spirit of water, who took the form of a beautiful maiden and could control rivers and tides. The third was Sylph, an invisible spirit of the air, who could create powerful gusts of wind or carry messages silently across great distances. To summon and control these beings required immense skill and willpower, as they were wild and dangerous forces of nature. A single mistake in the ritual could lead to the wizards own destruction, consumed by the very power they sought to command.",
        "abstract": "elemental spirit",
        "explanation_of_abstract": "is a mythical being that is attuned with one of the four classical elements.",
        "instances": ["Salamander", "Undine", "Sylph"],
    },
    {
        "id": 37,
        "context": "The automotive museum showcased a collection of iconic sports cars from different eras. One of the highlights was a pristine 1962 Ferrari 250 GTO, considered by many to be the most beautiful and valuable car ever made. Next to it was a 1994 McLaren F1, a technological marvel that held the record for the fastest production car for over a decade. The final car in the trio was a modern Porsche 911 GT3, a track-focused machine that represented the pinnacle of contemporary automotive engineering. Each vehicle was a masterpiece of design and performance, a symbol of its time. They were more than just machines; they were rolling sculptures, expressions of speed, style, and the relentless pursuit of automotive perfection.",
        "abstract": "sports car",
        "explanation_of_abstract": "is a car designed with an emphasis on dynamic performance, such as handling, acceleration, or thrill of driving.",
        "instances": ["Ferrari 250 GTO", "McLaren F1", "Porsche 911 GT3"],
    },
    {
        "id": 38,
        "context": "The folklore of the region was rich with tales of mischievous nature spirits. The Pixies were said to live in mushroom circles and loved to play harmless pranks on travelers, like tying their shoelaces together or leading them astray on forest paths. The Goblins, on the other hand, were more malevolent creatures that lived in caves and were known for hoarding stolen trinkets and causing trouble for nearby villages. The two groups were often at odds, their territories overlapping in the ancient woods. The villagers learned to leave offerings of milk for the pixies to stay in their good graces, while avoiding the dark caves that the goblins called home. It was a delicate balance of superstition and respect.",
        "abstract": "mythical creature",
        "explanation_of_abstract": "is a supernatural animal or being, especially one that is imaginary or legendary.",
        "instances": ["Pixies", "Goblins"],
    },
    {
        "id": 39,
        "context": "The computer science course covered several foundational programming languages. The first was C, a powerful and efficient language that gives developers a lot of control over computer memory. It formed the basis for many other languages and operating systems. Next, the professor introduced Python, a high-level language known for its simple, clean syntax, which makes it popular for beginners, data science, and web development. Finally, they discussed Java, an object-oriented language designed to be portable, with the philosophy of write once, run anywhere. Understanding the strengths and weaknesses of these three languages was essential for any aspiring software engineer, as they represented different approaches to problem-solving and software design in the digital world.",
        "abstract": "programming language",
        "explanation_of_abstract": "is a formal language comprising a set of instructions that produce various kinds of output.",
        "instances": ["C", "Python", "Java"],
    },
    {
        "id": 40,
        "context": "The spy thriller revolved around the hunt for a notorious assassin known only by his codename, Ghost. He was a phantom, leaving no trace and seemingly able to bypass any security system. The only person who had ever survived an encounter with him was a disgraced MI6 agent named Alex, who was brought back into service to track him down. The cat-and-mouse game between them spanned the globe, from the crowded markets of Istanbul to the sterile skyscrapers of Tokyo. Alex knew that to catch a ghost, she had to think like one, anticipating his moves and using his own methods against him. It was a deadly duel of wits and skill, where a single mistake would be fatal.",
        "abstract": "spy",
        "explanation_of_abstract": "is a person who secretly collects and reports information on the activities, movements, and plans of an enemy or competitor.",
        "instances": ["Ghost", "Alex"],
    },
    {
        "id": 41,
        "context": "The royal armory housed a collection of legendary swords, each with its own name and history. The most famous was Excalibur, the mythical blade of King Arthur, said to have been given to him by the Lady of the Lake and possessing magical powers. Another prized weapon was Durandal, the sword of the paladin Roland, which was said to be indestructible and contain a sacred relic within its golden hilt. The final sword on display was Kusanagi, a blade from Japanese mythology, found in the body of an eight-headed serpent and used by the storm god Susanoo. These weapons were more than just steel; they were symbols of power, destiny, and the heroic ideals of the cultures that created their legends.",
        "abstract": "legendary sword",
        "explanation_of_abstract": "is a sword that is famous in mythology or legend.",
        "instances": ["Excalibur", "Durandal", "Kusanagi"],
    },
    {
        "id": 42,
        "context": "The fantasy epic followed the intertwined destinies of three noble houses. The Starks of the North were a hardy and honorable family, ruling from their ancestral castle of Winterfell, their motto a grim reminder that Winter is Coming. In the wealthy western lands, the Lannisters held power, a cunning and ambitious family whose vast fortune was their greatest weapon. Far to the east, across the sea, the last scion of the Targaryens, a dynasty of dragon riders, plotted her return to reclaim the throne her ancestors had forged. The political maneuvering, betrayals, and wars between these great houses formed the central conflict of the sprawling narrative, a complex game of thrones where you win or you die.",
        "abstract": "noble house",
        "explanation_of_abstract": "is a family of high social or political rank.",
        "instances": ["Starks", "Lannisters", "Targaryens"],
    },
    {
        "id": 43,
        "context": "The course on Greek mythology detailed the three most powerful Olympian gods. Zeus, the king of the gods, ruled the sky and wielded the thunderbolt as his weapon. His brother, Poseidon, was the lord of the sea, his temper as unpredictable as the ocean itself, and he carried a mighty trident. The third brother, Hades, governed the underworld, the realm of the dead, a gloomy and mysterious figure who rarely left his domain. These three brothers had divided the world among themselves after overthrowing their father, Cronus, and the Titans. Their rule, marked by family squabbles, heroic quests, and divine interventions in mortal affairs, formed the core of Greek religious belief and storytelling for centuries.",
        "abstract": "Greek god",
        "explanation_of_abstract": "is a deity from the pantheon of ancient Greek mythology.",
        "instances": ["Zeus", "Poseidon", "Hades"],
    },
    {
        "id": 44,
        "context": "The naval museum had detailed models of three of the most famous warships in history. The HMS Victory, a British ship of the line, was best known as Lord Nelsons flagship at the Battle of Trafalgar. Its wooden hull and towering masts were a symbol of the age of sail. Representing a later era was the Bismarck, a formidable German battleship from World War II, whose sinking was a major turning point in the Atlantic campaign. The final model was the USS Enterprise, the first nuclear-powered aircraft carrier, a floating city that projected American power across the globe for over 50 years. Each vessel represented the pinnacle of naval technology for its time, a testament to human ingenuity in the art of sea warfare.",
        "abstract": "warship",
        "explanation_of_abstract": "is a ship equipped with weapons and designed to take part in warfare at sea.",
        "instances": ["HMS Victory", "Bismarck", "USS Enterprise"],
    },
    {
        "id": 45,
        "context": "The high-fantasy novel described a world where magic was drawn from three distinct sources. Arcane magic was the most common, a scholarly pursuit involving complex incantations and gestures to manipulate the raw energies of the universe. The second type was Divine magic, granted by the gods to their faithful clerics and paladins as a reward for their devotion. The final and rarest form was Primal magic, an intuitive and wild power drawn from the natural world itself, used by druids and rangers. A mage might spend their life studying a single spell, while a priest could call down a miracle with a simple prayer. This diversity of magical systems created a rich and complex world for the characters to navigate.",
        "abstract": "type of magic",
        "explanation_of_abstract": "is a category or school of supernatural power in a fictional setting.",
        "instances": ["Arcane", "Divine", "Primal"],
    },
    {
        "id": 46,
        "context": "The history of philosophy was shaped by a few towering figures in ancient Greece. Socrates was a foundational figure, known for his method of questioning and his assertion that the unexamined life is not worth living. His student, Plato, wrote extensively, exploring concepts like justice and reality through dialogues, and proposed the theory of Forms. The third in this intellectual lineage was Aristotle, Platos student, who made significant contributions to logic, ethics, biology, and politics. His empirical approach contrasted with Platos idealism. The works of these three thinkers laid the groundwork for Western philosophy, and their ideas continue to be debated and studied more than two thousand years later, a testament to their enduring intellectual legacy.",
        "abstract": "philosopher",
        "explanation_of_abstract": "is a person engaged or learned in philosophy, especially as an academic discipline.",
        "instances": ["Socrates", "Plato", "Aristotle"],
    },
    {
        "id": 47,
        "context": "The cyberpunk novel was set in a futuristic city controlled by massive, omnipresent corporations. OmniCorp was the largest, a conglomerate that had its hand in everything from consumer goods to private security. Their main rival was DataDyne, a company that specialized in information technology and cybernetics, pushing the boundaries of human augmentation. A smaller, more aggressive player was Tekken, a Japanese zaibatsu known for its advanced robotics and black-market weapons. The storys protagonist, a freelance hacker, found himself caught in the middle of a shadow war between these corporate giants. He had to navigate a world where information was power, and corporate loyalty was a commodity that could be bought and sold to the highest bidder.",
        "abstract": "corporation",
        "explanation_of_abstract": "is a company or group of people authorized to act as a single entity.",
        "instances": ["OmniCorp", "DataDyne", "Tekken"],
    },
    {
        "id": 48,
        "context": "The world of Middle-earth is home to many iconic characters. Frodo Baggins, a humble hobbit from the Shire, is tasked with the monumental quest of destroying the One Ring. He is guided and protected by Gandalf, a wise and powerful wizard who orchestrates the fight against the dark lord Sauron. One of their key companions is Aragorn, the heir to the throne of Gondor, who must embrace his destiny to unite the free peoples. The personal journeys of these three characters are central to the epic tale. Frodos resilience, Gandalfs wisdom, and Aragorns courage are the forces that ultimately stand against the overwhelming power of darkness, proving that even the smallest person can change the course of the future.",
        "abstract": "fictional character",
        "explanation_of_abstract": "is a person or being in a narrative work of art.",
        "instances": ["Frodo Baggins", "Gandalf", "Aragorn"],
    },
    {
        "id": 49,
        "context": "The biologist was studying symbiotic relationships in a coral reef ecosystem. She observed the classic partnership between the Clownfish and the Sea Anemone. The clownfish, immune to the anemones stinging tentacles, gains protection from predators, while the anemone benefits from the clownfish cleaning it and luring in other prey. This mutualistic relationship is a perfect example of co-evolution, a delicate dance that benefits both species. The reef was a complex web of such interactions, a bustling underwater city where survival depended on cooperation as much as competition. Every creature, from the smallest shrimp to the largest shark, played a role in the health and balance of this vibrant and fragile environment.",
        "abstract": "marine animal",
        "explanation_of_abstract": "is an animal that lives in the saltwater of the sea or ocean.",
        "instances": ["Clownfish", "Sea Anemone"],
    },
    {
        "id": 50,
        "context": "The documentary on World War II fighter planes focused on three legendary aircraft. The British Spitfire was celebrated for its elegance and agility, playing a crucial role in the Battle of Britain. Its American counterpart was the P-51 Mustang, a long-range fighter that could escort bombers all the way to Berlin and back, a key factor in winning air superiority over Europe. On the Pacific front, the Japanese Zero was dominant in the early years of the war, known for its exceptional maneuverability and range. These machines were at the cutting edge of aviation technology, and the pilots who flew them became modern-day knights, dueling in the skies for the fate of their nations.",
        "abstract": "fighter plane",
        "explanation_of_abstract": "is a military aircraft designed primarily for air-to-air combat against other aircraft.",
        "instances": ["Spitfire", "P-51 Mustang", "Zero"],
    },
    # ... 50 more samples ...
    {
        "id": 51,
        "context": "The celestial observatory tracked several near-Earth objects. The most prominent was Apophis, an asteroid that caused a brief period of concern in 2004 due to a small probability of impacting Earth. Another object of interest was Halleys Comet, a famous short-period comet that is visible from Earth every 75–76 years. Its appearance has been recorded by astronomers since at least 240 BC. These celestial visitors serve as reminders of the dynamic nature of our solar system. While asteroids pose a potential threat, comets are beautiful spectacles, and both provide valuable scientific data about the early history of our cosmic neighborhood. They are time capsules from the dawn of the solar system.",
        "abstract": "celestial object",
        "explanation_of_abstract": "is a naturally occurring physical entity, association, or structure that exists in the observable universe.",
        "instances": ["Apophis", "Halleys Comet"],
    },
    {
        "id": 52,
        "context": "The role-playing game allowed players to join one of three factions. The Steel Legion was a militaristic order dedicated to law and technological advancement, believing that a strong, centralized authority was the only way to protect civilization. In opposition, the Free Spirits were a loose confederation of rebels, artists, and explorers who valued individual liberty above all else, living in nomadic caravans and hidden settlements. The third group was the Earth Wardens, a neutral faction of druids and shamans who sought to protect the natural balance of the world from the excesses of the other two. A players choice of faction would dramatically alter their storyline and the alliances they could form.",
        "abstract": "fictional faction",
        "explanation_of_abstract": "is a group of individuals within a larger entity, united by a particular common political purpose.",
        "instances": ["Steel Legion", "Free Spirits", "Earth Wardens"],
    },
    {
        "id": 53,
        "context": "The culinary school taught the five mother sauces of French cuisine, from which most other sauces can be derived. The first is Béchamel, a milk-based sauce, thickened with a white roux. The second is Velouté, a light stock-based sauce, thickened with a blond roux. Espagnole is the third, a dark brown sauce made from a dark stock, mirepoix, and a brown roux. The fourth is Hollandaise, an emulsion of egg yolk, melted butter, and an acidic element like lemon juice. The final sauce is Tomato, based primarily on cooked tomatoes. Mastering these five sauces was considered the foundation of a professional culinary education, a gateway to a world of flavor and technique.",
        "abstract": "culinary sauce",
        "explanation_of_abstract": "is a liquid, cream, or semi-solid food, served on or used in preparing other foods.",
        "instances": ["Béchamel", "Velouté", "Espagnole", "Hollandaise", "Tomato"],
        # Note: 5 instances here for variety
    },
    {
        "id": 54,
        "context": "The Norse sagas are filled with powerful artifacts. Mjolnir, the hammer of the thunder god Thor, was said to be capable of leveling mountains and would always return to its owners hand when thrown. Another significant item was Gungnir, the spear of Odin, the Allfather. This spear was enchanted to never miss its mark. Finally, there was the ship Skidbladnir, which belonged to the god Freyr. It was large enough to hold all the gods, but could be magically folded up to fit inside a small pouch. These items were not just weapons or tools; they were symbols of their owners power and divine authority, central to many of the most famous myths of the Viking Age.",
        "abstract": "mythological artifact",
        "explanation_of_abstract": "is an object of historical or cultural interest from a myth or legend.",
        "instances": ["Mjolnir", "Gungnir", "Skidbladnir"],
    },
    {
        "id": 55,
        "context": "The companys new line of processors was designed to compete at every market segment. The flagship model was the Zenith X, a high-end chip for enthusiasts and content creators, offering the best possible performance at a premium price. For the mainstream market, they offered the Aura 5, which provided a great balance of performance and value, making it ideal for everyday computing and gaming. At the budget end was the Core 3, an entry-level processor designed for basic tasks like web browsing and office applications. This tiered strategy allowed the company to cater to a wide range of customers, ensuring that there was a product available for every need and every budget in the competitive PC hardware market.",
        "abstract": "computer processor",
        "explanation_of_abstract": "is the electronic circuitry that executes instructions comprising a computer program.",
        "instances": ["Zenith X", "Aura 5", "Core 3"],
    },
    {
        "id": 56,
        "context": "The mobile operating system market is largely a duopoly. Android, developed by Google, is an open-source platform known for its flexibility and wide adoption across a vast range of devices from different manufacturers. Its main competitor is iOS, Apples closed-source operating system, which runs exclusively on Apple hardware like the iPhone. iOS is praised for its ease of use, strong security, and tightly integrated ecosystem. The competition between these two platforms has driven innovation in the mobile industry for over a decade. Each has its own dedicated fanbase and a distinct design philosophy, offering consumers a clear choice between an open, customizable system and a polished, curated experience.",
        "abstract": "operating system",
        "explanation_of_abstract": "is the software that supports a computers basic functions, such as scheduling tasks, executing applications, and controlling peripherals.",
        "instances": ["Android", "iOS"],
    },
    {
        "id": 57,
        "context": "The cryptozoologist was searching for evidence of undiscovered hominids. His primary focus was Bigfoot, also known as Sasquatch, a large, hairy, ape-like creature said to roam the forests of North America. He also had an interest in the Yeti, or Abominable Snowman, a similar creature believed to inhabit the Himalayan mountain range. While mainstream science dismisses these creatures as folklore, the cryptozoologist was convinced of their existence. He spent his life collecting plaster casts of large footprints, analyzing blurry photographs, and interviewing alleged eyewitnesses. His quest was a lonely one, driven by a deep-seated belief that the world still held great mysteries waiting to be uncovered in its most remote corners.",
        "abstract": "cryptid",
        "explanation_of_abstract": "is an animal whose existence or survival is disputed or unsubstantiated.",
        "instances": ["Bigfoot", "Yeti"],
    },
    {
        "id": 58,
        "context": "The fantasy setting had several distinct schools of magic. The College of Winterhold was a prestigious academy in the frozen north, where mages studied the arcane arts in a formal, structured environment. In the southern deserts, the Sorcerers of the Red Sands practiced a more chaotic and intuitive form of magic, their power drawn from the intense heat and ancient spirits of the dunes. These two institutions were rivals, their philosophies on the nature of magic fundamentally opposed. The College saw magic as a science to be understood and controlled, while the Sorcerers viewed it as a living force to be communed with and respected. Their conflict was one of order versus chaos.",
        "abstract": "magical institution",
        "explanation_of_abstract": "is a fictional organization or school dedicated to the study and practice of magic.",
        "instances": ["College of Winterhold", "Sorcerers of the Red Sands"],
    },
    {
        "id": 59,
        "context": "The space opera featured a galactic government known as the Terran Republic, a sprawling democracy that had maintained peace for over a thousand years. However, a secessionist movement, calling itself the Confederacy of Independent Systems, began to challenge its authority, arguing for more local autonomy and less centralized control. The conflict between these two powers plunged the galaxy into a devastating civil war. The Republic, once a beacon of hope and stability, was forced to create a massive army to defend itself. The Confederacy, initially seen as a righteous rebellion, resorted to ruthless tactics to achieve its goals. It was a tragic conflict with no clear heroes or villains.",
        "abstract": "fictional government",
        "explanation_of_abstract": "is a governing body that exists only in a work of fiction.",
        "instances": ["Terran Republic", "Confederacy of Independent Systems"],
    },
    {
        "id": 60,
        "context": "The history of aviation is marked by several groundbreaking aircraft. The Wright Flyer, built by the Wright brothers, was the first successful heavier-than-air powered aircraft, making its inaugural flight in 1903. Decades later, the Bell X-1 became the first aircraft to exceed the speed of sound in level flight, piloted by Chuck Yeager. This achievement opened the door to the supersonic age. These planes were more than just machines; they were milestones of human ingenuity. They represent the courage of the test pilots who risked their lives and the brilliance of the engineers who dared to dream of conquering the skies, pushing the boundaries of what was believed to be possible for humanity.",
        "abstract": "aircraft",
        "explanation_of_abstract": "is an airplane, helicopter, or other machine capable of flight.",
        "instances": ["Wright Flyer", "Bell X-1"],
    },
    {
        "id": 61,
        "context": "The ancient world was home to many architectural wonders. The Great Pyramid of Giza, the oldest and largest of the three pyramids in the Giza pyramid complex, is the only one of the Seven Wonders of the Ancient World still largely intact. Another marvel was the Colossus of Rhodes, a massive statue of the sun-god Helios, which was said to have straddled the harbor entrance of the city of Rhodes. It stood for only 54 years before it was destroyed by an earthquake. These structures were testaments to the ambition and engineering skill of ancient civilizations, built to honor gods and immortalize pharaohs, their scale continuing to inspire awe thousands of years later.",
        "abstract": "ancient wonder",
        "explanation_of_abstract": "is a spectacular man-made structure from classical antiquity.",
        "instances": ["Great Pyramid of Giza", "Colossus of Rhodes"],
    },
    {
        "id": 62,
        "context": "The detective novel featured a brilliant but eccentric private investigator named Sherlock Holmes, who resided at 221B Baker Street in London. His incredible powers of observation and deductive reasoning allowed him to solve crimes that baffled the police. His loyal friend and biographer, Dr. John Watson, accompanied him on his cases, serving as a narrator and a foil to Holmess genius. The stories are a masterclass in logical deduction, as Holmes pieces together seemingly insignificant clues to expose the culprit. His methods, while unorthodox, were undeniably effective, establishing him as the archetypal detective in fiction, a legacy that has endured for over a century.",
        "abstract": "fictional detective",
        "explanation_of_abstract": "is an investigator of crime who exists in a work of fiction.",
        "instances": ["Sherlock Holmes", "Dr. John Watson"],
    },
    {
        "id": 63,
        "context": "The biology textbook explained the different types of social insects. Honeybees are a classic example, living in large colonies with a single queen, male drones, and thousands of female worker bees. Each has a specific role that contributes to the survival of the hive. Another example is the Termite, which lives in massive mounds and has a caste system that includes workers, soldiers, and reproductives. These eusocial insects are fascinating subjects of study, as their colonies function almost like a single superorganism. Their complex social structures and communication methods, such as the honeybees waggle dance, demonstrate that intricate societies are not limited to vertebrates, but can be found throughout the animal kingdom.",
        "abstract": "insect",
        "explanation_of_abstract": "is a small arthropod animal that has six legs and generally one or two pairs of wings.",
        "instances": ["Honeybees", "Termite"],
    },
    {
        "id": 64,
        "context": "The fantasy kingdom of Eldoria was protected by three ancient orders of knights. The Order of the Silver Griffin were noble paladins who defended the innocent and upheld justice. The Order of the Obsidian Serpent was a secretive group of spies and assassins who protected the kingdom from the shadows, using methods that were often morally ambiguous. The third was the Order of the Azure Dragon, an elite cadre of dragon riders who served as the kingdoms air force. While their methods and philosophies differed greatly, all three orders were fiercely loyal to the crown. Their combined strength—the griffins virtue, the serpents cunning, and the dragons might—made Eldoria an unassailable power in the land.",
        "abstract": "knightly order",
        "explanation_of_abstract": "is a society of knights bound by a common code of conduct and purpose.",
        "instances": [
            "Order of the Silver Griffin",
            "Order of the Obsidian Serpent",
            "Order of the Azure Dragon",
        ],
    },
    {
        "id": 65,
        "context": "The concert featured the works of three of the most influential composers of the Classical period. The symphony opened with a piece by Mozart, a child prodigy whose music is known for its clarity, balance, and transparency. This was followed by a piano sonata from Beethoven, a transitional figure whose work bridged the Classical and Romantic eras, expressing great emotion and struggle. The final piece was a string quartet by Haydn, often called the Father of the Symphony and Father of the String Quartet for his important contributions to these forms. The evening was a journey through a pivotal moment in music history, showcasing the genius and innovation of the masters who defined the era.",
        "abstract": "composer",
        "explanation_of_abstract": "is a person who writes music, especially as a professional occupation.",
        "instances": ["Mozart", "Beethoven", "Haydn"],
    },
    {
        "id": 66,
        "context": "The tech startup was developing a new artificial intelligence assistant. Unlike existing products on the market, this one, named Aura, was designed to understand emotional cues in the users voice and respond with empathy. The goal was to create a more natural and supportive human-computer interaction. The project was ambitious, requiring breakthroughs in natural language processing and sentiment analysis. The development team believed that the future of AI was not just about processing information, but about understanding human emotion. If successful, Aura could revolutionize everything from customer service to mental healthcare, creating a more compassionate and personalized digital world for everyone to enjoy and benefit from in their daily lives.",
        "abstract": "AI assistant",
        "explanation_of_abstract": "is a software agent that can perform tasks or services for an individual based on commands or questions.",
        "instances": ["Aura"],
    },
    {
        "id": 67,
        "context": "The comic book universe was populated by countless superheroes. Superman, the last son of Krypton, is arguably the most famous, a symbol of hope and truth with incredible powers. In stark contrast is Batman, a mortal man who uses his intellect, physical prowess, and advanced technology to fight crime in the dark city of Gotham. A third iconic hero is Wonder Woman, an Amazonian princess and demigoddess, who serves as an ambassador for peace and justice. These three characters form the trinity of their universe, representing different aspects of the heroic ideal. Superman is the powerful protector, Batman is the determined guardian, and Wonder Woman is the compassionate warrior, each inspiring millions of readers.",
        "abstract": "superhero",
        "explanation_of_abstract": "is a benevolent fictional character with superhuman powers.",
        "instances": ["Superman", "Batman", "Wonder Woman"],
    },
    {
        "id": 68,
        "context": "The geology class was learning about different types of volcanoes. Shield volcanoes, like Mauna Loa in Hawaii, are broad with gently sloping sides, formed by the eruption of fluid, low-viscosity lava. Stratovolcanoes, such as Mount Fuji in Japan, are tall, conical volcanoes built up by many layers of hardened lava, tephra, and volcanic ash. They are characterized by explosive eruptions. Finally, Cinder cones, like Parícutin in Mexico, are the simplest type, built from ejected lava fragments that solidify and fall as cinders around a single vent. Understanding these classifications helps geologists predict the style of eruption and the potential hazards associated with a particular volcano, which is crucial for protecting nearby communities.",
        "abstract": "type of volcano",
        "explanation_of_abstract": "is a classification of a volcano based on its shape and eruption style.",
        "instances": ["Shield volcanoes", "Stratovolcanoes", "Cinder cones"],
    },
    {
        "id": 69,
        "context": "The medieval fantasy story centered on a powerful magical sword named Soulfire. The blade was forged in a dragons breath and was destined to be wielded by the true king. The hero of the story, a young farm boy, discovers the sword and embarks on a quest to reclaim his throne from a usurping tyrant. The sword itself seems to have a will of its own, glowing with a soft blue light in the presence of evil and guiding the heros hand in battle. It is more than a weapon; it is a companion and a symbol of his birthright. The legend of Soulfire was known throughout the land, a beacon of hope for a people living under oppression.",
        "abstract": "magical sword",
        "explanation_of_abstract": "is a sword that has supernatural properties.",
        "instances": ["Soulfire"],
    },
    {
        "id": 70,
        "context": "The anime series followed the adventures of a group of monster trainers. The protagonists main partner was Pikachu, a small, yellow, mouse-like creature that can generate powerful electrical shocks. His rival, on the other hand, had a Charizard, a large, fire-breathing, dragon-like being that was incredibly powerful but often disobedient. These creatures, known collectively as Pokémon, could be captured, trained, and used to battle against each other in friendly competitions. The shows core message was about friendship, perseverance, and the bond between humans and their Pokémon partners. The goal was not just to win battles, but to understand and respect the creatures that shared their world, a theme that resonated with millions of fans.",
        "abstract": "fictional creature",
        "explanation_of_abstract": "is an animal or being that exists only in a work of fiction.",
        "instances": ["Pikachu", "Charizard", "Pokémon"],
    },
    {
        "id": 71,
        "context": "The financial news was dominated by the meteoric rise of several cryptocurrencies. Bitcoin, the original and most well-known, was created in 2009 and is praised for its decentralized nature. Ethereum, the second-largest, introduced the concept of smart contracts, allowing for decentralized applications to be built on its blockchain. A third notable example is Dogecoin, which started as a joke based on a popular internet meme but gained a massive following and significant market value. The world of digital currency is volatile and controversial, with proponents hailing it as the future of finance and detractors warning of a speculative bubble. It remains a fascinating and rapidly evolving technological and social experiment.",
        "abstract": "cryptocurrency",
        "explanation_of_abstract": "is a digital currency in which transactions are verified and records maintained by a decentralized system using cryptography.",
        "instances": ["Bitcoin", "Ethereum", "Dogecoin"],
    },
    {
        "id": 72,
        "context": "The literature class was analyzing famous dystopian novels. They started with Nineteen Eighty-Four by George Orwell, a chilling depiction of a totalitarian society under the constant surveillance of Big Brother. Next, they discussed Brave New World by Aldous Huxley, which presents a society controlled not by force, but by pleasure and conditioning. The final book was Fahrenheit 451 by Ray Bradbury, where books are outlawed and firemen burn any that are found. Though written at different times, these novels all serve as powerful warnings about the dangers of unchecked government power, censorship, and the loss of individuality. They explore what it means to be human in a world that seeks to crush the human spirit.",
        "abstract": "dystopian novel",
        "explanation_of_abstract": "is a genre of fiction that explores social and political structures in a dark, nightmare world.",
        "instances": ["Nineteen Eighty-Four", "Brave New World", "Fahrenheit 451"],
    },
    {
        "id": 73,
        "context": "The galactic museum contained relics from several long-dead alien civilizations. The Protheans were an advanced race that was wiped out 50,000 years ago, leaving behind advanced technology that formed the basis for the galaxys current technological level. Another exhibit was dedicated to the Rachni, an insectoid species that once waged a devastating war against the galaxy before being driven to extinction. Their hive-mind consciousness and biological starships were a source of both fear and fascination. These extinct races served as cautionary tales for the current inhabitants of the galaxy. Their ruins and artifacts were a constant reminder that even the most powerful civilizations could fall, and that the silence of the stars could hide ancient threats.",
        "abstract": "alien civilization",
        "explanation_of_abstract": "is a hypothetical extraterrestrial culture.",
        "instances": ["Protheans", "Rachni"],
    },
    {
        "id": 74,
        "context": "The company was a pioneer in the field of virtual reality, with two flagship products. The Oasis VR was their high-end headset, offering the highest resolution and widest field of view on the market, aimed at hardcore gamers and professional users. For the mainstream consumer, they produced the Portal, a more affordable, standalone headset that did not require a powerful PC to operate. This two-tiered approach allowed them to capture both the enthusiast and casual markets. They believed that virtual reality was the next major computing platform, and their goal was to make it accessible to everyone. The Oasis provided the ultimate experience, while the Portal opened the door for millions to take their first step into a larger world.",
        "abstract": "virtual reality headset",
        "explanation_of_abstract": "is a head-mounted device that provides virtual reality for the wearer.",
        "instances": ["Oasis VR", "Portal"],
    },
    {
        "id": 75,
        "context": "The mythology of ancient Egypt is populated by a vast pantheon of gods. Ra was the sun god, one of the most important deities, who was believed to rule in all parts of the created world: the sky, the Earth, and the underworld. Another key figure was Anubis, the god associated with mummification and the afterlife, who was depicted with the head of a jackal. He guided souls into the afterlife and weighed their hearts. These gods were central to Egyptian life, influencing everything from agriculture to funerary practices. The Egyptians built massive temples to honor them and developed complex rituals to ensure their favor, believing that the stability of the cosmos depended on it.",
        "abstract": "Egyptian god",
        "explanation_of_abstract": "is a deity from the pantheon of ancient Egyptian mythology.",
        "instances": ["Ra", "Anubis"],
    },
    {
        "id": 76,
        "context": "The special operations unit had several codenames for their ongoing missions. Operation Nightfall was a covert plan to extract a high-value asset from a hostile city. Operation Sandstorm was a long-term counter-insurgency campaign in a desert region, focused on winning the hearts and minds of the local population. The most critical mission was Operation Thunderbolt, a high-risk raid on an enemy weapons facility. The success or failure of these operations would have significant geopolitical consequences. The soldiers who carried them out were highly trained professionals, operating in extreme conditions with limited support. They were the unseen hand of foreign policy, their actions shaping events from the shadows, far from the public eye and media scrutiny.",
        "abstract": "military operation",
        "explanation_of_abstract": "is a coordinated military action of a state in response to a developing situation.",
        "instances": [
            "Operation Nightfall",
            "Operation Sandstorm",
            "Operation Thunderbolt",
        ],
    },
    {
        "id": 77,
        "context": "The fantasy world was home to many dangerous monsters. The Manticore was a fearsome beast with the body of a lion, the head of a man, and a tail that could shoot venomous spines. It was a cunning and cruel predator that enjoyed toying with its victims. In the swamps lived the Hydra, a multi-headed serpentine creature. For every head that was cut off, two more would grow in its place, making it nearly impossible to kill. These creatures were not mindless beasts, but intelligent and malevolent beings that posed a significant threat to any traveler foolish enough to enter their territory. They were the stuff of nightmares, legends used to scare children and warn adventurers away from the wild places of the world.",
        "abstract": "mythological monster",
        "explanation_of_abstract": "is a type of grotesque creature, whose appearance is often frightening and whose powers are destructive.",
        "instances": ["Manticore", "Hydra"],
    },
    {
        "id": 78,
        "context": "The film festival was showcasing the early works of two influential directors. The first was Alfred Hitchcock, the Master of Suspense, whose films like Psycho and The Birds terrified audiences with their psychological tension and innovative cinematography. The second director was Akira Kurosawa, a Japanese filmmaker whose samurai epics, such as Seven Samurai and Yojimbo, had a profound impact on Western cinema, particularly the Western genre. Though they worked in different cultures and genres, both were masters of their craft. They understood the power of visual storytelling and used the language of film to create unforgettable emotional experiences. Their work continues to be studied and admired by filmmakers and cinephiles around the world.",
        "abstract": "film director",
        "explanation_of_abstract": "is a person who directs the making of a film.",
        "instances": ["Alfred Hitchcock", "Akira Kurosawa"],
    },
    {
        "id": 79,
        "context": "The history of personal computing was shaped by a few iconic machines. The Apple II, introduced in 1977, was one of the first successful mass-produced microcomputers, known for its color graphics and open architecture. A few years later, the IBM PC was launched, which, thanks to its open standard, led to a market of compatible clones that dominated the industry for decades. These two machines represented different philosophies. The Apple II was a complete, user-friendly package, while the IBM PC was a versatile, modular system. Their competition spurred innovation and helped bring computing from the hobbyists garage into homes and offices around the world, igniting the digital revolution that continues to shape our modern society.",
        "abstract": "personal computer",
        "explanation_of_abstract": "is a multi-purpose computer whose size, capabilities, and price make it feasible for individual use.",
        "instances": ["Apple II", "IBM PC"],
    },
    {
        "id": 80,
        "context": "The magical university was divided into several houses, each named after a legendary founder. Gryffindor valued courage, bravery, and determination. Its members were known for their daring and chivalry. Slytherin prized ambition, cunning, and resourcefulness, with a reputation for producing powerful and influential wizards. Ravenclaw celebrated intelligence, creativity, and wisdom, attracting the most scholarly students. Finally, Hufflepuff esteemed dedication, patience, and loyalty, creating a welcoming and inclusive environment. The Sorting Hat placed new students into the house that best suited their personality. This system fostered both camaraderie within the houses and a friendly rivalry between them, which was a central part of the schools culture and identity for centuries.",
        "abstract": "fictional house",
        "explanation_of_abstract": "is a group or division within a fictional school or institution.",
        "instances": [
            "Gryffindor",
            "Slytherin",
            "Ravenclaw",
            "Hufflepuff",
        ],  # Note: 4 instances
    },
    {
        "id": 81,
        "context": "The world of professional wrestling is filled with larger-than-life characters. Hulk Hogan was a pop culture icon in the 1980s, known for his Hulkamania persona and his message to train, say your prayers, and eat your vitamins. In the late 1990s, Stone Cold Steve Austin became the face of the rebellious Attitude Era, an anti-authority figure who resonated with a more cynical audience. A contemporary of Austin was The Rock, whose charisma and catchphrases propelled him to superstardom both in the ring and later in Hollywood. These performers were masters of storytelling and audience engagement, creating compelling rivalries and moments that are still remembered by fans today. They transformed wrestling into a global entertainment phenomenon.",
        "abstract": "professional wrestler",
        "explanation_of_abstract": "is an athlete who competes in professional wrestling.",
        "instances": ["Hulk Hogan", "Stone Cold Steve Austin", "The Rock"],
    },
    {
        "id": 82,
        "context": "The musician was a master of several stringed instruments. His favorite was the guitar, which he used to compose folk songs and rock anthems. He was also proficient with the violin, a versatile instrument that he played in a local orchestra, its soaring notes perfect for classical music. For more intimate performances, he preferred the ukulele, a small, four-stringed instrument with a bright, cheerful tone that was perfect for accompanying his singing. Each instrument had its own voice and character, allowing him to express different moods and musical ideas. His ability to switch between them made him a sought-after session musician, able to contribute to a wide variety of musical genres and styles.",
        "abstract": "musical instrument",
        "explanation_of_abstract": "is an instrument created or adapted to make musical sounds.",
        "instances": ["guitar", "violin", "ukulele"],
    },
    {
        "id": 83,
        "context": "The city of Neo-Kyoto was patrolled by advanced law enforcement mechs. The Enforcer Mk V was the standard patrol unit, a heavily armored bipedal machine equipped with non-lethal crowd control weapons. For high-threat situations, the police deployed the Judicator, a larger, more powerful mech armed with heavy cannons and missiles. This formidable machine was a deterrent in itself, its presence enough to quell most riots. The combination of the versatile Enforcer and the powerful Judicator allowed the Neo-Kyoto police to maintain order in a city plagued by cybernetically enhanced gangs and corporate espionage. The pilots of these mechs were an elite force, the last line of defense between civilization and chaos in the sprawling metropolis.",
        "abstract": "mech",
        "explanation_of_abstract": "is a large armored robot, typically controlled by a pilot inside the vehicle.",
        "instances": ["Enforcer Mk V", "Judicator"],
    },
    {
        "id": 84,
        "context": "The fantasy novel described a pantheon of gods representing different aspects of life and death. The Weaver was the goddess of fate, who spun the destinies of all mortals on her cosmic loom. The Reaper was the god of death, a silent, cloaked figure who guided souls to the afterlife, not as a figure of evil, but as a necessary and peaceful end. The Jester was the god of chaos and luck, a trickster figure whose actions were unpredictable and often had unintended consequences. These three deities were not worshipped in temples, but their influence was felt by everyone. Mortals would curse the Jester for their bad luck or thank the Weaver for a fortunate turn of events.",
        "abstract": "fictional deity",
        "explanation_of_abstract": "is a god or goddess that exists only in a work of fiction.",
        "instances": ["Weaver", "Reaper", "Jester"],
    },
    {
        "id": 85,
        "context": "The art history class focused on the Italian Renaissance, highlighting two of its most famous artists. Leonardo da Vinci was the ultimate Renaissance man, a painter, sculptor, architect, musician, scientist, and inventor. His most famous work is the Mona Lisa. A younger contemporary was Michelangelo, a sculptor and painter of immense talent, known for his statue of David and the ceiling of the Sistine Chapel. Though they were rivals, their work defined the High Renaissance. Leonardos scientific curiosity and subtle realism contrasted with Michelangelos emotional intensity and idealized forms. Together, they elevated the status of the artist and produced some of the most iconic works in the history of art.",
        "abstract": "Renaissance artist",
        "explanation_of_abstract": "is an artist who worked during the Renaissance period in Europe.",
        "instances": ["Leonardo da Vinci", "Michelangelo"],
    },
    {
        "id": 86,
        "context": "The video game series Mass Effect allows players to choose a character class, each with a unique set of abilities. The Soldier is a combat specialist, proficient with all types of weapons. The Adept is the ultimate biotic, capable of lifting enemies with their mind and creating powerful singularities. The Engineer is a tech expert, able to deploy combat drones and hack robotic enemies. These classes provide different gameplay experiences. A Soldier playthrough is a fast-paced shooter, while an Adept focuses on crowd control and biotic combos. An Engineer, meanwhile, controls the battlefield through strategic use of tech powers. This variety encourages multiple playthroughs to experience all the different combat styles the game has to offer.",
        "abstract": "character class",
        "explanation_of_abstract": "is a job or profession for a character in a role-playing game.",
        "instances": ["Soldier", "Adept", "Engineer"],
    },
    {
        "id": 87,
        "context": "The ancient world was filled with stories of mythical beasts. The Griffin was a creature with the body of a lion and the head and wings of an eagle, often depicted as a guardian of treasure. The Centaur, from Greek mythology, had the upper body of a human and the lower body of a horse, representing the struggle between civilization and barbarism. These creatures captured the human imagination, combining familiar elements in unfamiliar ways. They served as symbols, monsters, and sometimes even wise counselors in the stories of heroes and gods. Their presence in art and literature across many cultures shows a universal fascination with the fantastic and the monstrous, the creatures that live at the edge of the known world.",
        "abstract": "mythical beast",
        "explanation_of_abstract": "is an animal from mythology or legend.",
        "instances": ["Griffin", "Centaur"],
    },
    {
        "id": 88,
        "context": "The new streaming service launched with several exclusive, high-profile television shows. Starfall was a sprawling science fiction epic, with stunning visual effects and a complex, multi-season storyline. For fans of fantasy, there was The Crimson Crown, a dark and gritty adaptation of a popular book series. The service also offered Echo Park, a critically acclaimed character-driven drama about a group of friends living in Los Angeles. This diverse slate of original content was designed to attract a wide range of subscribers. The company was betting that high-quality, exclusive programming was the key to competing in the crowded streaming market. The success of these initial shows would determine the future of the entire platform.",
        "abstract": "television show",
        "explanation_of_abstract": "is any content produced for viewing on a television set.",
        "instances": ["Starfall", "The Crimson Crown", "Echo Park"],
    },
    {
        "id": 89,
        "context": "The digital artist used several software tools to create her illustrations. For sketching and painting, she primarily used Procreate, an intuitive and powerful application on her iPad. When she needed to create scalable vector graphics for logos or icons, she switched to Adobe Illustrator on her desktop computer. This program offered precise control over lines and shapes. The ability to seamlessly move her work between these two applications was crucial to her workflow. Procreate offered creative freedom and portability, while Illustrator provided the technical precision required for professional design work. Together, they gave her a complete digital studio, allowing her to bring any idea to life with professional-quality results.",
        "abstract": "software application",
        "explanation_of_abstract": "is a computer program designed to help people perform an activity.",
        "instances": ["Procreate", "Adobe Illustrator"],
    },
    {
        "id": 90,
        "context": "The martial arts film featured a protagonist who was a master of Wing Chun, a close-range combat system that emphasizes structure and simultaneous blocking and striking. The main antagonist, in contrast, was a practitioner of Taekwondo, a Korean martial art known for its powerful and acrobatic kicks. The fight scenes between them were a spectacular display of contrasting styles. The Wing Chun master stayed grounded, using efficiency of motion to deflect and counter, while the Taekwondo expert used his dynamic kicks to attack from a distance. The film was a celebration of martial arts philosophy, showing how different disciplines approach the same problem of combat with their own unique principles and techniques.",
        "abstract": "martial art",
        "explanation_of_abstract": "is a codified system and tradition of combat practices.",
        "instances": ["Wing Chun", "Taekwondo"],
    },
    {
        "id": 91,
        "context": "The colonization of the solar system was led by three powerful megacorporations. The Weyland-Yutani Corporation specialized in terraforming and xenotechnology, often cutting corners on safety to maximize profits. The Tyrell Corporation was the leader in android manufacturing, creating artificial humans that were more human than human. The third major player was the Cyberdyne Systems Corporation, a defense contractor that developed advanced AI and autonomous weapons systems. These companies were often in fierce competition, their corporate wars fought in boardrooms, through industrial espionage, and sometimes in open conflict on distant off-world colonies. They were the new governments, their power and influence eclipsing that of any nation-state from Old Earth, a new era of corporate feudalism.",
        "abstract": "fictional corporation",
        "explanation_of_abstract": "is a company or business that exists only in a work of fiction.",
        "instances": [
            "Weyland-Yutani Corporation",
            "Tyrell Corporation",
            "Cyberdyne Systems Corporation",
        ],
    },
    {
        "id": 92,
        "context": "The fantasy worlds creation myth involved three primordial dragons. Ignis, the dragon of fire, created the sun and the stars. Terra, the dragon of earth, shaped the mountains and the continents. Aqua, the dragon of water, filled the oceans and the rivers. These three beings worked in harmony to forge the world from the chaos of the void. After their work was done, they fell into a deep slumber, their immense bodies becoming part of the landscape itself. Legends say they will only awaken at the end of time to unmake the world they created. The volcanoes are said to be the slow breathing of Ignis, a reminder of the immense power sleeping just beneath the surface.",
        "abstract": "primordial being",
        "explanation_of_abstract": "is a powerful entity that has existed since the beginning of time in a fictional universe.",
        "instances": ["Ignis", "Terra", "Aqua"],
    },
    {
        "id": 93,
        "context": "The browser wars of the late 1990s and early 2000s were a fierce competition for market share. Netscape Navigator was the early dominant player, but it was eventually overtaken by Microsoft's Internet Explorer, which was bundled with the Windows operating system. This led to a long period of IE dominance, which some argue led to stagnation in web standards. The landscape changed again with the release of Firefox, an open-source browser from the Mozilla Foundation, which gained popularity for its features and security. This renewed competition spurred innovation and ultimately led to the feature-rich, standards-compliant web browsers that we use today, a testament to the power of competition in the tech industry.",
        "abstract": "web browser",
        "explanation_of_abstract": "is a software application for accessing information on the World Wide Web.",
        "instances": ["Netscape Navigator", "Internet Explorer", "Firefox"],
    },
    {
        "id": 94,
        "context": "The role-playing game was set in the city of Waterdeep, a bustling port metropolis known as the City of Splendors. It is a hub of trade, political intrigue, and adventure. Not far from there lies the city of Baldurs Gate, a place with a darker reputation, known for its high crime rate and the shadowy organizations that vie for control of its underworld. Both cities are major locations in the Forgotten Realms campaign setting. Waterdeep is a place of opportunity and high society, while Baldurs Gate is a city of grit and survival. Adventurers traveling between them would experience a stark contrast in culture, law, and the dangers they might face on the streets.",
        "abstract": "fictional city",
        "explanation_of_abstract": "is a city that exists only in a work of fiction.",
        "instances": ["Waterdeep", "Baldurs Gate"],
    },
    {
        "id": 95,
        "context": "The secret agent, codenamed 007, was equipped with a signature handgun, the Walther PPK. It was a small, concealable pistol that had been his reliable sidearm for many missions. In the latest film, however, he was issued a new, more modern weapon, the Heckler & Koch VP9, a polymer-framed, striker-fired pistol with a higher magazine capacity. The change was symbolic of the characters evolution, moving from classic elegance to modern efficiency. While the Walther PPK represented the tradition and history of the character, the VP9 signified a shift towards a more realistic and tactical approach to espionage in the 21st century, a nod to the changing nature of modern warfare.",
        "abstract": "handgun",
        "explanation_of_abstract": "is a gun designed to be held in one hand.",
        "instances": ["Walther PPK", "Heckler & Koch VP9"],
    },
    {
        "id": 96,
        "context": "The story followed a young sorceress who was an apprentice to a powerful archmage named Elminster Aumar. He was a legendary figure, known for his wisdom and his role in many historical events. The sorceresss rival was another young mage, a prodigious but arrogant student of the famous Mordenkainen, an archmage known for his pragmatism and the powerful spells that bore his name. The two masters had a complex relationship of respect and rivalry, which was mirrored in their students. The story was as much about the clash of their magical philosophies as it was about the adventures of the young protagonists, exploring themes of mentorship, ambition, and the responsibility that comes with great power.",
        "abstract": "archmage",
        "explanation_of_abstract": "is a fictional term for a powerful and influential wizard or sorcerer.",
        "instances": ["Elminster Aumar", "Mordenkainen"],
    },
    {
        "id": 97,
        "context": "The game developer was creating a new real-time strategy game. It featured three unique factions. The Global Defense Initiative was a futuristic version of the UN military, relying on advanced conventional warfare and powerful tanks. Their enemy was the Brotherhood of Nod, a mysterious techno-religious cult that utilized stealth, guerilla tactics, and Tiberium-based weaponry. The third faction, the Scrin, was an alien race with powerful air units and biomechanical war machines. The asymmetrical design of these factions meant that each required a completely different strategy to play effectively. The GDI was straightforward, Nod required cunning, and the Scrin demanded mastery of air power, providing a deep and varied gameplay experience for players.",
        "abstract": "fictional faction",
        "explanation_of_abstract": "is a group of individuals within a larger entity, united by a particular common political purpose.",
        "instances": ["Global Defense Initiative", "Brotherhood of Nod", "Scrin"],
    },
    {
        "id": 98,
        "context": "The literature course examined the works of two major figures of the Lost Generation. Ernest Hemingway was known for his spare, direct prose style and his themes of war, loss, and masculinity, as seen in novels like A Farewell to Arms. In contrast, F. Scott Fitzgerald captured the glamour and disillusionment of the Jazz Age, with a more lyrical and evocative style, most famously in The Great Gatsby. Both authors lived as expatriates in Paris in the 1920s and their work reflects the sense of alienation and moral confusion that followed World War I. They were friends and rivals, their lives and work intertwined, providing a fascinating snapshot of a pivotal moment in American literary history.",
        "abstract": "author",
        "explanation_of_abstract": "is a writer of a book, article, or report.",
        "instances": ["Ernest Hemingway", "F. Scott Fitzgerald"],
    },
    {
        "id": 99,
        "context": "The conspiracy theorist was obsessed with secret societies that he believed controlled the world. He spoke at length about the Illuminati, a supposed cabal of global elites seeking to establish a New World Order. He also claimed to have evidence of the Freemasons, a fraternal organization that he believed was manipulating world events from behind the scenes. His apartment was covered in charts and newspaper clippings, connecting seemingly unrelated events into a grand, overarching conspiracy. While most people dismissed him as a crank, his theories found a receptive audience online. He was a product of an age of information overload, where discerning fact from fiction had become increasingly difficult for many.",
        "abstract": "secret society",
        "explanation_of_abstract": "is a club or an organization whose activities, events, and inner functioning are concealed from non-members.",
        "instances": ["Illuminati", "Freemasons"],
    },
    {
        "id": 100,
        "context": "The new racing game featured a roster of hypercars. The Bugatti Chiron was one of the stars, a marvel of engineering with a top speed of over 300 mph. Another featured vehicle was the Koenigsegg Jesko, a Swedish hypercar designed for extreme downforce and track performance. The games physics engine was incredibly realistic, allowing players to feel the difference between these two machines. The Chiron was a stable, luxurious speed missile, while the Jesko was a raw, aggressive track weapon. The game celebrated the pinnacle of automotive achievement, giving players the chance to experience cars that few will ever see in person, let alone drive at their full potential.",
        "abstract": "hypercar",
        "explanation_of_abstract": "is a high-performance supercar, of which only a very small number are made.",
        "instances": ["Bugatti Chiron", "Koenigsegg Jesko"],
    },
]


def matches(pred, gold_list):
    """Return True if pred matches any gold instance according to rules."""
    pred_lower = pred.lower()
    for gold in gold_list:
        gold_lower = gold.lower()
        if pred_lower == gold_lower:
            return True
        if pred_lower in gold_lower:
            return True
        if gold_lower in pred_lower:
            return True
    return False


def evaluate_extractor(extractor, test_cases):
    """
    Evaluate the extractor on given test cases.

    Returns:
        case_results: list of dicts with per-case metrics
        overall: dict with overall metrics
    """
    case_results = []

    total_TP = total_FP = total_FN = 0

    for test in test_cases:
        gold_instances = test["instances"]
        # run extractor
        predictions = [
            span
            for span, score, start, end in extractor.extract(
                test["context"],
                test["abstract"],
                explanation_of_abstract=test.get("explanation_of_abstract"),
            )
        ]

        matched_gold = set()
        matched_pred = set()
        if len(gold_instances) == 0 and len(predictions) == 0:
            TP = 1
            FP = 0
            FN = 0
        else:
            for i, pred in enumerate(predictions):
                if matches(pred, gold_instances):
                    matched_pred.add(i)
                    for j, gold in enumerate(gold_instances):
                        if matches(pred, [gold]):
                            matched_gold.add(j)

            TP = len(matched_pred)
            FP = len(predictions) - TP
            FN = len(gold_instances) - len(matched_gold)

        total_TP += TP
        total_FP += FP
        total_FN += FN

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        case_results.append(
            {
                "id": test["id"],
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    overall_precision = (
        total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    )
    overall_recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision + overall_recall > 0
        else 0
    )

    overall = {
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
    }

    return case_results, overall


def plot_performance(case_results, overall, model_name):
    """Generate plots for per-case metrics and overall confusion counts."""
    ids = [c["id"] for c in case_results]
    precisions = [c["precision"] for c in case_results]
    recalls = [c["recall"] for c in case_results]
    f1s = [c["f1"] for c in case_results]

    # Per-case line plot
    plt.figure(figsize=(10, 6))
    plt.plot(ids, precisions, marker="o", label="Precision")
    plt.plot(ids, recalls, marker="o", label="Recall")
    plt.plot(ids, f1s, marker="o", label="F1-score")
    plt.xlabel("Test Case ID")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(f"Per-Test-Case Performance {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"./roast_bench/Per-Test-Case Performance {model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Overall TP/FP/FN bar chart
    plt.figure(figsize=(6, 5))
    plt.bar(
        ["TP", "FP", "FN"],
        [overall["TP"], overall["FP"], overall["FN"]],
        color=["green", "red", "orange"],
    )
    plt.title(f"Overall Instance-Level Confusion Counts {model_name}")
    plt.ylabel("Count")
    plt.savefig(
        f"./roast_bench/Overall Instance-Level Confusion Counts {model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    MODEL_OPTS = [
        ("QA_RoBERTA_SQUADv2", "nn"),
        ("QA_XLM_RoBERTA", "nn"),
        ("gemma3:1b", "llm"),
        ("llama3.2", "llm"),
        ("phi4-mini", "llm"),
        ("gemma3", "llm"),
        ("gemma3:12b", "llm"),
    ]

    # MODEL_NAME = "QA_RoBERTA_SQUADv2", # score_threshold=0.3
    # MODEL_NAME = "QA_XLM_RoBERTA" # score_threshold=0.3
    # MODEL_NAME = "gemma3:1b"   # 1   B
    # MODEL_NAME = "llama3.2"    # 3.2 B
    # MODEL_NAME = "phi4-mini"   # 3.8 B
    # MODEL_NAME = "gemma3"      # 4.3 B
    # MODEL_NAME = "gemma3:12b"  # 12  B

    for MODEL_NAME, DRIVER in MODEL_OPTS:
        print(f"\nInitializing expert extractor with model: {MODEL_NAME}...")
        try:
            extractor = None
            if DRIVER == "nn":
                extractor = ExpertInstanceExtractor(
                    model_name_or_path=MODEL_NAME, score_threshold=0.3
                )
            elif DRIVER == "llm":
                extractor = ExpertInstanceGenerator(model_name=MODEL_NAME)
            start_time = time.time()
            case_results, overall = evaluate_extractor(extractor, TEST_CASES)
            end_time = time.time()

            print("Overall Precision:", round(overall["precision"], 3))
            print("Overall Recall:", round(overall["recall"], 3))
            print("Overall F1:", round(overall["f1"], 3))
            print("total Time (sec):", round(end_time - start_time))

            plot_performance(case_results, overall, MODEL_NAME)
            if DRIVER == "nn":
                del extractor.reader
                del extractor
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)

        except Exception as main_exception:
            logging.error(
                f"An error occurred during execution: {main_exception}", exc_info=True
            )
