import logging
import math
import string
from typing import List, Tuple, Dict, Any, Optional
import ollama
import json
from haystack import component, Document, Answer
from haystack.components.readers import ExtractiveReader
from abc import ABC, abstractmethod

class ROASTDriver(ABC):
    """
    Abstract Base Class defining the standard interface for a ROAST-style instance extractor.

    This class establishes a contract for any implementation that aims to find specific
    instances of an abstract concept within a given text. It ensures that different
    approaches (e.g., extractive vs. generative) can be used interchangeably
    by downstream components that rely on this common interface.
    """

    @staticmethod
    def get_extractor(driver_type : str, model_name_or_path : str, score_threshold : float):
        if driver_type == 'llm':
            return ExpertInstanceGenerator(model_name_or_path)
        elif driver_type == 'nn':
            return ExpertInstanceExtractor(model_name_or_path,score_threshold=score_threshold)
        else:
            raise ValueError(f"Invalid driver type specified: {driver_type}. Choose 'nn' or 'llm'.")


    @abstractmethod
    def extract(self, context: str, abstract_concept: str, explanation_of_abstract: Optional[str] = None) -> \
            List[Tuple[str, float, int, int]]:
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


@component
class QuestionGenerator:
    """
    Generates a set of varied, natural-language questions to guide the
    ExtractiveReader, improving its ability to find diverse instances.
    """

    @component.output_types(questions=List[str])
    def run(self, abstract_concept: str, explanation: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates questions based on the abstract concept.
        :param abstract_concept: The concept to find instances of.
        :param explanation: An optional explanation of the abstract concept.
        :return: A dictionary containing a list of questions.
        """
        if not isinstance(abstract_concept, str) or not abstract_concept:
            logging.warning("QuestionGenerator received an invalid abstract_concept.")
            return {"questions": []}

        # Generate a plural form for more natural-sounding questions.
        if abstract_concept.endswith('y') and len(abstract_concept) > 1 and abstract_concept[-2] not in "aeiou":
            plural_concept = f"{abstract_concept[:-1]}ies"
        elif abstract_concept.endswith(('s', 'x', 'z', 'ch', 'sh')):
            plural_concept = f"{abstract_concept}es"
        else:
            plural_concept = f"{abstract_concept}s"

        # Create a context-setting prefix if an explanation is provided.
        prefix = ""
        if explanation:
            prefix = f"A {abstract_concept} {explanation}. "

        # A set of diverse questions to improve the model's recall.
        questions = [
            f"{prefix}. Which instances of {abstract_concept} are mentioned in the text?",
            f"{prefix}. Which {plural_concept} are described in the passage?",
            # f"{prefix}. What specific {plural_concept} are listed in the document?",
            # f"{prefix}. Identify the names of the {plural_concept} in the text.",
        ]
        return {"questions": questions}

@component
class AnswerFilter:
    """
    Filters and refines a list of Haystack Answers using score normalization
    and span containment to find the best candidate answers.
    """

    @staticmethod
    def _is_overlapping(answer1: Answer, answer2: Answer) -> bool:
        """
        Checks if one answer's span is fully contained within the other.
        This is a static method as its logic does not depend on the state
        of an AnswerFilter instance.
        """
        # Ensure both answers have valid document_offset attributes to compare.
        if not all(hasattr(ans, 'document_offset') and ans.document_offset for ans in [answer1, answer2]):
            return False
        start1, end1 = answer1.document_offset.start, answer1.document_offset.end
        start2, end2 = answer2.document_offset.start, answer2.document_offset.end

        return (start1 >= start2 and end1 <= end2) or (start2 >= start1 and end2 <= end1)

    @component.output_types(filtered_answers=List[Answer])
    def run(self, answers: List[Answer]) -> Dict[str, Any]:
        """
        Filters answers using score normalization and span containment.
        :param answers: A list of Answer objects from the reader.
        :return: A dictionary containing the filtered list of answers.
        """
        # Calculate a normalized score to penalize overly long answers.
        for ans_item in answers:
            if ans_item.data:
                # Add a small constant to length to avoid division by zero or log(1) issues.
                ans_item.meta['normalized_score'] = ans_item.score / math.log(len(ans_item.data) + 1.1)
            else:
                ans_item.meta['normalized_score'] = 0

        # Sort by the new normalized score to prioritize concise, high-confidence answers.
        sorted_answers = sorted(answers, key=lambda x: x.meta.get('normalized_score', 0), reverse=True)

        # Filter out overlapping answers, keeping the one with the higher normalized score.
        final_answers: List[Answer] = []
        for candidate_answer in sorted_answers:
            if candidate_answer.data is None:
                continue
            if not any(self._is_overlapping(candidate_answer, kept_answer) for kept_answer in final_answers):
                final_answers.append(candidate_answer)
        return {"filtered_answers": final_answers}

class ExpertInstanceExtractor(ROASTDriver):
    """
    Orchestrates Haystack components and applies advanced heuristic filtering
    to perform highly accurate instance extraction.
    """

    def __init__(
            self,
            model_name_or_path: str,
            device: Optional[str] = None,
            reader_top_k: int = 20,
            score_threshold: float = 0.0,
    ):
        self.q_gen = QuestionGenerator()
        self.reader = ExtractiveReader(model=model_name_or_path, device=device, top_k=reader_top_k, no_answer=True)
        self.filter = AnswerFilter()
        self.reader.warm_up()
        self.score_threshold = score_threshold

        # An expanded set of "function words" to better identify descriptive phrases.
        self.FUNCTION_WORDS = {
            'a', 'an', 'the', 'in', 'on', 'of', 'for', 'to', 'with', 'by', 'at', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can', 'could',
            'may', 'might', 'must', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'some', 'any', 'all', 'several', 'many', 'few', 'other', 'another', 'various', 'its', 'their', 'my',
            'your', 'his', 'her', 'first', 'second', 'third', 'last', 'next', 'former', 'latter', 'main',
            'largest', 'smallest', 'older', 'newer', 'red', 'green', 'blue', 'named', 'called', 'known',
            'described', 'including', 'such', 'as', 'and', 'or', 'but', 'performance-critical', 'sections'
        }
        logging.info("ExpertInstanceExtractor components are initialized and ready.")

    def _is_valid_instance(self, span_to_check: str, abstract_concept: str) -> bool:
        """
        A final, intelligent validation gate to ensure the answer is a clean entity.
        :param span_to_check: The cleaned answer string.
        :param abstract_concept: The original concept being searched for.
        :return: True if the span is a valid instance, False otherwise.
        """
        words = span_to_check.lower().split()

        # Rule 1: Must not be excessively long (e.g., more than 3 words).
        if len(words) > 3:
            return False

        # Rule 2: Must not be an "echo" of the abstract concept.
        abstract_words = set(abstract_concept.lower().split())
        if abstract_words.issubset(set(words)):
            return False

        # Rule 3: Must contain at least one "substantive" word.
        if not any(word not in self.FUNCTION_WORDS for word in words):
            return False

        return True

    def extract(self, context: str, abstract_concept: str, explanation_of_abstract: Optional[str] = None) -> List[
        Tuple[str, float, int, int]]:
        """
        Runs the full extraction and filtering pipeline.
        :param context: The text to search within.
        :param abstract_concept: The abstract concept to find instances of.
        :param explanation_of_abstract: An optional explanation of the abstract concept.
        :return: A list of tuples, each containing (instance, score, start_offset, end_offset).
        """
        if not context or not abstract_concept:
            return []
        docs = [Document(content=context)]
        questions = self.q_gen.run(abstract_concept=abstract_concept, explanation=explanation_of_abstract)["questions"]

        # Step 1: Gather all possible answers from the reader for all questions.
        all_raw_answers: List[Answer] = []
        for query_text in questions:
            try:
                reader_result = self.reader.run(query=query_text, documents=docs)
                all_raw_answers.extend(reader_result.get("answers", []))
            except Exception as e:
                logging.error(f"Error running reader for question '{query_text}': {e}")
                continue

        # Step 2: Apply the score threshold early for efficiency.
        thresholded_answers = [ans for ans in all_raw_answers if ans.score >= self.score_threshold]

        # Step 3: Run the initial filtering component.
        filter_result = self.filter.run(answers=thresholded_answers)
        candidate_answers = filter_result["filtered_answers"]

        # Step 4: Add a 'is_proper' flag to metadata for ranking.
        for ans_item in candidate_answers:
            if ans_item.data and ans_item.data[0].isupper():
                ans_item.meta['is_proper'] = True
            else:
                ans_item.meta['is_proper'] = False

        # Step 5: Re-rank candidates, prioritizing proper nouns, then by original score.
        ranked_candidates = sorted(candidate_answers, key=lambda x: (x.meta.get('is_proper', False), x.score),
                                   reverse=True)

        # Step 6: Final processing loop with validation and de-duplication.
        seen_strings = set()
        results: List[Tuple[str, float, int, int]] = []
        for current_ans in ranked_candidates:
            if current_ans.data is None:
                continue

            clean_span = current_ans.data.strip(string.punctuation + string.whitespace)

            if self._is_valid_instance(clean_span, abstract_concept):
                if clean_span and clean_span.lower() not in seen_strings:
                    # Get start and end from the .document_offset attribute, handling None.
                    start_pos = current_ans.document_offset.start if current_ans.document_offset else -1
                    end_pos = current_ans.document_offset.end if current_ans.document_offset else -1
                    results.append((clean_span, round(current_ans.score, 4), start_pos, end_pos))
                    seen_strings.add(clean_span.lower())

        return results

class ExpertInstanceGenerator(ROASTDriver):
    """
    Orchestrates a generative model via Ollama to perform instance extraction,
    mimicking the functionality of the original ROAST but with a generative driver.
    """

    def __init__(self, model_name: str = 'llama3', max_retries: int = 3):
        """
        Initializes the generator.
        :param model_name: The name of the Ollama model to use (e.g., 'llama3', 'mistral').
        :param max_retries: The number of times to retry on a parsing failure.
        """
        self.model_name = model_name
        self.client = ollama.Client()
        self.max_retries = max_retries
        logging.info(f"ExpertInstanceGenerator initialized with Ollama model: {self.model_name}")

    def _build_prompt(self, context: str, abstract_concept: str, explanation: Optional[str]) -> str:
        """
        Builds the complete few-shot prompt for the generative model.
        """
        ROAST_PROMPT = """
# System:
**Role**: You are an instance detector of a Special abstract.

**Output**: the output must be a list of instances you found. if instance is not found, return empty list [].

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

- "context": "At the zoo, the keeper introduced two animals: Zuri, the clever chimpanzee, and Bobo, a lazy orangutan who preferred sleeping to climbing trees."
- "abstract": "ape"
- "explanation_of_abstract": "is a kind of intelligent primate without a tail"
- "question": "An {ape} {is a kind of intelligent primate without a tail}. Which instances of {ape} are described in the passage?"
+ "answers": ["Zuri", "Bobo"]

---
"""

        explanation_text = explanation if explanation else "a concept"
        question = f"A {{{abstract_concept}}} {{{explanation_text}}}. Which instances of {{{abstract_concept}}} are described in the passage?"

        problem_part = f"""
# problem:

- "context": "{context}"
- "abstract": "{abstract_concept}"
- "explanation_of_abstract": "{explanation_text}"
- "question": "{question}"
+ "answers" : 
"""
        return ROAST_PROMPT + problem_part

    def _find_positions(self, context: str, span: str) -> Tuple[int, int]:
        """
        Finds the start and end character positions of a span within the context.
        :param context: The original text.
        :param span: The instance string to find.
        :return: A tuple of (start, end) positions, or (-1, -1) if not found.
        """
        start = context.find(span)
        if start != -1:
            end = start + len(span)
            return start, end
        return -1, -1

    def _parse_response(self, response_text: str, context: str) -> Optional[List[Tuple[str, float, int, int]]]:
        """
        Parses the raw string response from the LLM into a list of tuples.
        Each tuple contains (instance, score, start_pos, end_pos).
        Returns None if parsing fails.
        """
        try:
            clean_text = response_text.strip().replace("`", "")
            if clean_text.startswith("json"):
                clean_text = clean_text[4:].strip()

            parsed_list = json.loads(clean_text)
            if isinstance(parsed_list, list):
                results = []
                for item in parsed_list:
                    instance_str = str(item)
                    start, end = self._find_positions(context, instance_str)
                    if start == -1 or end == -1:
                        continue

                    results.append((instance_str, 1.0, start, end))
                return results
            else:
                logging.warning(f"LLM output was valid JSON but not a list: {parsed_list}")
                return []
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM output as JSON: {response_text}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during response parsing: {e}")
            return None

    def extract(self, context: str, abstract_concept: str, explanation_of_abstract: Optional[str] = None) -> List[Tuple[str, float, int, int]]:
        """
        Runs the full extraction pipeline using the generative model with retries.
        :param context: The text to search within.
        :param abstract_concept: The abstract concept to find instances of.
        :param explanation_of_abstract: An optional explanation of the abstract concept.
        :return: A list of tuples containing (instance, score, start_pos, end_pos).
        """
        if not context or not abstract_concept:
            return []

        prompt = self._build_prompt(context, abstract_concept, explanation_of_abstract)

        for attempt in range(self.max_retries):
            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=False,
                    options={"temperature": 0.0}
                )
                raw_response_text = response.get('response', '')
                parsed_results = self._parse_response(raw_response_text, context)

                # If parsing was successful (i.e., not None), we can return the results.
                if parsed_results is not None:
                    return parsed_results
                else:
                    logging.warning(f"Parsing failed on attempt {attempt + 1}/{self.max_retries}. Retrying...")

            except Exception as e:
                logging.error(f"Ollama call failed on attempt {attempt + 1}/{self.max_retries}: {e}")

        logging.error(f"Failed to get a valid response from Ollama after {self.max_retries} attempts.")
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # MODEL_NAME = "QA_RoBERTA_SQUADv2"
    # MODEL_NAME = "deepset/xlm-roberta-large-squad2"
    # MODEL_NAME = "QA_XLM_RoBERTA"
    MODEL_NAME = "phi4-mini"
    print(f"\nInitializing expert extractor with model: {MODEL_NAME}...")
    try:

        # extractor = ExpertInstanceExtractor(
        #     model_name_or_path=MODEL_NAME,
        #     score_threshold=0.0
        # )

        extractor = ExpertInstanceGenerator(model_name=MODEL_NAME)
        test_cases = [

            {"id": 1,
             "context": "In the ancient valley, a creature with obsidian scales and burning eyes guarded the gate. The villagers called it Narthul. Another dragon, a smaller but faster one, patrolled the skies. This second dragon was known as Ignis. Unlike Narthul, Ignis had shimmering silver scales.",
             "abstract": "dragon", "explanation_of_abstract": "is a flying special and fire breath creature"},

            {"id": 2,
             "context": "The wizard mumbled an incantation under his breath, causing the torch to levitate. He later revealed the spell was called Ignis Volare. Another technique he favored, known only to a few, allowed him to vanish instantly in a puff of green smoke.",
             "abstract": "magic spell",
             "explanation_of_abstract": "is a mystical incantation that produces supernatural effects"},

            {"id": 3,
             "context": "She had lived in many places: the fog-choked alleys of Mirehaven, a floating city in the clouds called Aerith, and even a simple village nestled among whispering trees.",
             "abstract": "city",
             "explanation_of_abstract": "is a large and permanent human settlement with infrastructure and governance"},

            {"id": 4,
             "context": "In the sacred chamber, three potions stood on a pedestal. One shimmered with a rainbow hue and smelled of ozone. The second, Vitae Essence, promised eternal youth. The last was simply labeled Void.",
             "abstract": "potion",
             "explanation_of_abstract": "is a magical or alchemical liquid with special effects when consumed"},

            {"id": 5,
             "context": "Among the travelers was an elf named Lirael, a quiet ranger with sharp eyes and a bow carved from moonwood. Another was a tall, stoic figure described only as a forest-dweller, whose footsteps left no trace.",
             "abstract": "elf",
             "explanation_of_abstract": "is a mythical humanoid creature known for agility, long life, and connection to nature"},

            {"id": 6,
             "context": "The system uses several programming languages, including Python for scripting and C++ for performance-critical sections. We also experimented with Rust.",
             "abstract": "programming language",
             "explanation_of_abstract": "is a formal language used to write software and applications. C++ or C or PHP are some examples"},

            {"id": 7,
             "context": "The starship, named the 'Odyssey', jumped to hyperspace. Its sister ship, the 'Venture', followed close behind. A third, older vessel also made the journey.",
             "abstract": "starship",
             "explanation_of_abstract": "is a spacecraft designed for interstellar or interplanetary travel"},

            {"id": 8,
             "context": "Our solar system contains many planets. The largest is Jupiter, a gas giant. Mars, the red planet, is our neighbor.",
             "abstract": "planet",
             "explanation_of_abstract": "is a large celestial body that orbits a star and does not produce its own light"},

            {"id": 9,
             "context": "The company's flagship products are the Alpha-7 camera and the newer, more compact Beta-9. An unreleased prototype, the Gamma-1, is also in development.",
             "abstract": "product",
             "explanation_of_abstract": "is a manufactured item or good designed for sale or consumer use"},

            {"id": 10,
             "context": "He studied various philosophical concepts. His main focus was on Stoicism, but he also wrote papers on Existentialism and the idea of Absurdism.",
             "abstract": "philosophical concept",
             "explanation_of_abstract": "is an abstract idea or theory related to fundamental questions about existence, values, or reason"},

            {"id": 11, "context": "Queen Denis is flying with her balck dragon, the Drago!", "abstract": "dragon",
             "explanation_of_abstract": "is a flying special and fire breath creature"}

        ]

        for test in test_cases:
            print(f"\n--- Running Test Case {test['id']} ---")
            print(f"Abstract: {test['abstract']}")

            explanation = test.get("explanation_of_abstract")
            instances = extractor.extract(test['context'], test['abstract'], explanation_of_abstract=explanation)
            print(f"\n--- Found Instances (Test {test['id']}) ---")
            if instances:
                for span, score, start, end in instances:
                    print(f"- {span!r} (score: {score:.3f}) [pos: {start}-{end}]")
            else:
                print("No instances found.")
    except Exception as main_exception:
        logging.error(f"An error occurred during execution: {main_exception}", exc_info=True)
        print("\nPlease ensure you have the required packages (`pip install farm-haystack haystack-ai`)")
