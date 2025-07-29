import logging
import math
import string
from typing import List, Tuple, Dict, Any, Optional
from haystack import component, Document, Answer
from haystack.components.readers import ExtractiveReader


@component
class QuestionGenerator:
    """
    Generates a set of varied, natural-language questions to guide the
    ExtractiveReader, improving its ability to find diverse instances.
    """

    @component.output_types(questions=List[str])
    def run(self, abstract_concept: str) -> Dict[str, Any]:
        if not isinstance(abstract_concept, str) or not abstract_concept:
            return {"questions": []}

        if abstract_concept.endswith('y') and len(abstract_concept) > 1 and abstract_concept[-2] not in "aeiou":
            plural_concept = f"{abstract_concept[:-1]}ies"
        elif abstract_concept.endswith(('s', 'x', 'z', 'ch', 'sh')):
            plural_concept = f"{abstract_concept}es"
        else:
            plural_concept = f"{abstract_concept}s"

        questions = [
            f"What are the instances of {abstract_concept} mentioned in the text?",
            f"Which {plural_concept} are described in the passage?",
            f"What specific {plural_concept} are listed in the document?",
            f"Identify the names of the {plural_concept} in the text.",
        ]
        return {"questions": questions}

@component
class AnswerFilter:
    """
    Filters and refines a list of Haystack Answers using score normalization
    and span containment to find the best candidate answers.
    """

    def _is_overlapping(self, answer1: Answer, answer2: Answer) -> bool:
        if not all(hasattr(answer, 'meta') and 'start' in answer.meta and 'end' in answer.meta for answer in
                   [answer1, answer2]):
            return False
        start1, end1 = answer1.meta['start'], answer1.meta['end']
        start2, end2 = answer2.meta['start'], answer2.meta['end']
        return (start1 >= start2 and end1 <= end2) or (start2 >= start1 and end2 <= end1)

    @component.output_types(filtered_answers=List[Answer])
    def run(self, answers: List[Answer]) -> Dict[str, Any]:
        for ans in answers:
            if ans.data:
                ans.meta['normalized_score'] = ans.score / math.log(len(ans.data) + 1.1)
            else:
                ans.meta['normalized_score'] = 0
        sorted_answers = sorted(answers, key=lambda ans: ans.meta.get('normalized_score', 0), reverse=True)
        final_answers: List[Answer] = []
        for candidate_answer in sorted_answers:
            if candidate_answer.data is None:
                continue
            if not any(self._is_overlapping(candidate_answer, kept_answer) for kept_answer in final_answers):
                final_answers.append(candidate_answer)
        return {"filtered_answers": final_answers}

class ExpertInstanceExtractor:
    """
    Orchestrates Haystack components and applies advanced heuristic filtering
    to perform highly accurate instance extraction.
    """

    def __init__(
            self,
            model_name_or_path: str,
            device: Optional[str] = None,
            reader_top_k: int = 20,  # Increased to get more candidates
    ):
        self.q_gen = QuestionGenerator()
        self.reader = ExtractiveReader(model=model_name_or_path, device=device, top_k=reader_top_k, no_answer=True)
        self.filter = AnswerFilter()
        self.reader.warm_up()

        self.FUNCTION_WORDS = {'a', 'an', 'the', 'in', 'on', 'of', 'for', 'to', 'with', 'by', 'at', 'is', 'are', 'was',
                               'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                               'would', 'should', 'can', 'could', 'may', 'might', 'must', 'one', 'two', 'three', 'four',
                               'five', 'six', 'seven', 'eight', 'nine', 'ten', 'some', 'any', 'all', 'several', 'many',
                               'few', 'other', 'another', 'various', 'its', 'their', 'my', 'your', 'his', 'her',
                               'first', 'second', 'third', 'last', 'next', 'former', 'latter', 'largest', 'smallest',
                               'older', 'newer', 'red', 'green', 'blue', 'performance-critical', 'sections'}
        logging.info("ExpertInstanceExtractor components are initialized and ready.")

    def _is_valid_instance(self, span: str, abstract_concept: str) -> bool:
        """
        A final, intelligent validation gate to ensure the answer is a clean entity.
        """
        words = span.lower().split()

        if len(words) > 3:
            return False

        abstract_words = set(abstract_concept.lower().split())
        if abstract_words.issubset(set(words)):
            return False

        if not any(word not in self.FUNCTION_WORDS for word in words):
            return False

        return True

    def extract(self, context: str, abstract_concept: str) -> List[Tuple[str, float]]:
        """
        Runs the full extraction and filtering pipeline.
        """
        if not context or not abstract_concept:
            return []
        docs = [Document(content=context)]
        questions = self.q_gen.run(abstract_concept=abstract_concept)["questions"]

        all_raw_answers: List[Answer] = []
        for q in questions:
            try:
                reader_result = self.reader.run(query=q, documents=docs)
                all_raw_answers.extend(reader_result.get("answers", []))
            except Exception as e:
                logging.error(f"Error running reader for question '{q}': {e}")
                continue

        filter_result = self.filter.run(answers=all_raw_answers)
        candidate_answers = filter_result["filtered_answers"]

        for ans in candidate_answers:
            if ans.data and ans.data[0].isupper():
                ans.meta['is_proper'] = True
            else:
                ans.meta['is_proper'] = False

        ranked_candidates = sorted(candidate_answers, key=lambda x: (x.meta.get('is_proper', False), x.score),
                                   reverse=True)

        seen_strings = set()
        results = []
        for ans in ranked_candidates:
            if ans.data is None:
                continue

            clean_span = ans.data.strip(string.punctuation + string.whitespace)

            if self._is_valid_instance(clean_span, abstract_concept):
                if clean_span.lower() not in seen_strings:
                    results.append((clean_span, round(ans.score, 4)))
                    seen_strings.add(clean_span.lower())

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    MODEL_NAME = "QA_RoBERTA_SQUADv2"
    print(f"\nInitializing expert extractor with model: {MODEL_NAME}...")
    try:
        extractor = ExpertInstanceExtractor(model_name_or_path=MODEL_NAME)
        test_cases = [
            {"id": 1,
             "context": "In the ancient valley, a creature with obsidian scales and burning eyes guarded the gate. The villagers called it Narthul. Another dragon, a smaller but faster one, patrolled the skies. This second dragon was known as Ignis. Unlike Narthul, Ignis had shimmering silver scales.",
             "abstract": "dragon"},
            {"id": 2,
             "context": "The wizard mumbled an incantation under his breath, causing the torch to levitate. He later revealed the spell was called Ignis Volare. Another technique he favored, known only to a few, allowed him to vanish instantly in a puff of green smoke.",
             "abstract": "magic spell"},
            {"id": 3,
             "context": "She had lived in many places: the fog-choked alleys of Mirehaven, a floating city in the clouds called Aerith, and even a simple village nestled among whispering trees.",
             "abstract": "city"},
            {"id": 4,
             "context": "In the sacred chamber, three potions stood on a pedestal. One shimmered with a rainbow hue and smelled of ozone. The second, Vitae Essence, promised eternal youth. The last was simply labeled Void.",
             "abstract": "potion"},
            {"id": 5,
             "context": "Among the travelers was an elf named Lirael, a quiet ranger with sharp eyes and a bow carved from moonwood. Another was a tall, stoic figure described only as a forest-dweller, whose footsteps left no trace.",
             "abstract": "elf"},
            {"id": 6,
             "context": "The system uses several programming languages, including Python for scripting and C++ for performance-critical sections. We also experimented with Rust.",
             "abstract": "programming language"},
            {"id": 7,
             "context": "The starship, named the 'Odyssey', jumped to hyperspace. Its sister ship, the 'Venture', followed close behind. A third, older vessel also made the journey.",
             "abstract": "starship"},
            {"id": 8,
             "context": "Our solar system contains many planets. The largest is Jupiter, a gas giant. Mars, the red planet, is our neighbor.",
             "abstract": "planet"},
            {"id": 9,
             "context": "The company's flagship products are the Alpha-7 camera and the newer, more compact Beta-9. An unreleased prototype, the Gamma-1, is also in development.",
             "abstract": "product"},
            {"id": 10,
             "context": "He studied various philosophical concepts. His main focus was on Stoicism, but he also wrote papers on Existentialism and the idea of Absurdism.",
             "abstract": "philosophical concept"},
            {"id": 11,
             "context": "Queen Denis is flying with her balck dragon, the Drago!",
             "abstract": "philosophical concept"}
        ]
        for test in test_cases:
            print(f"\n--- Running Test Case {test['id']} ---")
            print(f"Abstract: {test['abstract']}")
            instances = extractor.extract(test['context'], test['abstract'])
            print(f"\n--- Found Instances (Test {test['id']}) ---")
            if instances:
                for span, score in instances:
                    print(f"- {span!r} (score: {score:.3f})")
            else:
                print("No instances found.")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)
        print("\nPlease ensure you have the required packages (`pip install farm-haystack haystack-ai`)")
