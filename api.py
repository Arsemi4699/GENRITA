import asyncio
import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4
import threading

import redis
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, Json
from tqdm import tqdm

# Assuming these are in the same directory or installed as a package
# You might need to adjust imports based on your project structure.
from GENRITA import GENRITADriver
from ROAST import ROASTDriver

# --- Basic Configuration ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_QUEUE_NAME = "grpipeline_job_queue"
OUTPUT_DIR = Path("output_results")
OUTPUT_DIR.mkdir(exist_ok=True)  # Ensure output directory exists

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GRPipeline API with Redis & File Uploads",
    description="An API for processing text documents with a configurable classifier and instance extractor, using a Redis-backed job queue.",
    version="1.2.0"
)

# --- Redis Connection ---
# This creates a connection pool for Redis.
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    redis_client.ping()
    logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Could not connect to Redis. Please ensure it is running at {REDIS_HOST}:{REDIS_PORT}. Error: {e}")
    # Exit if Redis is not available, as it's a critical dependency.
    exit(1)


# --- GRPipeline Class (Adapted for API) ---
class GRPipeline:
    """
    Processes text documents, integrating configurable classifiers and extractors.
    Adapted for use within the FastAPI application.
    """

    def __init__(self, classifier_driver_type: str, classifier_params: dict, extractor_driver_type: Optional[str],
                 roast_params: dict, processing_params: dict):
        logger.info(f"--- Initializing GRPipeline ---")
        logger.info(f"Classifier Driver: '{classifier_driver_type.upper()}'")
        self.classifier = GENRITADriver.get_classifer(classifier_driver_type, classifier_params)
        logger.info("Classifier initialized successfully.")

        self.target_abstracts = roast_params.get('target_abstracts')
        if self.target_abstracts and extractor_driver_type:
            logger.info(f"Extractor Driver: '{extractor_driver_type.upper()}'")
            logger.info(f"Loading ROAST Model from: {roast_params['roast_model_path']}")
            self.extractor = ROASTDriver.get_extractor(
                driver_type=extractor_driver_type,
                model_name_or_path=roast_params['roast_model_path'],
                score_threshold=roast_params.get('roast_score_threshold', 0.55)
            )
            logger.info(f"ROAST will extract instances for: {list(self.target_abstracts.keys())}")
        else:
            self.extractor = None
            logger.info("ROAST extractor not configured or extractor driver not specified.")

        self.confidence_threshold = processing_params.get('confidence_threshold', 0.0)
        self.allowed_sense_ids = set(processing_params.get('allowed_senses')) if processing_params.get(
            'allowed_senses') else None
        self.allowed_age_ids = set(processing_params.get('allowed_ages')) if processing_params.get(
            'allowed_ages') else None

        logger.info(f"--- Pipeline ready ---")
        if self.allowed_sense_ids: logger.info(f"Filtering for Sense IDs: {self.allowed_sense_ids}")
        if self.allowed_age_ids: logger.info(f"Filtering for Age IDs: {self.allowed_age_ids}")
        logger.info(f"Confidence threshold set to: {self.confidence_threshold}")

    @staticmethod
    def _chunk_text(text: str, target_words: int = 128) -> list[str]:
        text = text.replace('\n', ' ').strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences: return []
        chunks, current_chunk_sentences = [], []
        for sentence in sentences:
            if not sentence: continue
            current_chunk_sentences.append(sentence)
            if len(" ".join(current_chunk_sentences).split()) >= target_words:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        return chunks

    def process_text_content(self, text_content: str, title: str = "Untitled") -> dict:
        logger.info(f"Processing document titled: '{title}'")
        paragraphs = self._chunk_text(text_content)
        if not paragraphs:
            logger.warning("Input text was empty or could not be split into paragraphs.")
            return {'title': title, 'paragraphs': []}
        logger.info(f"Split text into {len(paragraphs)} paragraphs.")

        all_results = []
        last_successful_prediction = None

        for text_paragraph in tqdm(paragraphs, desc=f"Analyzing paragraphs for '{title}'"):
            classification_result = self.classifier.classify(text_paragraph, self.allowed_sense_ids,
                                                             self.allowed_age_ids)
            sense_pred = classification_result['sense_prediction']
            age_pred = classification_result['age_prediction']

            if last_successful_prediction:
                if sense_pred["confidence"] < self.confidence_threshold:
                    sense_pred = last_successful_prediction["sense_prediction"]
                if age_pred["confidence"] < self.confidence_threshold:
                    age_pred = last_successful_prediction["age_prediction"]

            entities = []
            if self.extractor and self.target_abstracts:
                for abstract, explanation in self.target_abstracts.items():
                    try:
                        instances = self.extractor.extract(text_paragraph, abstract.strip(), explanation.strip())
                        for span, score, start, end in instances:
                            entities.append(
                                {"type": abstract, "sample": span, "start_pos": start, "end_pos": end, "score": score})
                    except Exception as e:
                        logger.warning(f"ROAST failed for abstract '{abstract}' on a paragraph. Error: {e}")

            final_result = {
                "text": text_paragraph,
                "sense_prediction": sense_pred,
                "age_prediction": age_pred,
                "entities": entities
            }
            all_results.append(final_result)

            if sense_pred['class_id'] != -1 and age_pred['class_id'] != -1:
                last_successful_prediction = final_result

        return {'title': title, 'paragraphs': all_results}


# --- Pydantic Models for API Request Validation ---
class JobConfig(BaseModel):
    """Pydantic model for job configuration, received as a JSON string in a form field."""
    title: Optional[str] = Field(None, description="Document title (optional).")
    classifier_driver: str = Field(..., description="Driver for classification.", pattern="^(nn|llm)$")
    extractor_driver: Optional[str] = Field(None, description="Driver for ROAST extraction (optional).",
                                            pattern="^(nn|llm)$")
    confidence_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Confidence threshold for classification.")
    allowed_senses: Optional[List[int]] = Field(None, description="List of allowed sense class IDs.")
    allowed_ages: Optional[List[int]] = Field(None, description="List of allowed age class IDs.")
    nn_checkpoint_path: Optional[str] = Field("checkpoints/best-model.ckpt",
                                              description="Path to the classifier .ckpt file (if classifier_driver='nn').")
    llm_ollama_model: Optional[str] = Field("gemma3:1b", description="Name of the Ollama model (if a driver is 'llm').")
    roast_model_path: Optional[str] = Field("QA_RoBERTA_SQUADv2",
                                            description="Path to the ROAST (extractive QA) model.")
    target_abstracts: Optional[Dict[str, str]] = Field(None,
                                                       description="Dictionary of abstract concepts and their explanations to extract.")


class FullJobPayload(JobConfig):
    """Internal model that includes the text content for processing."""
    input_text: str


# --- Background Worker ---
def process_queue_worker():
    """
    Worker function that pulls jobs from the Redis queue and processes them.
    This runs in a separate thread to avoid blocking the main FastAPI event loop.
    """
    logger.info("Job queue worker thread started.")
    while True:
        try:
            _, job_data_json = redis_client.blpop(REDIS_QUEUE_NAME)
            job_data = json.loads(job_data_json)
            job_id = job_data['job_id']
            # Use the internal model to validate the full payload from Redis
            config = FullJobPayload(**job_data['config'])

            logger.info(f"Pulled job {job_id} from Redis queue. Starting processing.")
            redis_client.hset(f"job:{job_id}", "status", "processing")

            # --- Parameter Setup ---
            classifier_params = {}
            if config.classifier_driver == 'nn':
                classifier_params['checkpoint_path'] = config.nn_checkpoint_path
            elif config.classifier_driver == 'llm':
                classifier_params['ollama_model_name'] = config.llm_ollama_model

            roast_params = {
                'roast_model_path': config.roast_model_path,
                'target_abstracts': config.target_abstracts
            }

            processing_params = {
                'confidence_threshold': config.confidence_threshold,
                'allowed_senses': config.allowed_senses,
                'allowed_ages': config.allowed_ages
            }

            # --- Pipeline Initialization and Execution ---
            pipeline = GRPipeline(
                classifier_driver_type=config.classifier_driver,
                classifier_params=classifier_params,
                extractor_driver_type=config.extractor_driver,
                roast_params=roast_params,
                processing_params=processing_params
            )

            results = pipeline.process_text_content(
                text_content=config.input_text,
                title=config.title or "Untitled Document"
            )

            # --- Save result to file and update Redis ---
            output_path = OUTPUT_DIR / f"{job_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            logger.info(f"Job {job_id} completed successfully. Results saved to {output_path}")
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "completed",
                "result_path": str(output_path)
            })

        except Exception as e:
            job_id_for_error = "unknown"
            try:
                job_id_for_error = json.loads(job_data_json).get('job_id', 'unknown')
            except:
                pass

            logger.critical(f"Job {job_id_for_error} failed with a critical error: {e}", exc_info=True)
            if job_id_for_error != "unknown":
                redis_client.hset(f"job:{job_id_for_error}", mapping={
                    "status": "failed",
                    "error_message": str(e)
                })


# --- API Endpoints ---
@app.on_event("startup")
def startup_event():
    """On application startup, start the background worker thread."""
    threading.Thread(target=process_queue_worker, daemon=True).start()


@app.post("/process", status_code=202)
async def create_processing_job(
        file: UploadFile = File(..., description="The .txt file to process."),
        config_json: str = Form(..., description="A JSON string with the job configuration.")
):
    """
    Accepts a .txt file and a JSON string of configuration options.
    It adds the job to the Redis queue and returns a job ID.
    """
    job_id = str(uuid4())

    # 1. Validate the incoming JSON configuration string
    try:
        config = JobConfig.parse_raw(config_json)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid configuration JSON: {e}")

    # 2. Read the uploaded file content
    try:
        input_text = await file.read()
        input_text = input_text.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Could not read or decode the uploaded file. Ensure it is a valid UTF-8 text file. Error: {e}")
    finally:
        await file.close()

    # 3. Store initial job status in a Redis hash
    redis_client.hset(f"job:{job_id}", "status", "queued")

    # 4. Create the full job payload, including the text from the file
    full_config_payload = config.model_dump()
    full_config_payload['input_text'] = input_text

    job_data = {
        "job_id": job_id,
        "config": full_config_payload
    }

    # 5. Push the job to the Redis queue
    redis_client.rpush(REDIS_QUEUE_NAME, json.dumps(job_data))

    logger.info(f"Job {job_id} accepted and queued in Redis for file '{file.filename}'.")
    return {"job_id": job_id, "status": "queued", "message": "Your request has been queued for processing."}


@app.get("/results/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll this endpoint with a job ID to get the current status.
    If completed, it provides the path to the result file.
    """
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    return {"job_id": job_id, **job_data}


@app.get("/results/{job_id}/download")
async def download_job_result(job_id: str):
    """
    Downloads the result file for a completed job.
    """
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    if job_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {job_data.get('status')}")

    result_path = job_data.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found on server.")

    return FileResponse(path=result_path, filename=Path(result_path).name, media_type='application/json')

# To run this application:
# 1. Make sure you have fastapi, uvicorn, and redis-py installed:
#    pip install "fastapi[all]" redis
# 2. Make sure you have a Redis server running.
# 3. Save the code as `api.py`.
# 4. Run the server from your terminal:
#    uvicorn api:app --reload
