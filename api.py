import logging
import json
from pathlib import Path
from uuid import uuid4
import threading
import redis
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from schema import *
from GRP import GRPipeline


# --- Basic Configuration ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_QUEUE_NAME = "grpipeline_job_queue"
OUTPUT_DIR = Path("output_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- FastAPI App Initialization ---
app = FastAPI(
    title="GRPipeline API with Redis & File Uploads",
    description="An API for processing text documents with a configurable classifier and instance extractor, using a Redis-backed job queue.",
    version="2.0.0",
)


# --- Redis Connection ---
try:
    redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
    )
    redis_client.ping()
    logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.ConnectionError as e:
    logger.error(
        f"Could not connect to Redis. Please ensure it is running at {REDIS_HOST}:{REDIS_PORT}. Error: {e}"
    )
    exit(1)


# --- Background Worker ---
def process_queue_worker():
    """
    Worker function that pulls jobs from the Redis queue and processes them.
    This runs in a separate thread to avoid blocking the main FastAPI event loop.
    """
    logger.info("Job queue worker thread started.")
    pipeline = None
    config = None
    while True:
        try:
            _, job_data_json = redis_client.blpop(REDIS_QUEUE_NAME)
            job_data = json.loads(job_data_json)
            job_id = job_data["job_id"]

            config = FullJobPayload(**job_data["config"])

            logger.info(f"Pulled job {job_id} from Redis queue. Starting processing.")
            redis_client.hset(f"job:{job_id}", "status", "processing")

            # --- Parameter Setup ---
            classifier_params = {}
            if config.classifier_driver == "nn":
                classifier_params["checkpoint_path"] = config.nn_checkpoint_path
            elif config.classifier_driver == "llm":
                classifier_params["ollama_model_name"] = config.llm_ollama_model

            roast_params = {
                "roast_model_path": config.roast_model_path,
                "target_abstracts": config.target_abstracts,
            }

            processing_params = {
                "confidence_threshold": config.confidence_threshold,
                "allowed_senses": config.allowed_senses,
                "allowed_ages": config.allowed_ages,
            }

            # --- Pipeline Initialization and Execution ---
            pipeline = GRPipeline(
                classifier_driver_type=config.classifier_driver,
                classifier_params=classifier_params,
                extractor_driver_type=config.extractor_driver,
                roast_params=roast_params,
                processing_params=processing_params,
                logger=logger
            )

            results = pipeline.process_text_content(
                text_content=config.input_text,
                title=config.title or "Untitled Document",
            )

            # --- Save result to file and update Redis ---
            results["job_id"] = job_id
            output_path = OUTPUT_DIR / f"{job_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            logger.info(
                f"Job {job_id} completed successfully. Results saved to {output_path}"
            )
            redis_client.hset(
                f"job:{job_id}",
                mapping={"status": "completed", "result_path": str(output_path)},
            )
        except Exception as e:
            job_id_for_error = "unknown"
            try:
                job_id_for_error = json.loads(job_data_json).get("job_id", "unknown")
            except:
                pass

            logger.critical(
                f"Job {job_id_for_error} failed with a critical error: {e}",
                exc_info=True,
            )
            if job_id_for_error != "unknown":
                redis_client.hset(
                    f"job:{job_id_for_error}",
                    mapping={"status": "failed", "error_message": str(e)},
                )
        finally:
            if pipeline:
                pipeline.cleanup(config.classifier_driver, config.extractor_driver)


# --- API Endpoints ---
@app.on_event("startup")
def startup_event():
    """On application startup, start the background worker thread."""
    threading.Thread(target=process_queue_worker, daemon=True).start()


@app.post("/process", status_code=202)
async def create_processing_job(
    file: UploadFile = File(..., description="The .txt file to process."),
    config_json: str = Form(
        ..., description="A JSON string with the job configuration."
    ),
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
        input_text = input_text.decode("utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read or decode the uploaded file. Ensure it is a valid UTF-8 text file. Error: {e}",
        )
    finally:
        await file.close()

    # 3. Store initial job status in a Redis hash
    redis_client.hset(f"job:{job_id}", "status", "queued")

    # 4. Create the full job payload, including the text from the file
    full_config_payload = config.model_dump()
    full_config_payload["input_text"] = input_text

    job_data = {"job_id": job_id, "config": full_config_payload}

    # 5. Push the job to the Redis queue
    redis_client.rpush(REDIS_QUEUE_NAME, json.dumps(job_data))

    logger.info(
        f"Job {job_id} accepted and queued in Redis for file '{file.filename}'."
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Your request has been queued for processing.",
    }


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
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job_data.get('status')}",
        )

    result_path = job_data.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found on server.")

    return FileResponse(
        path=result_path, filename=Path(result_path).name, media_type="application/json"
    )


# To run this application:
#    uvicorn api:app --reload
