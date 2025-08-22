from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class JobConfig(BaseModel):
    """Pydantic model for job configuration, received as a JSON string in a form field."""

    title: Optional[str] = Field(None, description="Document title (optional).")
    classifier_driver: str = Field(
        ..., description="Driver for classification.", pattern="^(nn|llm)$"
    )
    extractor_driver: Optional[str] = Field(
        None,
        description="Driver for ROAST extraction (optional).",
        pattern="^(nn|llm)$",
    )
    confidence_threshold: float = Field(
        0.9, ge=0.0, le=1.0, description="Confidence threshold for classification."
    )
    allowed_senses: Optional[List[int]] = Field(
        None, description="List of allowed sense class IDs."
    )
    allowed_ages: Optional[List[int]] = Field(
        None, description="List of allowed age class IDs."
    )
    nn_checkpoint_path: Optional[str] = Field(
        "checkpoints/best-model.ckpt",
        description="Path to the classifier .ckpt file (if classifier_driver='nn').",
    )
    llm_ollama_model: Optional[str] = Field(
        "gemma3:1b", description="Name of the Ollama model (if a driver is 'llm')."
    )
    roast_model_path: Optional[str] = Field(
        "QA_RoBERTA_SQUADv2", description="Path to the ROAST (extractive QA) model."
    )
    target_abstracts: Optional[Dict[str, str]] = Field(
        None,
        description="Dictionary of abstract concepts and their explanations to extract.",
    )


class FullJobPayload(JobConfig):
    """Internal model that includes the text content for processing."""

    input_text: str

