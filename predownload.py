from transformers import RobertaModel
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    print("downloading RoBERTa")
    RobertaModel.from_pretrained("roberta-base")
    print("done")
    print("downloading mpnet")
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("done")