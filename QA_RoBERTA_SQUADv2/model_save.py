from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Replace this with your actual directory
save_path = "./"

model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
