from transformers import pipeline

hf_account = "jorgeortizfuentes"
model_name = "fake-news-bert-base-spanish-wwm-cased"
pipe = pipeline(model=f"{hf_account}/{model_name}")

prediction = pipe("Investigadores descubren que los elefantes son alienígenas llegados hace 10.000 años a la Tierra")

print(prediction)

# Clean memory GPU
del pipe
import torch
torch.cuda.empty_cache()