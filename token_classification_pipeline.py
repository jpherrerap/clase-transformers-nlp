from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

hf_account = "jorgeortizfuentes"
model_name = "nominal-groups-recognition-bert-base-spanish-wwm-cased"

tokenizer = AutoTokenizer.from_pretrained(f"{hf_account}/{model_name}")

pipe = pipeline("ner", model=f"{hf_account}/{model_name}", tokenizer=tokenizer, aggregation_strategy="simple")

text = """Traidor. Augusto Pinochet construyó una carrera basada en una extrema obsecuencia con cualquiera que tuviera poder.Asesino. Militares y sacerdotes. Estudiantes y campesinos. Artistas y diplomáticos. La lista de ejecutados por la dictadura de Pinochet se lee como un compendio del horror extendido sobre la sociedad chilena, con el Estado convertido en una máquina de represión y muerte, al servicio del ansia de poder de un solo hombre. Terrorista. Pinochet se presentó como un luchador contra el terrorismo, pero fue el peor terrorista de la historia de Chile. Usando el terrorismo de Estado para expandir el pavor, su dictadura torturó a 28.459 chilenos, ejecutó a 2.125 e hizo desaparecer a otros 1.102. Ladrón. La justicia acreditó, en el Caso Riggs, que Pinochet lideró por años una trama para desviar dinero público hacia su patrimonio personal. Cobarde. Ajeno a cualquier concepto de responsabilidad del mando o de honor militar, Pinochet cargó todas las culpas sobre sus subordinados."""
results = pipe(text)

print(results)

# Clean memory GPU
del pipe
import torch
torch.cuda.empty_cache()