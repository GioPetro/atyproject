import torch
from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
app = FastAPI()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)
tokenizer = AutoTokenizer.from_pretrained("SuiGio/roberta_pubmesh")
model = AutoModelForSequenceClassification.from_pretrained("SuiGio/roberta_pubmesh")

# try:
#     model.cuda()
# except:
#     pass


label_names= [ "Anatomy [A]",
"Organisms [B]",
"Diseases [C]",
"Chemicals and Drugs [D]",
"Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]",
"Psychiatry and Psychology [F]",
"Phenomena and Processes [G]",
"Disciplines and Occupations [H]",
"Anthropology, Education, Sociology, and Social Phenomena [I]",
"Technology, Industry, and Agriculture [J]",
"Information Science [L]",
"Named Groups [M]",
"Health Care [N]",
"Geographicals [Z]"
]

model.config.label2id= {
"Anatomy [A]": 0,
"Organisms [B]": 1,
"Diseases [C]": 2,
"Chemicals and Drugs [D]": 3,
"Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]": 4,
"Psychiatry and Psychology [F]": 5,
"Phenomena and Processes [G]": 6,
"Disciplines and Occupations [H]": 7,
"Anthropology, Education, Sociology, and Social Phenomena [I]": 8,
"Technology, Industry, and Agriculture [J]": 9,
"Information Science [L]": 10,
"Named Groups [M]": 11,
"Health Care [N]": 12,
"Geographicals [Z]": 13,
}


model.config.id2label={
    "0": "Anatomy [A]",
    "1": "Organisms [B]",
    "2": "Diseases [C]",
    "3": "Chemicals and Drugs [D]",
    "4": "Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]",
    "5": "Psychiatry and Psychology [F]",
    "6": "Phenomena and Processes [G]",
    "7": "Disciplines and Occupations [H]",
    "8": "Anthropology, Education, Sociology, and Social Phenomena [I]",
    "9": "Technology, Industry, and Agriculture [J]",
    "10": "Information Science [L]",
    "11": "Named Groups [M]",
    "12": "Health Care [N]",
    "13": "Geographicals [Z]"
}


user_text = " BACKGROUND: The distal GI microbiota of hospitalized preterm neonates has been established to be unique from that of healthy full-term infants; the proximal GI, more specifically gastroesophageal colonization has not been systematically addressed. We prospectively evaluated early colonization of gastroesophageal portion of the GI tract of VLBW infants.METHODS: This study involved 12 infants admitted to a level III NICU with gestational age (GA) 27 +/- 0.5 weeks and birth weight 1105 +/- 77 grams. The gastroesophageal microbial flora was evaluated using 16S rDNA analysis of aspirates collected in a sterile manner during the first 28 days of life.RESULTS: Bacteria were detected in 9 of the 12 neonates. Ureaplasma was the dominant species in the first week of life, however, staphylococci were the predominant bacteria overall. By the fourth week, Gram (-) bacteria increased in abundance to account for 50% of the total organisms. Firmicutes were present in the majority of the neonates and persisted throughout the 4 weeks comprising nearly half of the sequenced clones. Noticeably, only two distinct species of Staphylococcus epidermidis were found, suggesting acquisition from the environment.CONCLUSIONS: In our neonates, the esophagus and stomach environment changed from being relatively sterile at birth to becoming colonized by a phylogenetically diverse microbiota of low individual complexity. By the fourth week, we found predominance of Firmicutes and Proteobacteria. Bacteria from both phyla (CONS and Gram (-) organisms) are strongly implicated as causes of hospital-acquired infections (HAI). Evaluation of the measures preventing colonization with potentially pathogenic and pathogenic microorganisms from the hospital environment may be warranted and may suggest novel approaches to improving quality in neonatal care."


def get_preds(user_text, model=model, tokenizer=tokenizer, label_names=label_names):
    inputs = tokenizer(user_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"]#.to(device)
    attention_mask = inputs["attention_mask"]#.to(device)
    model.eval()

    # Forward pass with the model
    with torch.no_grad():
        # Forward pass with the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Process the outputs as needed
    # For example, if you have a classification task:
    predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Get the predicted label indices for each prediction
    predicted_label_indices = predictions.argmax(dim=1).cpu().numpy()

    # Get the confidence scores (probabilities) for each prediction
    confidence_scores = predictions.max(dim=1).values.cpu().numpy()

    # Combine the category names, predicted label indices, and confidence scores into a list of tuples
    predictions_with_confidence = list(zip(predicted_label_indices, confidence_scores))

    # Sort the predictions with confidence scores in descending order
    predictions_with_confidence_sorted = sorted(predictions_with_confidence, key=lambda x: x[1], reverse=True)

    # Print all predictions with category names and confidence scores in descending order
    category_list, conf_sc = [], []
    for label_index, confidence_score in predictions_with_confidence_sorted:
        category_name = model.config.id2label[str(label_index)]
        print(f"Category: {category_name}, Confidence: {confidence_score:.2f}")
        category_list.append(category_name)
        conf_sc.append(confidence_score)

    return category_list

@app.get("/", tags=["Home"])
def api_home(user_text: str):
    return {'detail': 'Welcome to pub med multi class inference!'}

@app.post("/api/generate", summary="Get multilabel predictions from medical documents", tags=["Predict"])
def inference(user_text: str):
    return get_preds(user_text=user_text)


if __name__=='__main__':
    uvicorn.run('main:app', reload=True)