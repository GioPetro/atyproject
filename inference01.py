import torch
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("SuiGio/roberta_pubmesh")
model = AutoModelForSequenceClassification.from_pretrained("SuiGio/roberta_pubmesh")
model.cuda()

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
user_text1 = "The aim of this study was to evaluate the effect of anipamil, a phenyalkylamine-derived Ca(2+)-antagonist, on aortic intimal thickening and smooth muscle cell (SMC) phenotype in 2K-1C hypertensive rabbits. Monoclonal antimyosin antibodies [SM-E7, NM-G2, and NM-F6, which respectively, recognize smooth muscle (SM), A-type-, and B-type-like nonmuscle (NM) myosin heavy chains (MyHC)] identify different aortic SMC types: adult (SM-E7-positive), postnatal (SM-E7- and NM-G2-positive), and fetal (SM-E7-, NM-G2-, and NM-F6-positive). Twenty-four hypertensive rabbits were studied 2.5 months (n = 12) and 4 months (n = 12) after clipping. Six animals from each group were given anipamil (40 mg orally, once daily) immediately after surgery. The remaining 2K-1C were given a daily oral placebo. Normotensive age-matched controls were also studied. Transverse cryosections of aorta were taken for computerized morphometry and immunocytochemical studies. Primary and secondary SMC cultures were used to define potential changes in cell phenotype after adding anipamil to the culture medium. In untreated 2K-1C, intimal thickening, mainly composed of postnatal-type SMC, was found by 2.5 months after clipping. Morphometric and immunofluorescence studies in anipamil-treated 2K-1C rabbits revealed absent or negligible intimal thickening and a decrease of postnatal-type SMC from the underlying media. In culture experiments, growth inhibition of SMC by anipamil was accompanied by the expression of SM-MyHC in all SMC, ie, the appearance of a more differentiated cell phenotype compared to control cultures. In conclusion, prevention of intimal thickening in anipamil-treated 2K-1C was achieved through selective reduction in the media of the postnatal-type SMC. This could be achieved by reducing NM-MyHC content or increasing synthesis of SM-MyHC expression. As blood pressure was not significantly lowered by anipamil treatment, a direct and specific antiproliferative action of this drug on medial SMC is likely to take place."
user_text = "The current literature considers a number of clinical factors which affect the fit of all-ceramic laminate veneers. However, little consideration has been given to the refractory die materials, and the laboratory techniques used during the construction of these restorations. This study found a wide range of dimensional change occurred during setting and through six firing cycles, for seven refractories recommended for the construction of laminate veneers. It is therefore important that where patient treatment involves the use of veneers the clinician considers the suitability of the materials offered by the laboratory, in order to obtain the optimum marginal integrity."
inputs = tokenizer(user_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

model.eval()

# Forward pass with the model
with torch.no_grad():
    # Move inputs to appropriate device (e.g., GPU if available)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

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
for label_index, confidence_score in predictions_with_confidence_sorted:
    category_name = model.config.id2label[str(label_index)]
    print(f"Category: {category_name}, Confidence: {confidence_score:.2f}")

