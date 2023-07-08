import csv

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

label_map = {0: "Non-toxic", 1: "Toxic"}

model_name = "unitary/multilingual-toxic-xlm-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def classify_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)

    return probabilities.item()


if __name__ == '__main__':
    with open("text_to_classify.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            text, label = row

            probability = classify_toxicity(text)
            prediction = 1 if probability > 0.5 else 0
            class_label = label_map[prediction]

            print(f"Text: {text}")
            print(f"True Label: {label}")
            print(f"Predicted Label: {class_label}")
            if class_label == label:
                print("Equal")
            else:

                print("Not equal")
            print("="*50)
