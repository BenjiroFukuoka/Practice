import csv
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

toxic_label_map = ["non-toxic", "toxic"]
adult_content_label_map = ["non-sex", "sex"]

toxicity_model_name = "unitary/multilingual-toxic-xlm-roberta"
adult_content_model_name = "ziadA123/adultcontentclassifier"

toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name)

adult_content_tokenizer = AutoTokenizer.from_pretrained(adult_content_model_name)
adult_content_model = AutoModelForSequenceClassification.from_pretrained(adult_content_model_name)


def classify_toxicity(text):
    inputs = toxicity_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = toxicity_model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)

    return probabilities.item()


def classify_adult_content(text):
    inputs = adult_content_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = adult_content_model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)

    return probabilities.tolist()[0]


if __name__ == '__main__':
    execution_times = []

    with open("text_to_classify.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            text, label1, label2 = row

            start_time = time.process_time_ns()
            toxic_probability = classify_toxicity(text)
            adult_content_probability = classify_adult_content(text)
            end_time = time.process_time_ns()
            execution_time = (end_time - start_time) / 1e9

            execution_times.append(execution_time)

            toxic_prediction = 1 if toxic_probability > 0.5 else 0
            toxic_class_label = toxic_label_map[toxic_prediction]

            adult_content_prediction = 1 if adult_content_probability[0] < 0.5 else 0
            adult_content_class_label = adult_content_label_map[adult_content_prediction]

            print(f"Text: {text}")
            print(f"True Label: {label1}, {label2}")
            print(f"Toxicity Predicted Label: {toxic_class_label}")
            print(f"Adult Content Predicted Label: {adult_content_class_label}")
            print(f"Execution Time: {execution_time} seconds")
            print("="*50)

    avg_execution_time = sum(execution_times) / len(execution_times)
    print(f"Average Execution Time: {avg_execution_time} seconds")
