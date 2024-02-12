from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Inicjalizacja tokenizera i modelu BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.eval()

# Przykładowe zdanie
sentence = "This is a great product!"

# Tokenizacja i kodowanie tekstu
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

# Przewidywanie sentymentu
predictions = torch.softmax(outputs.logits, dim=1)
positive_score = predictions[0][1].item()
negative_score = predictions[0][0].item()

if positive_score > negative_score:
    print("Positive sentiment")
else:
    print("Negative sentiment")
