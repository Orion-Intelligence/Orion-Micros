from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# Load tokenizer and model
model_name = "ibm-research/CTI-BERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Label mapping
label_map = {
    0: "normal",
    1: "cti_classifier"
}

# CLI input loop
print("🔐 CTI Binary Classifier Ready. Type a message (or type 'exit' to quit):\n")

while True:
    user_input = input("📝 Message: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("👋 Exiting CTI classifier.")
        break

    # Tokenize and run
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    print(f"🔐 Classification: {label_map[pred]} (Score: {probs[0][pred]:.3f})")
    print(f"📉 Min: {torch.min(probs):.3f} | 📈 Max: {torch.max(probs):.3f}\n")
