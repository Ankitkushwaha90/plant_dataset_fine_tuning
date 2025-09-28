import torch
import numpy as np
import evaluate
import math
import random
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# 1️⃣ Generate expanded dataset
# -----------------------------
print("📥 Generating expanded dataset for training...")

def generate_expanded_dataset(original_texts, target_size=10000):
    crops = ["rice", "wheat", "maize", "tomatoes", "potatoes", "mangoes", "bananas", "sugarcane", "cotton"]
    plant_diseases = ["blight", "rust", "powdery mildew", "leaf spot", "root rot", "anthracnose", "fusarium wilt", "bacterial canker", "verticillium wilt"]
    solutions = ["fungicides", "crop rotation", "resistant varieties", "organic treatments", "biological controls", "drip irrigation", "soil testing", "pruning", "integrated pest management"]
    adjectives = ["effective", "sustainable", "severe", "preventable", "innovative", "resilient", "manageable", "natural", "advanced"]
    verbs = ["treats", "prevents", "manages", "applies", "controls", "implements", "monitors", "diagnoses", "promotes"]

    # Templates focused on plant diseases and solutions, without farmer or village references
    templates = [
        "{disease} on {crop} is managed with {solution}.",
        "{solution} is {adjective} for controlling {disease} on {crop}.",
        "{disease} affects {crop} and is treated with {solution}.",
        "{verb} {disease} on {crop} using {solution} is {adjective}.",
        "{solution} helps prevent {disease} on {crop}.",
        "{disease} on {crop} is a {adjective} problem solved by {solution}.",
        "{verb} {disease} with {solution} ensures {crop} health.",
        "{solution} is used to combat {disease} on {crop}.",
        "{disease} on {crop} requires {adjective} {solution}.",
        "{verb} {solution} controls {disease} on {crop}.",
        "{solution} is {adjective} for {disease} prevention on {crop}.",
        "{disease} impacts {crop} but {solution} is effective.",
        "{verb} {adjective} {solution} mitigates {disease} on {crop}.",
        "{solution} addresses {disease} on {crop} effectively.",
        "{disease} on {crop} is prevented using {adjective} {solution}.",
        "{verb} {disease} on {crop} with {solution} improves yield.",
        "{solution} is applied to manage {disease} on {crop}.",
        "{disease} on {crop} is controlled with {adjective} {solution}.",
        "{verb} {solution} for {disease} on {crop} is {adjective}.",
        "{solution} promotes {adjective} control of {disease} on {crop}.",
        "{disease} on {crop} is a challenge addressed by {solution}.",
        "{verb} {adjective} {solution} prevents {disease} on {crop}.",
        "{solution} ensures {crop} is protected from {disease}."
    ]

    expanded_texts = original_texts.copy()
    while len(expanded_texts) < target_size:
        template = random.choice(templates)
        crop = random.choice(crops)
        disease = random.choice(plant_diseases)
        sentence = template.format(
            crop=crop,
            disease=disease,
            solution=random.choice(solutions),
            adjective=random.choice(adjectives),
            verb=random.choice(verbs)
        )
        if sentence not in expanded_texts:
            expanded_texts.append(sentence)

    return expanded_texts[:target_size]

# Original dataset
original_train_texts = [
    "Blight on rice is managed with fungicides.",
    "Rust on wheat is controlled with crop rotation.",
    "Leaf spot on maize is treated with resistant varieties.",
    "Powdery mildew on tomatoes is prevented with organic treatments.",
    "Root rot on potatoes is managed with soil testing.",
    "Anthracnose on mangoes is controlled with drip irrigation.",
    "Fusarium wilt on bananas is treated with biological controls.",
    "Bacterial canker on cotton is prevented with pruning.",
    "Blight on sugarcane is managed with integrated pest management.",
    "Rust on wheat is addressed with sustainable fungicides.",
    "Leaf spot on tomatoes is controlled with organic treatments.",
    "Powdery mildew on maize is prevented with resistant varieties.",
    "Root rot on maize is managed with soil testing.",
    "Anthracnose on mangoes is treated with biological controls.",
    "Fusarium wilt on rice is prevented with crop rotation."
]

# Generate 10,000 training examples
train_texts = generate_expanded_dataset(original_train_texts, target_size=10000)

# Test dataset
test_texts = [
    "Blight on rice is controlled with resistant varieties.",
    "Rust on wheat is managed with organic treatments.",
    "Leaf spot on maize is prevented with crop rotation.",
    "Powdery mildew on tomatoes is treated with fungicides.",
    "Fusarium wilt on bananas is controlled with biological controls.",
    "Anthracnose on mangoes is managed with resistant varieties.",
    "Root rot on potatoes is prevented with drip irrigation.",
    "Bacterial canker on cotton is treated with soil testing.",
    "Blight on sugarcane is controlled with integrated pest management.",
    "Rust on wheat is prevented with sustainable solutions.",
    "Leaf spot on tomatoes is managed with pruning.",
    "Powdery mildew on maize is treated with organic treatments.",
    "Root rot on maize is prevented with resistant varieties.",
    "Anthracnose on mangoes is controlled with crop rotation.",
    "Fusarium wilt on rice is managed with fungicides.",
    "Blight on cotton is prevented with biological controls."
]

# Expand test dataset
test_texts = generate_expanded_dataset(test_texts, target_size=1000)

# Convert to Hugging Face Datasets
train_data = Dataset.from_dict({"text": train_texts})
test_data = Dataset.from_dict({"text": test_texts})

# Print sample of training data
print("Sample of training data:")
for text in train_texts[:10]:
    print(f"- {text}")

# -----------------------------
# 2️⃣ Load tokenizer
# -----------------------------
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=64)

# Apply tokenization after dataset creation
train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

train_data.set_format('torch', columns=['input_ids', 'attention_mask'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask'])

# -----------------------------
# 3️⃣ Load model
# -----------------------------
print("🤖 Loading model...")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()  # Reduce VRAM usage

# -----------------------------
# 4️⃣ Data collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------------
# 5️⃣ Training arguments
# -----------------------------
os.makedirs("./mini_gpt_safetensor", exist_ok=True)
training_args = TrainingArguments(
    output_dir="./mini_gpt_safetensor",
    overwrite_output_dir=False,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,  # Reduced from 20 to 10 to halve the training epochs
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=100,
    learning_rate=2e-4,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    fp16=True,
    save_total_limit=2,
    push_to_hub=False,
    report_to="none",
    optim="adamw_torch",
    save_safetensors=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    resume_from_checkpoint=True
)

# -----------------------------
# 6️⃣ Custom compute metrics function for perplexity
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits) if isinstance(logits, np.ndarray) else logits
    labels = torch.tensor(labels) if isinstance(labels, np.ndarray) else labels
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity}

# -----------------------------
# 7️⃣ Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# -----------------------------
# 8️⃣ Train model
# -----------------------------
print("🏋️ Training model with reduced epochs...")
trainer.train()

# -----------------------------
# 9️⃣ Evaluate model
# -----------------------------
print("📊 Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
print(f"Perplexity: {eval_results['eval_perplexity']:.2f}")

# -----------------------------
# 10️⃣ Save model in safetensor format
# -----------------------------
print("💾 Saving model in safetensors format...")
trainer.save_model("./mini_gpt_safetensor")

# Verify model save
if os.path.exists("./mini_gpt_safetensor") and os.path.exists("./mini_gpt_safetensor/model.safetensors"):
    print("✅ Model successfully saved!")
else:
    print("❌ Model save failed!")
    exit()

print("✅ Training complete!")
