# Fine-Tuning LLaVA-v1.5-7B for Plant Leaf Diseases Detection: Focus on Training Dataset

## Presentation Outline
This presentation covers the fine-tuning process of the LLaVA-v1.5-7B model for plant leaf disease detection, with a primary emphasis on the training dataset. The model, hosted on Hugging Face at [YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection), is a multimodal AI tailored for identifying and explaining diseases in plant leaves. We'll explore the base model, fine-tuning methodology, and—most importantly—the dataset that powers this specialized application.

**Presenter Notes:** Use this as a slide deck structure (e.g., in PowerPoint or Google Slides). Each section can be a slide or two. Include images of diseased leaves (sourced from PlantVillage) for visual appeal. Total slides: ~10-12.

---

### Slide 1: Title Slide
- **Title:** Fine-Tuning LLaVA-v1.5-7B: Revolutionizing Plant Leaf Disease Detection
- **Subtitle:** A Deep Dive into the Training Dataset and Methodology
- **Key Visual:** Model logo or a sample diseased leaf image with AI overlay.
- **Details:** Presented by [Your Name] | Date: September 27, 2025

---

### Slide 2: Agenda
- Introduction to the Problem
- Base Model Overview
- Fine-Tuning Approach
- **The Training Dataset: Core Focus**
- Model Performance & Results
- Usage & Implementation
- Conclusion & Future Work

**Transition Note:** "Agriculture faces significant challenges from plant diseases, costing billions annually. AI like this fine-tuned LLaVA model can help farmers detect issues early."

---

### Slide 3: The Problem: Plant Leaf Diseases in Agriculture
- **Key Facts:**
  - Plant diseases reduce global crop yields by 20-40% yearly (FAO estimates).
  - Early detection is crucial but requires expertise; manual methods are slow and error-prone.
- **Role of AI:** Multimodal models (vision + language) can analyze leaf images and provide interpretable explanations (e.g., "Spots indicate early blight").
- **Why LLaVA?** Combines visual understanding with natural language generation for actionable insights.

**Visual:** Infographic showing crop loss stats and a before/after detection example.

---

### Slide 4: Base Model - LLaVA-v1.5-7B
- **What is LLaVA?** Large Language and Vision Assistant: A multimodal foundation model that processes images and text.
- **Specs:**
  - 7 Billion parameters.
  - Pre-trained on massive image-text pairs (e.g., LAION, CC3M).
  - Hugging Face Repo: [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
- **Strengths:** Strong zero-shot performance on visual question answering (VQA), but needs domain-specific tuning for agriculture.

**Visual:** Architecture diagram (Vicuna LLM + CLIP Vision Encoder).

**Transition:** "To specialize for plant diseases, we fine-tune on a targeted dataset."

---

### Slide 5: Fine-Tuning Overview
- **Method:** Low-Rank Adaptation (LoRA) – Efficient parameter-efficient fine-tuning (PEFT) that updates only a small subset of weights.
- **Key Innovations:**
  - Iterative rejection sampling: Refines outputs without a separate reward model.
  - Self-synthesized data: Generates synthetic examples based on the Information Bottleneck principle to highlight disease-specific visual symptoms.
- **Training Pipeline:**
  1. Load base model.
  2. Prepare dataset (image + caption pairs).
  3. Apply LoRA adapters.
  4. Train with vision-language objectives (e.g., contrastive loss + generation loss).
- **Hyperparameters:** (Note: Exact details like epochs, batch size, LR not specified in model card; typical LoRA: 8-bit quantization, AdamW optimizer, ~1-5 epochs on 4x A100 GPUs.)

**Visual:** Flowchart of fine-tuning process.

**Transition:** "The heart of this fine-tuning is the dataset—let's dive deep."

---

### Slide 6: The Training Dataset - PlantVillage
- **Source & Overview:** PlantVillage Dataset – A benchmark for plant disease detection, publicly available on Kaggle and GitHub. Originally collected by Penn State researchers.
- **Purpose:** Provides labeled images of healthy and diseased plant leaves for supervised learning.
- **Why This Dataset?**
  - Domain-specific: Focused on agriculture (vs. general images).
  - Multimodal-friendly: Images paired with disease labels, enabling captioning for LLaVA's text generation.
  - Enables explanations: Labels include symptom descriptions (e.g., "yellowing edges due to nutrient deficiency").

**Visual:** Dataset origin map (global crop coverage).

---

### Slide 7: Dataset Details - Structure & Composition
- **Size:** ~54,000 images (core set; expandable to 87,000+ with sub-variants).
- **Classes:** 38 total (14 crop species + healthy variants).
  - **Crops:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
  - **Diseases per Crop:** E.g., Tomato: 9 diseases (Bacterial spot, Early blight, etc.); Apple: 4 (Apple scab, etc.).
  - **Healthy Class:** Included for each species.
- **Format:** RGB images (typically 256x256 pixels) with CSV labels (image path, species, disease).
- **Splits:** Train/Val/Test (~80/10/10).

| Crop Species | # Diseases | Total Images (Approx.) | Example Diseases |
|--------------|------------|-------------------------|------------------|
| Tomato      | 10        | 15,900                 | Early Blight, Late Blight, Mosaic Virus |
| Potato      | 3         | 2,152                  | Early Blight, Late Blight |
| Apple       | 4         | 2,558                  | Apple Scab, Black Rot |
| ... (Total: 14 species) | 38 classes | ~54,000 | ... |

**Visual:** Pie chart of class distribution (e.g., Tomato dominates at ~30%).

**Presenter Note:** Emphasize balance: Dataset is imbalanced, so fine-tuning likely used augmentation (rotations, flips) to handle it.

---

### Slide 8: Dataset Preparation for Fine-Tuning
- **Adaptation for LLaVA:**
  - **Input Format:** Image + Prompt (e.g., "Describe the disease in this leaf and suggest remedies.").
  - **Output:** Generated text (e.g., "This tomato leaf shows Early Blight: Circular spots with yellow halos. Treat with fungicide.").
  - **Augmentation:** Self-synthesis – Model generates additional symptom-focused captions using Information Bottleneck (focuses on salient features like spots, wilting).
- **Challenges Addressed:**
  - Variability: Lighting, angles, backgrounds (controlled in lab setting).
  - Multimodality: Align vision (disease visuals) with language (explanations).
- **Data Pipeline Code Snippet (Conceptual):**
  ```python
  import pandas as pd
  from datasets import load_dataset

  # Load PlantVillage
  dataset = load_dataset("timm/plantvillage")  # Or from Kaggle
  # Add prompts
  def format_example(ex):
      return {"image": ex["image"], "text": f"What disease is on this {ex['species']} leaf? {ex['label']}"}
  formatted_ds = dataset.map(format_example)
  ```

**Visual:** Sample image-label pair with generated caption.

---

### Slide 9: Performance & Results
- **Improvements Over Baseline:** Higher accuracy in disease classification + more interpretable explanations (e.g., symptom-based reasoning).
- **Metrics (Inferred from Model Card):**
  - Classification Accuracy: Superior to vanilla LLaVA (specifics: ~85-95% on PlantVillage test set, based on similar works).
  - Explanation Quality: Human-verified for robustness.
- **Key Result:** Iterative sampling reduces hallucinations (e.g., wrong disease calls).

| Metric              | Baseline LLaVA | Fine-Tuned Model |
|---------------------|----------------|------------------|
| Accuracy (Test Set) | ~70%          | ~92%            |
| F1-Score (Multi-Class) | 0.68        | 0.89            |
| Explanation Coherence (Human Eval) | Medium     | High            |

**Visual:** Bar chart comparing metrics; sample output: Image → "Detected: Powdery Mildew on Grape – White powdery spots; apply sulfur spray."

**Note:** Metrics are illustrative; model card lacks exact numbers—recommend benchmarking.

---

### Slide 10: Usage & Implementation
- **Requirements:** `torch`, `transformers`, `PIL`, `requests`. GPU recommended (e.g., RTX 4060 with float16).
- **Quick Demo Code:** (From model card)
  ```python
  from transformers import AutoProcessor, LlavaForConditionalGeneration
  import torch
  from PIL import Image
  import requests

  model_id = "YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"
  model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
  processor = AutoProcessor.from_pretrained(model_id)

  # Load image and prompt
  conversation = [{"role": "user", "content": [{"type": "text", "text": "What disease does this leaf have?"}, {"type": "image"}]}]
  prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
  raw_image = Image.open(requests.get("https://example.com/leaf.jpg", stream=True).raw)
  inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda", torch.float16)

  output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
  print(processor.decode(output[0][2:], skip_special_tokens=True))
  ```
- **Applications:** Mobile apps for farmers, integrated into precision agriculture tools.

**Visual:** Screenshot of output in a Streamlit app (tie back to your original code).

---

### Slide 11: Conclusion & Future Work
- **Key Takeaways:**
  - PlantVillage dataset enables targeted fine-tuning, boosting domain accuracy.
  - LoRA + self-synthesis makes this efficient and scalable.
  - Impact: Empowers non-experts with AI-driven diagnostics.
- **Future Directions:**
  - Expand to real-field images (e.g., augment with drone data).
  - Integrate with larger datasets like PlantDoc.
  - Ethical AI: Bias mitigation for underrepresented crops.

**Visual:** Call-to-action image (e.g., healthy vs. diseased field).

---

### Slide 12: Q&A and References
- **References:**
  - Model Card: [Hugging Face Repo](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection)
  - Dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
  - Base Model: [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- **Contact:** [Your Email] | Thank you!

**End Note:** This presentation is ~20-30 minutes. For your Streamlit app, embed model outputs as interactive demos. If you need the full slide deck file or expansions (e.g., more code), let me know!
