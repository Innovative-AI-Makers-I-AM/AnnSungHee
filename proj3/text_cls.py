# STEP 1. import modules
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# STEP 2. create inference
# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
# model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
# with torch.no_grad():
#     logits = model(**inputs).logits

# STEP 2. create inference instance
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# STEP 3. prepare input date
text = "너무 싫어"

# inputs = tokenizer(text, return_tensors="pt")


# STEP 4. inference
result = classifier(text)

# 4-1 preprocessing(data -> tensor(blob))
# 4-2 inference(tensor(blog) -> logit)
# 4-3 postprocessing (logit -> data)

# STEP 5.
print(result)