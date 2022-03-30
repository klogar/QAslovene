import torch
from transformers import BartTokenizer
from bart import MyBart

base_model = "facebook/bart-large"
unifiedqa_path = "unifiedQA-uncased/best-model.pt" # path to the downloaded checkpoint

tokenizer = BartTokenizer.from_pretrained(base_model)
model = MyBart.from_pretrained(base_model, state_dict=torch.load(unifiedqa_path))
model.eval()

x = model.generate_from_string("Which is best conductor? \\n (A) iron (B) feather", tokenizer=tokenizer)
print (x)

x = model.generate_from_string("What is the sum of 3 and 5? \\n (A) 8 (B) 3 (C) 5 (D) 10", tokenizer=tokenizer)
print (x)