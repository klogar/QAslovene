from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

print(run_model("which is best conductor? \\n (a) iron (b) feather"))