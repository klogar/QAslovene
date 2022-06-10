# from transformers import MT5ForConditionalGeneration, T5Tokenizer
# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
# tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
# article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
# summary = "Weiter Verhandlung in Syrien."
# inputs = tokenizer(article, return_tensors="pt")
# with tokenizer.as_target_tokenizer():
#     labels = tokenizer(summary, return_tensors="pt")
#
# outputs = model(**inputs, labels=labels["input_ids"])
# loss = outputs.loss
# print(outputs)
# print(loss)

from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

sentence = "What does being a Conservative do to the seriousness of a protest? \\n Howard Zinn writes, ""There may be many times when protesters choose to go to jail, as a way of continuing their protest, as a way of reminding their countrymen of injustice. But that is different than the notion that they must go to jail as part of a rule connected with civil disobedience. The key point is that the spirit of protest should be maintained all the way, whether it is done by remaining in jail, or by evading it. To accept jail penitently as an accession to 'the rules' is to switch suddenly to a spirit of subservience, to demean the seriousness of the protest...In particular, the neo-conservative insistence on a guilty plea should be eliminated."""
sentence = sentence.lower()
print(run_model(sentence, num_beams=4, num_return_sequences=4
))