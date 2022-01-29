from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('./local_model/')
model = AutoModelForCausalLM.from_pretrained('./local_model/',pad_token_id=tokenizer.eos_token_id)

tokenizer.decode(tokenizer.eos_token_id)

sentence = "Love is"
input_ids = tokenizer.encode(sentence, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=500,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))