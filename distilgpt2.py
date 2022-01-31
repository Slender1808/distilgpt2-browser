from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained('./local_model/')

model = TFGPT2LMHeadModel.from_pretrained('./local_model/',pad_token_id=tokenizer.eos_token_id)

tokenizer.decode(tokenizer.eos_token_id)


sentence = "Hello, my dog is cute"
input_ids = tokenizer.encode(sentence, return_tensors='tf')

output = model.generate(
    input_ids,
    max_length=500,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))