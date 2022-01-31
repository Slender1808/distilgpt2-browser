from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = TFGPT2LMHeadModel.from_pretrained("distilgpt2")

tokenizer.save_pretrained('./local_model/')
model.save_pretrained('./local_model/')