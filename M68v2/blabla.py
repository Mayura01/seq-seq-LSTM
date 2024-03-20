from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_sentence(input_word, temperature=0.7):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode(input_word, return_tensors="pt")

    # Generate text based on the input word and temperature
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=temperature)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

input_word = "programming"

generated_text = generate_sentence(input_word, temperature=1.0)
print("bot:", generated_text)
print("\n")
