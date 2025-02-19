import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can also use "gpt2-medium", "gpt2-large", or "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Function to generate text based on a prompt
def generate_text(prompt, max_length=150, num_return_sequences=1):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id  # Use EOS token for padding
        )

    # Decode the generated text
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_texts

# Main loop for user input
if _name_ == "_main_":
    print("Welcome to the Text Generation Model!")
    while True:
        user_prompt = input("Enter a prompt (or type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
        
        # Generate text based on the user prompt
        generated_paragraphs = generate_text(user_prompt, max_length=200, num_return_sequences=1)  # Increased max_length
        
        # Display the generated text
        print("\nGenerated Text:")
        for paragraph in generated_paragraphs:
            print(paragraph)
        print("\n" + "="*50 + "\n")
