# GENERATIVE-TEXT-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: TEJASWINI MADDIRALA

*INTERN ID*: CT12QRY

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

## DESCRIPTION
The code is Designed to generate text using the GPT-2 model from the Transformers library by Hugging Face.GPT-2 is a state-of-the-art language model that can generate human-like text based on a given prompt.

EDITOR PLATFORM: VS CODE

Key Components of the Code:
Imports:
torch: The PyTorch library, which is the backend framework used for deep learning.
GPT2LMHeadModel and GPT2Tokenizer: These are classes from the Hugging Face Transformers library. GPT2LMHeadModel is the GPT-2 model architecture for text generation, while GPT2Tokenizer is used to convert text to tokens (numbers) that the model can process and to decode the tokens back into human-readable text.

Loading the Pre-trained GPT-2 Model:
model_name = "gpt2" specifies that the base GPT-2 model is being used. There are larger variants available like gpt2-medium, gpt2-large, and gpt2-xl, which can generate text with more complexity but require more computational resources.
tokenizer = GPT2Tokenizer.from_pretrained(model_name) loads the pre-trained tokenizer for GPT-2.
model = GPT2LMHeadModel.from_pretrained(model_name) loads the actual pre-trained GPT-2 model.

Setting the Model to Evaluation Mode:
model.eval() switches the model to evaluation mode, which is necessary when using the model for inference (text generation), as this disables unnecessary operations used during training (like dropout).

Function to Generate Text:
generate_text(prompt, max_length=150, num_return_sequences=1) is the function responsible for generating text based on a user-supplied prompt.
prompt: The input text given by the user.
max_length=150: The maximum length of the generated text.
num_return_sequences=1: The number of different text generations to return.
Encoding the Prompt: The prompt is encoded into token IDs (numerical format) using the tokenizer: input_ids = tokenizer.encode(prompt, return_tensors='pt').
Generating Text: The model generates text based on the input prompt using model.generate(), which performs autoregressive text generation. The options within this function, like max_length, no_repeat_ngram_size, and early_stopping, control the length and quality of the output.
no_repeat_ngram_size=2 ensures that the model does not generate repetitive n-grams (e.g., repeating words or phrases).
pad_token_id=tokenizer.eos_token_id sets the padding token to be the EOS (End Of Sequence) token, which tells the model when to stop generating text.
Decoding the Output: After generating the token IDs, the function decodes them back into human-readable text with tokenizer.decode(). This converts the model's output into text.

Main Loop for User Input:
The if __name__ == "__main__": block (corrected from _name_) ensures that the code runs only if the script is executed directly (not imported as a module).
The user is prompted to enter text, and the program continuously generates text based on that input until the user types 'exit' to quit the loop.

Displaying the Generated Text:
The generated text is printed to the console. If multiple sequences are generated, they are displayed one by one.

How the Code Works:
When you run the script, the model and tokenizer are loaded.
You are prompted to enter a text prompt. The model will then generate a text continuation based on that input.
The output is displayed, and the program asks for another prompt.
If you type 'exit', the program will stop running.

Potential Use Cases of the Text Generation Model:
Creative Writing
Chatbots
Idea Generation
Text completion
Learning and Education
Entertainment and Games
Content Moderation and filtering

This code snippet demonstrates a simple yet powerful application of GPT-2 for text generation. It can be easily extended or modified for more specific tasks like 
conversational agents, content generation, or creative writing aids. It's a good starting point for anyone interested in working with state-of-the-art language 
models in Python using the Hugging Face transformers library.

# OUTPUT:

Giving Input: 
![Image](https://github.com/user-attachments/assets/e8e63faf-113b-4fe2-b160-ddaf130fe122)

Output:
![Image](https://github.com/user-attachments/assets/016b94a2-af6f-4e26-b6be-9d6dc9ff2675)


