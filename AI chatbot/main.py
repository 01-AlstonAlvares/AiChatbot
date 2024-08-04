# Import required libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT model and tokenizer
model_name = "gpt2"  # You can use 'gpt2-medium', 'gpt2-large', etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def ask_question(question):
    # Tokenize the input question
    inputs = tokenizer.encode(question, return_tensors="pt")

    # Generate a response
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

    # Decode the response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = ask_question(question)
        print("Answer:", answer)
