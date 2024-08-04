import openai

# Replace 'your_api_key' with your actual OpenAI API key
openai.api_key = 'your_api_key'

def ask_question_openai(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=100
    )
    answer = response.choices[0].text.strip()
    return answer

if __name__ == "__main__":
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = ask_question_openai(question)
        print("Answer:", answer)
