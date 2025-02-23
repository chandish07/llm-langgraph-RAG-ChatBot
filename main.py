from app import chat

def main():
    print("GovSearchAI chat")
    print("Welcome to GovSearchAI chat, TechnoMile's latest LLM that serves up information from our data lake about solicitations, awards, contract vehicles, federal vendors, and much more!")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        try:
            response = chat(user_input, chat_history)
            chat_history = response
            print("\nGovSearchAI:", response[-1].content)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'quit' to exit")

if __name__ == "__main__":
    main()
