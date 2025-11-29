# test_azure_connection.py
from llm_gpt4o import get_client, warmup, chat_call, MODEL_ID

print("Testing Azure OpenAI connection...")
print(f"Model: {MODEL_ID}")

try:
    client = get_client(timeout=30.0)
    print("✓ Client created successfully")

    warmup(client)
    print("✓ Warmup successful")

    response = chat_call(
        client,
        [{"role": "user", "content": "Say 'Hello, Azure!'"}],
        max_tokens=10
    )
    print(f"✓ Test response: {response}")
    print("\n✓ Azure OpenAI connection is working correctly!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nPlease check your Azure OpenAI credentials in config.ini")