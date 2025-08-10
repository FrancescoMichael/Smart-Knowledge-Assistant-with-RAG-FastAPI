import requests

def test_upload():
    url = "http://localhost:8000/upload"
    
    with open("data.pdf", "rb") as file:
        files = {"file": ("data.pdf", file, "application/pdf")}
        response = requests.post(url, files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_health():
    response = requests.get("http://localhost:8000/health")
    print(f"Health: {response.json()}")

def test_question():
    response = requests.post(
        "http://localhost:8000/ask",
        json={"query": "What is this document about?"}
    )
    print(f"Answer: {response.json()}")

def test_stream_question():
    url = "http://localhost:8000/ask_stream"
    payload = {"query": "What is this document about?"}

    with requests.post(url, json=payload, stream=True) as response:
        print(f"Status: {response.status_code}")
        print("Streaming response:")
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode(), end="", flush=True)
        print()

if __name__ == "__main__":
    test_health()
    # test_upload()
    test_health()
    test_question()
    test_stream_question()