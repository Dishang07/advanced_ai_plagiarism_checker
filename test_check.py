import uvicorn
from main import app
import threading
import time
import requests

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=8500, log_level='error')

thread = threading.Thread(target=run_server, daemon=True)
thread.start()
time.sleep(3) # Wait for server to start

start_time = time.time()
print('Sending request to test PDF processing...')
try:
    with open('dummy.pdf', 'rb') as f:
        files = {'file': ('dummy.pdf', f, 'application/pdf')}
        response = requests.post('http://127.0.0.1:8500/api/check', files=files)
    print(f'Status: {response.status_code}')
    print(f'Time taken: {time.time() - start_time:.2f} seconds')
    data = response.json()
    print(f'Total chunks processed: {data.get("total_sentences")}')
    print(f'Plagiarism score: {data.get("plagiarism_score")}')
except Exception as e:
    print(f'Error: {e}')
