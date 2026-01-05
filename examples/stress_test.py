
import requests
import base64
import time
import threading
import concurrent.futures
import os
import sys

# Configuration
API_URL = "http://127.0.0.1:8000/generate"
REF_AUDIO_PATH = "test.mp3" # Can be absolute or relative
NUM_USERS = 100
REQUESTS_PER_USER = 5

def send_request(user_id, req_id, audio_b64):
    payload = {
        "text": f"This is request {req_id} from user {user_id}.",
        "ref_audio": audio_b64,
        "ref_text": "Reference text for stress testing.",
        "speed": 1.0,
        "steps": 20 # Keep steps lower for faster stress test? Or default 50. Let's use 20.
    }
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        elapsed = time.time() - start_time
        return True, elapsed, len(response.content)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"User {user_id} Req {req_id} failed: {e}")
        return False, elapsed, 0

def user_routine(user_id, audio_b64, results):
    print(f"User {user_id} started.")
    for i in range(REQUESTS_PER_USER):
        success, elapsed, size = send_request(user_id, i, audio_b64)
        results.append({
            "user": user_id,
            "req": i,
            "success": success,
            "latency": elapsed,
            "size": size
        })
        time.sleep(0.1) # Slight delay between user requests
    print(f"User {user_id} finished.")

def main():
    if not os.path.exists(REF_AUDIO_PATH):
        print(f"Error: {REF_AUDIO_PATH} not found.")
        # Try finding it in root if we are in examples
        alt_path = os.path.join("..", REF_AUDIO_PATH)
        if os.path.exists(alt_path):
             print(f"Found at {alt_path}")
             ref_audio_path = alt_path
        else:
             sys.exit(1)
    else:
        ref_audio_path = REF_AUDIO_PATH

    print(f"Loading {ref_audio_path}...")
    with open(ref_audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

    print(f"Starting stress test: {NUM_USERS} users, {REQUESTS_PER_USER} requests/user.")
    
    all_results = []
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_USERS) as executor:
        futures = []
        for u in range(NUM_USERS):
            # Pass a list to collect results for thread safety (one list per user or just append to managed list)
            # Actually appending to shared list is fine with GIL or simple append, or we can aggregate later.
            # Let's let the thread function append to a thread-safe list or just return results?
            # Easiest: pass a shared list, append is thread-safe in CPython for lists.
            futures.append(executor.submit(user_routine, u, audio_b64, all_results))
            
        concurrent.futures.wait(futures)

    end_total = time.time()
    total_time = end_total - start_total
    
    # Analysis
    total_reqs = len(all_results)
    success_reqs = sum(1 for r in all_results if r["success"])
    avg_latency = sum(r["latency"] for r in all_results) / total_reqs if total_reqs else 0
    
    print("\n=== Stress Test Results ===")
    print(f"Total Requests: {total_reqs}")
    print(f"Successful: {success_reqs}")
    print(f"Failed: {total_reqs - success_reqs}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Throughput: {total_reqs / total_time:.2f} req/s")

if __name__ == "__main__":
    main()
