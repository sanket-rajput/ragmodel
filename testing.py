import requests
import time
from datetime import datetime

URL = "http://127.0.0.1:8000/ask"
QUESTION = "Where can i register"

count = 0

print("🚀 Load test started...")
print("-" * 80)

while True:
    try:
        start_time = time.time()  # ⏱ start timer

        response = requests.post(
            URL,
            json={"question": QUESTION},
            timeout=30
        )

        end_time = time.time()  # ⏱ end timer
        latency_ms = (end_time - start_time) * 1000

        count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        print(
            f"[{timestamp}] "
            f"#{count} | "
            f"Latency: {latency_ms:.0f} ms | "
            f"Status: {response.status_code} | "
            f"Response: {response.json()}"
        )

        time.sleep(1)  # 🧘 throttle requests

    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        break

    except requests.exceptions.ConnectionError:
        print("❌ Connection error – is the server running?")
        break

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        break

    except Exception as e:
        print("⚠️ Unexpected error:", e)
        break
