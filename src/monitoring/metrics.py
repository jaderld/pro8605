from prometheus_client import start_http_server, Summary, Counter
import time

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests')

def start_metrics_server(port=8001):
    start_http_server(port)
    print(f"Prometheus metrics server running on port {port}")

@REQUEST_TIME.time()
def process_request():
    REQUEST_COUNT.inc()
    time.sleep(0.1)
