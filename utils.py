from dotenv import load_dotenv
from datetime import datetime
import csv

load_dotenv()

def log_openai_usage(total_tokens, prompt_tokens, completion_tokens, total_cost):
    with open("openai_api_usage.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                         total_tokens, prompt_tokens, completion_tokens, total_cost,
                         datetime.now().strftime("%m/%d/%Y")])
    