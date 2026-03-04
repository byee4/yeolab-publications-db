import os
import json
import time
import logging
import threading
import requests
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

# --- CONFIGURATION ---
BASE_PATH = Path("/tscc/projects/ps-yeolab3/bay001/codebase/yeolab-publications-db")
REPO_PATH = BASE_PATH
TEMPLATE_PATH = BASE_PATH / "code_templates"
MODEL_ID = 'models/gemini-2.5-flash'
MAX_THREADS = 10
API_CONCURRENCY = 2

# Ensure templates directory exists
TEMPLATE_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
file_lock = threading.Lock()
template_lock = threading.Lock()
api_semaphore = threading.BoundedSemaphore(API_CONCURRENCY)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- GLOBAL TEMPLATE CACHE ---
TEMPLATES = {}

def load_all_templates():
    global TEMPLATES
    count = 0
    for t_file in TEMPLATE_PATH.glob("*.json"):
        try:
            with open(t_file, 'r', encoding='utf-8') as f:
                t_data = json.load(f)
                raw = t_data.get("raw_text")
                if raw and t_data.get("steps"):
                    TEMPLATES[raw.strip()] = t_data["steps"]
                    count += 1
        except Exception:
            continue
    logging.info(f"Initialized cache with {count} existing templates.")

def is_url_valid(url):
    if not url or not url.startswith("http"): return False
    try:
        return requests.head(url, timeout=5, allow_redirects=True).status_code == 200
    except: return False

def get_ai_payload(step_data):
    desc = step_data.get("description", "")
    t_name = step_data.get("tool_name", "")
    t_ver = step_data.get("tool_version", "")

    prompt = f"""
    Context: Bioinformatics Pipeline Database
    Step Description: {desc}
    Tool: {t_name if t_name else 'Infer from description'}
    Version: {t_ver if t_ver else 'Infer from description'}

    Task:
    1. Identify tool and version if missing.
       - for eCLIP assays, pull tools/versions/commands primarily from either the CWL workflow https://github.com/yeolab/eclip/ (before 2021) or the Snakemake workflow https://github.com/yeolab/skipper/ (after 2021 or if the description contains "Skipper")
       - for eCLIP peak calling, use https://github.com/yeolab/clipper
       - for alignment of RNA-based assays, use a splice-aware aligner (eg. STAR, HISAT2, RSEM) or an alignment-free quantification tool (eg. Salmon, Kallisto).
       - if tool is not explicitly stated in the description, add "(Inferred with {MODEL_ID})"
    2. Provide a bash code block:
       - Try identifying parameters used inside the description. Description may or may not include the actual tool. 
       - Identify any reference datasets and their sources in the description, and if not exists, use the latest assembly as a placeholder. 
       - Comment out installation instructions (e.g., # conda install...)
       - Do NOT comment the actual execution command.
    3. Find the most relevant GitHub URL, publication or DOI link.
    
    Output Format: Return ONLY a JSON object with these keys:
    "inferred_tool", "inferred_version", "combined_code", "source_url"
    No markdown backticks.
    """
    with api_semaphore:
        for _ in range(3):
            try:
                response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=types.GenerateContentConfig(temperature=0.1))
                return json.loads(response.text.strip().replace('```json', '').replace('```', '').strip())
            except: time.sleep(10); continue
    return None

def save_as_template(raw_text, steps):
    """Saves a new unique raw_text entry to the templates folder."""
    text_hash = hashlib.md5(raw_text.encode()).hexdigest()
    template_file = TEMPLATE_PATH / f"template_{text_hash}.json"
    
    template_data = {
        "raw_text": raw_text,
        "steps": steps,
        "locked": True
    }
    
    with template_lock:
        if not template_file.exists():
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=4)
            TEMPLATES[raw_text] = steps # Update cache in-memory

def process_file(f_path):
    try:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_text = data.get("raw_text", "").strip()
        if not raw_text: return None
        
        modified = False

        # 1. Check local memory cache (Template Match)
        if raw_text in TEMPLATES:
            logging.info(f"MATCH: {f_path.name} -> Using Template.")
            data["steps"] = TEMPLATES[raw_text]
            modified = True
        
        # 2. Query AI and Create Template (Template Miss)
        else:
            logging.info(f"MISS: {f_path.name} -> Querying AI.")
            for step in data.get("steps", []):
                if not step.get("code_example") or not step.get("tool_name"):
                    res = get_ai_payload(step)
                    if res:
                        step["tool_name"] = step.get("tool_name") or res.get("inferred_tool", "")
                        step["tool_version"] = step.get("tool_version") or res.get("inferred_version", "")
                        step["code_example"] = res.get("combined_code", "")
                        url = res.get("source_url", "")
                        step["github_url"] = url if is_url_valid(url) else ""
                        step["code_language"] = "bash"
                        modified = True
                        time.sleep(5)
            
            # After filling all steps via AI, save this file's steps as a new template
            if modified:
                save_as_template(raw_text, data["steps"])

        if modified:
            with file_lock:
                with open(f_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
            return f"Updated {f_path.name}"
            
    except Exception as e:
        return f"Error {f_path.name}: {e}"
    return None

def main():
    load_all_templates()
    json_files = [f for f in REPO_PATH.rglob("GSE145480.json") if "code_templates" not in str(f)]
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_file, f) for f in json_files]
        for future in as_completed(futures):
            res = future.result()
            if res: print(res)

if __name__ == "__main__":
    main()