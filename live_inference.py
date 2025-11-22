import os
import torch
import time
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pythonosc import udp_client
import speech_recognition as sr
import sys
import csv
from datetime import datetime

# Force UTF-8 output for Windows terminals to prevent encoding crashes
sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
OSC_IP = "127.0.0.1"
OSC_PORT = 7000
MIC_INDEX = 58  # <---  specific microphone index (Live! Cam)

# Path to  DistilBERT model 
MODEL_DIR = "./results" 
ARCHITECTURE = "distilbert-base-uncased"

def get_best_model():
    """Finds the best available trained model checkpoint."""
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        # Find specific checkpoint folder (e.g., checkpoint-500)
        checkpoints = [d for d in os.listdir(MODEL_DIR) if d.startswith("checkpoint")]
        if checkpoints:
            # Sort to find the latest one
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            full_path = os.path.join(MODEL_DIR, latest_checkpoint)
            print(f"[*] Found Best Model: {full_path}")
            return full_path
    
    raise FileNotFoundError(f"[!] No checkpoints found in {MODEL_DIR}. Did you run Section 3?")

# --- 1. SETUP OSC CLIENT ---
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
print(f"[*] OSC Client Ready. Sending to {OSC_IP}:{OSC_PORT}")

# --- 2. LOAD DISTILBERT MODEL ---
model_path = get_best_model()

print(f"[*] Loading {ARCHITECTURE} from disk...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DistilBERT (Not RoBERTa)
tokenizer = DistilBertTokenizer.from_pretrained(ARCHITECTURE)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()
print("[*] Model Loaded & Ready on GPU.")

# --- 3. AUDIO LISTENING LOOP ---
recognizer = sr.Recognizer()

print(f"\n[*] LISTENING on Device {MIC_INDEX}... (Speak into your mic)")
print("---------------------------------------")

try:
    # Use your specific mic index
    with sr.Microphone(device_index=MIC_INDEX) as source:
        print("Adjusting for ambient noise... (stay quiet for 1 sec)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready! Speak now.")
        
        while True:
            print("\n[.] Listening...")
            try:
                # Listen for audio (timeout after 5s if silence)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Convert Speech to Text
                text = recognizer.recognize_google(audio)
                print(f"[YOU]: '{text}'")

                # --- AI INFERENCE (DistilBERT) ---
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    # Label 1 is "Abusive"
                    toxicity_score = probs[0][1].item()

                # Print visual bar
                bar_len = int(toxicity_score * 20)
                bar = "#" * bar_len + "-" * (20 - bar_len)
                print(f"[AI]:  [{bar}] {toxicity_score:.4f}")

                # --- SEND TO TOUCHDESIGNER ---
                client.send_message("/toxicity", toxicity_score)
                client.send_message("/text", text) 

            except sr.WaitTimeoutError:
                pass 
            except sr.UnknownValueError:
                pass 
            except Exception as e:
                print(f"[!] Error: {e}")

except KeyboardInterrupt:
    print("\n[!] Stopping Inference...")