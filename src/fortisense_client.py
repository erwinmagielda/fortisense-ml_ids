import os
import socket
import pickle

import pandas as pd

# ============================================================
# FortiSense - Part 5: Real-Time IDS Client
#
# - Connects to the IDS server.
# - Loads samples from KDDTest.csv.
# - Sends feature rows one by one to simulate live traffic.
# - Receives and prints predictions from the server.
# ============================================================

project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")

testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")

HOST = "127.0.0.1"
PORT = 5050

print("[*] FortiSense IDS Client - Loading test dataset...")

testing_dataframe = pd.read_csv(testing_dataset_path)

# Remove label columns so only features are sent
feature_dataframe = testing_dataframe.drop(columns=["label", "attack_type"])

print(f"[+] Test dataset loaded with {len(feature_dataframe)} rows.")
print()

print(f"[*] Connecting to IDS server at {HOST}:{PORT}...")

with socket.create_connection((HOST, PORT)) as client_socket:
    print("[+] Connected to IDS server.")
    print("[*] Sending first 20 samples as simulated network traffic...\n")

    for index in range(20):
        sample_row = feature_dataframe.iloc[index].to_dict()

        # Encode sample as bytes
        payload = pickle.dumps(sample_row)
        client_socket.send(payload)

        # Receive prediction from server
        prediction = client_socket.recv(4096).decode()
        print(f"Sample {index + 1:02d} - Prediction: {prediction}")

    print("\n[✓] Finished sending samples. Closing client connection.")
