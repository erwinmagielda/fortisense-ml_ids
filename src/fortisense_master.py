import os
import sys
import subprocess

# ============================================================
# FortiSense - Master Orchestrator
#
# Runs the full coursework pipeline in order:
#   1. EDA (fortisense_eda.py)
#   2. Classical ML (fortisense_ml.py)
#   3. Neural Network (fortisense_nn.py)
#   4. Model comparison (fortisense_compare.py)
#   5. Optionally starts the IDS server (fortisense_server.py)
#
# The client (fortisense_client.py) is run separately in a
# different terminal, once the server is up.
# ============================================================


def run_module(script_name, description):
    """
    Runs a Python module as a separate process and prints
    a clear status message around it.
    """
    script_directory = os.path.dirname(__file__)
    script_path = os.path.join(script_directory, script_name)

    print(f"\n[*] FortiSense Master - Running {description} ({script_name})...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"[+] Completed: {description}")
    except subprocess.CalledProcessError as error:
        print(f"[!] Error while running {description}: {error}")
        sys.exit(1)


def main():
    print("===============================================")
    print(" FortiSense - AI-based IDS Coursework Pipeline ")
    print("===============================================\n")

    # 1. EDA
    run_module("fortisense_eda.py", "Part 1 - Exploratory Data Analysis")

    # 2. Classical ML (Random Forest + Linear SVM)
    run_module("fortisense_ml.py", "Part 2 - Classical ML Models")

    # 3. Neural Network (PyTorch)
    run_module("fortisense_nn.py", "Part 3 - Neural Network Model")

    # 4. Model comparison (RF vs SVM vs NN)
    run_module("fortisense_compare.py", "Part 4 - Model Comparison")

    print("\n[✓] FortiSense Master - Core pipeline completed.")
    print("[*] At this point, all models are trained and saved to the 'models' directory.")
    print()

    # 5. Ask whether to start the IDS server
    user_choice = input("Start FortiSense IDS server now? (Y/N): ").strip().lower()

    if user_choice == "y":
        print("\n[*] Starting FortiSense IDS Server (Part 5 - Real-Time IDS Prototype)...")
        print("[*] The server will run in this terminal.")
        print("[*] Open a second terminal and run:")
        print("    python fortisense_client.py")
        print()

        script_directory = os.path.dirname(__file__)
        server_script_path = os.path.join(script_directory, "fortisense_server.py")

        try:
            # This will block and keep the server running
            subprocess.run([sys.executable, server_script_path], check=True)
        except subprocess.CalledProcessError as error:
            print(f"[!] Error while running IDS server: {error}")
            sys.exit(1)
    else:
        print("\n[+] Skipping IDS server start.")
        print("    You can start it manually later with:")
        print("    python fortisense_server.py")
        print()

    print("[✓] FortiSense Master - Finished.")


if __name__ == "__main__":
    main()
