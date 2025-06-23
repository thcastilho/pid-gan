import os
import subprocess
import argparse

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                print(f"Pulando {filename} (já existe no output).")
                continue

            command = [
                'python', 'inference.py',
                '--inputs', input_path,
                '--output', output_path,
                '--device', 'cuda:0'
            ]

            print(f"Inferindo {filename}...")
            subprocess.run(command, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
