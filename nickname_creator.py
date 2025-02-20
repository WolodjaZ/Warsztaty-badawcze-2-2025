import hashlib
import getpass
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Encode a name with password")
    parser.add_argument("--name", "-n", type=str, required=True, help="Name to encode")
    return parser.parse_args()

def encode_name(name: str, password: str) -> str:
    # Combine name and password and generate hash
    combined = f"{name}:{password}"
    
    # Take first 10 chars of hex hash
    hash_hex = hashlib.sha256(combined.encode()).hexdigest()[:10]
    
    # Make it even shorter by taking parts of the hash to create a 6-char result
    short_hash = hash_hex[:6]
    
    return short_hash

def main():
    args = get_args()
    try:
        # Get password securely (won't show on screen)
        password = getpass.getpass("Enter password: ")
        
        # Generate encoded result
        result = encode_name(args.name, password)
        print(f"Nickname: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()