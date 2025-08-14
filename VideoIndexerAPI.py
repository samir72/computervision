import os
import requests
import json
from dotenv import load_dotenv

def main():
    # Clear console (optional)
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        account_id = os.getenv("VideoIndexerAccountId")
        api_key = os.getenv("VideoIndexerKey")
        location = os.getenv("LOCATION", "trial")  # Default to trial if not set

        if not account_id or not api_key:
            raise ValueError("ACCOUNT_ID and API_KEY must be set in the .env file.")

        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')

        # Step 1: Get an access token
        token_url = f"https://api.videoindexer.ai/auth/{location}/Accounts/{account_id}/AccessToken"
        headers = {"Ocp-Apim-Subscription-Key": api_key}

        token_response = requests.get(token_url, headers=headers)
        token_response.raise_for_status()
        token = token_response.text.strip().strip('"')  # Remove extra quotes

        # Step 2: Use the token to get a list of videos
        videos_url = f"https://api.videoindexer.ai/{location}/Accounts/{account_id}/Videos"
        params = {"accessToken": token}

        videos_response = requests.get(videos_url, params=params)
        videos_response.raise_for_status()

        # Pretty print the JSON result
        videos_json = videos_response.json()
        print(json.dumps(videos_json, indent=2))

    except Exception as ex:
        # Useful, actionable error output
        print("Error:", ex)

if __name__ == "__main__":
    main()
