import os
import json
from urllib.request import urlopen, Request
import base64
from pathlib import Path
from dotenv import load_dotenv
import io
import requests
from PIL import Image, ImageDraw, ImageFont
# Add references
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

def main(): 

    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')
        
    try: 
    
        # Get configuration settings 
        load_dotenv()
        endpoint = os.getenv("Foundry_azure_openai_createimagefromtext_endpoint")
        model_deployment =  os.getenv("Foundry_createimagefromtext_modeldeployment")
        api_version = os.getenv("Foundry_azure_openai_createimagefromtext_api_version")


        # Initialize the client
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(exclude_environment_credential=True,
                exclude_managed_identity_credential=True), 
            "https://cognitiveservices.azure.com/.default"
        )
            
        client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider
                            )
        
        img_no = 0
        # Loop until the user types 'quit'
        while True:
            # Get input text
            input_text = input("Enter the prompt (or type 'quit' to exit): ")
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue
            
            # Generate an image
            result = client.images.generate(
                model=model_deployment,
                prompt=input_text,
                n=1,
                size="1792x1024",
                quality="hd", 
                style="natural"
            )

            json_response = json.loads(result.model_dump_json())
            image_url = json_response["data"][0]["url"] 

            # save the image
            img_no += 1
            file_name = f"image_{img_no}.png"
            # Set the directory for the stored image
            # Create the directory if it doesn't exist
            image_dir = "generated-images"
            create_dir(image_dir)
            save_image (image_dir,image_url, file_name)

    except Exception as ex:
        print(ex)

# Function to save the image
# This function downloads the image from the URL and saves it to a specified directory
# The directory is created if it does not exist
# The image is saved in PNG format
# The function prints the path where the image is saved
# Note: The image URL is expected to be a direct link to the image file
def save_image (image_dir,image_url, file_name):
    
    # Initialize the image path (note the filetype should be png)
    image_path = os.path.join(image_dir, file_name)

    # Retrieve the generated image
    generated_image = requests.get(image_url).content  # download the image
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)
    print (f"Image saved as {image_path}")

#Function to create directory
def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        print(f"An error occurred: {e}")

# function to get file counts
def get_file_count(directory):
    try:
        # Count files in the directory
        file_count = 0
        for item in os.listdir(directory):
            if not item.startswith(".") and os.path.isfile(os.path.join(directory, item)):
                file_count += 1
        print(f"Number of files in {directory}: {file_count}")
    except PermissionError:
        print(f"Permission denied: Cannot access {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return file_count

# function to get file names
def get_file(path):
    filename = []
    
    # Supported image file extensions
    supported_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    
    # Traverse the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if the file has a supported image extension
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)
                try:
                    # Open the image using PIL
                    #img = Image.open(file_path)
                    #image_files.append(img)
                    filename.append(file_path)
                except Exception as e:
                    print(f"Could not open {file_path}: {e}")

    return filename
    

if __name__ == '__main__': 
    main()