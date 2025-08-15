import os
from urllib.request import urlopen, Request
import base64
from pathlib import Path
from dotenv import load_dotenv
import io
from PIL import Image, ImageDraw, ImageFont
# Add references
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

def main(): 

    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')
        
    try: 
    
        # Get configuration settings 
        load_dotenv()
        project_endpoint = os.getenv("Foundry_visionenabledchatapp_endpoint")
        model_deployment =  os.getenv("Foundry_visionenabledchatapp_modeldeployment")


        # Initialize the project client
        project_client = AIProjectClient(            
                 credential=DefaultAzureCredential(
                     exclude_environment_credential=True,
                     exclude_managed_identity_credential=True
                 ),
                 endpoint=project_endpoint,
             )

        
        # Get a chat client
        openai_client = project_client.get_openai_client(api_version="2024-10-21")



        # Initialize prompts
        system_message = "You are an AI assistant in a grocery store that sells fruit. You provide detailed answers to questions about produce."
        prompt = ""

        # Loop until the user types 'quit'
        l = 0 # Counter to track images processed to maintain the context
        file_count = 0 # Counter to track the number of files in the directory
        k = 0 # Counter to track the number of images processed
        j = 0
        Images = [] # List to store image binaries
        while True:

                prompt = input("\nAsk a question about the image\n(or type 'quit' to exit)\n")
                if prompt.lower() == "quit":
                    break
                elif len(prompt) == 0:
                    print("Please enter a question.\n")
                else:
                    print("Getting a response ...\n")

            #  # Get a response to an image input from a URL (uncomment if needed)
            #     image_url = "https://github.com/MicrosoftLearning/mslearn-ai-vision/raw/refs/heads/main/Labfiles/gen-ai-vision/orange.jpeg"
            #     image_format = "jpeg"
            #     request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            #     image_data = base64.b64encode(urlopen(request).read()).decode("utf-8")
            #     data_url = f"data:image/{image_format};base64,{image_data}"

                # response = openai_client.chat.completions.create(
                #      model=model_deployment,
                #      messages=[
                #          {"role": "system", "content": system_message},
                #          { "role": "user", "content": [  
                #              { "type": "text", "text": prompt},
                #              { "type": "image_url", "image_url": {"url": data_url}}
                #          ] } 
                #      ]
                # )
                # print(response.choices[0].message.content)             
                
                # Get a response to image input stored in a local directory
              
                # Define the path to the directory containing images

                if l == 0 and file_count == 0: # Run this loop for files in the root directory
                    image_path ="Chat-app-images/"
                    mime_type = "image/jpeg"
                    # Create the directory if it doesn't exist
                    create_dir(image_path)
                    #Get file count in root directory
                    file_count = get_file_count(image_path)
                    # Read images into cache.
                    files = get_file(image_path)
                    j = 0
                    for i in files:
                        print(f"File : {files[j]}")
                        j += 1
                    # Store image binaries.
                    images = get_image(files)
                    #k = 0 #Counter to read all the images in the directory.
                    l = 1 # First check with images stored in the cache.

                for i in images:
                    while k < file_count: # Run this loop for files in the root directory
                
                        # Analyse the images.
                        print(f"\nRead Picture files : {files[k]}") 
                        # Encode the image file 
                        base64_encoded_data = base64.b64encode(images[k]).decode('utf-8')
                        # Include the image file data in the prompt
                        data_url = f"data:{mime_type};base64,{base64_encoded_data}"
                        response = openai_client.chat.completions.create(
                        model=model_deployment,
                        messages=[
                            {"role": "system", "content": system_message},
                            { "role": "user", "content": [  
                                { "type": "text", "text": prompt},
                                { "type": "image_url", "image_url": {"url": data_url}}
                            ] } 
                                ]
                                                                        )
                        print(response.choices[0].message.content)           
                        # Increment the counter
                        k += 1
                if l == 1:
                    print(f"Prompt with no images and maintained context.")
                    prompt = input("\nAsk a question about the image\n(or type 'quit' to exit)\n")
                    if prompt.lower() == "quit":
                        break
                    elif len(prompt) == 0:
                        print("Please enter a question.\n")
                    else:
                        print("Getting a response ...\n")
        
                    response = openai_client.chat.completions.create(
                        model=model_deployment,
                        messages=[
                            {"role": "system", "content": system_message},
                            { "role": "user", "content": [  
                                { "type": "text", "text": prompt}
                                #{ "type": "image_url", "image_url": {"url": data_url}}
                            ] } 
                                ]
                                                                        )
                    print(response.choices[0].message.content)  
                    images = [] # Reset the list of images    
    except Exception as ex:
        print(ex)

#Function to create directory

def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        print(f"An error occurred: {e}")

# function to get file names
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
    
# function to read binaries
def get_image(files):
    images = []
    imgs = []
            
    # Traverse the list
    for image in files:
        if not image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"Skipping {image}: not a recognized image format")
            continue
        try:
            with open(image, "rb") as f:
                image_data = f.read()
                # Try to load the image data to verify it's valid
                img = Image.open(io.BytesIO(image_data))
                img.verify()  # Verify image integrity (checks for corruption)
                img = Image.open(io.BytesIO(image_data))  # Reopen for further checks
                img.load()  # Ensure the image can be fully loaded
                # Optional: Check image properties
                if img.size[0] <= 0 or img.size[1] <= 0:
                    print(f"Skipping {image}: invalid dimensions")
                    continue
                images.append(image_data)
                imgs.append(img)
                print(f"Successfully validated {image} and its size is {img.size}")
        except Exception as e:
                    print(f"Could not read {image}: {e}")

    return images

if __name__ == '__main__': 
    main()