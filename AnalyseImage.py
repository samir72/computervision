import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from PIL import Image 
import io

def main():


    try:
        # Clear the console
        os.system('cls' if os.name=='nt' else 'clear')
        # Get configuration settings
        load_dotenv()
        endpoint = os.getenv("endpoint")
        key = os.getenv("credential")
        client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
        )
        
        # Read images into cache.
        path = "images/"
        files = get_file(path)
        j = 0
        for i in files:
            print(f"File : {files[j]}")
            j += 1
        
        # Store image binaries.
        images = get_image(files)
        k = 0
        for i in images:
            # Analyse the images.
            print(f"Analyse Image : {files[k]}")            
            result = analyse_image(client,images[k])
            # Get image captions
            if result.caption is not None:
                print("\nCaption:")
                print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))
                
            if result.dense_captions is not None:
                print("\nDense Captions:")
                for caption in result.dense_captions.list:
                    print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))
                #images[k].show()
                #print(f"Image : {images[k]}")
            k += 1
        
    except Exception as e:
        print("Error:", e) 

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
    
    
# function to run the analysis
# This function will analyze the images using the Azure Vision API
# and return the results.
def analyse_image(client,image):
    
    result = []
    try:
        result = client.analyze(
        image_data=image,
        visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.DENSE_CAPTIONS,
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.PEOPLE],
        )
    except Exception as e:
                print(f"Could not analyse {image}: {e}")

    return result

if __name__ == '__main__':
    main()