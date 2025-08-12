import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

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
        
        
        path = "images_text/"
        #Get file count in root directory
        file_count = get_file_count(path)
        # Read images into cache.
        files = get_file(path)
        j = 0
        for i in files:
            print(f"File : {files[j]}")
            j += 1
        
        # Store image binaries.
        images = get_image(files)
        k = 0
        for i in images:
            while k < file_count: # Run this loop for files in the root directory
                
                # Analyse the images.
                print(f"\nRead text : {files[k]}")            
                result = analyse_image(client,images[k])
                # Print the text
                if result.read is not None:
                    print("\nText:")
        
                    for line in result.read.blocks[0].lines:
                        print(f" {line.text}")        
                # Annotate the text in the image
                # Check it annotation is complete.
                l=0
                #Check line annotation
                # Create the directory if it doesn't exist
                directory_line = f"{path}annot-line/"
                if not os.path.exists(directory_line):
                    os.makedirs(directory_line)
                if checkannotation(files,k,l,result,directory_line,'lines'):
                    print(f"{files[k]} has been annotated twice for lines.")
                #Check word annotation
                # Create the directory if it doesn't exist
                directory_word = f"{path}annot-word/"
                if not os.path.exists(directory_word):
                    os.makedirs(directory_word)
                if checkannotation(files,k,l,result,directory_word,'words'):
                    print(f"Words of text in {files[k]} has been annotated twice for words.")
                k += 1
        
    except Exception as e:
        print("Error:", e) 


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
    
    
# function to run the analysis
# This function will analyze the images using the Azure Vision API
# and return the results.
def analyse_image(client,image):
    
    result = []
    try:
        result = client.analyze(
        image_data=image,
        visual_features=[
        VisualFeatures.READ]
        )
    except Exception as e:
                print(f"Could not analyse {image}: {e}")

    return result

def annotate_lines(file,fileinplace,detected_text):
    print(f'\nAnnotating lines of text in image...')

    try:  
        # Check if detected_text is None
        #if detected_text is None or detected_text.blocks is None or len(detected_text.blocks) == 0:
        if detected_text is None:
            print("No text detected in the image.")
            return
        # Prepare image for drawing
        image = Image.open(file)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        # Loop over detected text lines and annotate
        for line in detected_text.read.blocks[0].lines:
            # Get bounding polygon for each line
            r = line.bounding_polygon
            rectangle = ((r[0].x, r[0].y), (r[1].x, r[1].y), (r[2].x, r[2].y), (r[3].x, r[3].y))
            draw.polygon(rectangle, outline=color, width=3)

        # Convert image to numpy array for displaying with matplotlib
        image_np = np.array(image)
        # Display the image with annotations
        plt.imshow(image_np)
        plt.tight_layout(pad=0)

        # Save the annotated image
        textfile = f"{fileinplace}"
        fig.savefig(textfile)
        print('  Results saved in', textfile)
    except Exception as e:
        print(f"Error annotating lines-{file} : {e}")

def annotate_words(file,fileinplace,detected_text):
    print(f'\nAnnotating words of text in image...')

    try:  
        # Check if detected_text is None
        #if detected_text is None or detected_text.blocks is None or len(detected_text.blocks) == 0:
        if detected_text is None:
            print("No text detected in the image.")
            return
        # Prepare image for drawing
        image = Image.open(file)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        # Loop over detected text lines and annotate
        for line in detected_text.read.blocks[0].lines:
                for word in line.words:
                    # Get bounding polygon for each line
                    r = word.bounding_polygon
                    rectangle = ((r[0].x, r[0].y), (r[1].x, r[1].y), (r[2].x, r[2].y), (r[3].x, r[3].y))
                    draw.polygon(rectangle, outline=color, width=3)

        # Convert image to numpy array for displaying with matplotlib
        image_np = np.array(image)
        # Display the image with annotations
        plt.imshow(image_np)
        plt.tight_layout(pad=0)

        # Save the annotated image
        textfile = f"{fileinplace}"
        fig.savefig(textfile)
        print('  Results saved in', textfile)
    except Exception as e:
        print(f"Error annotating words-{file} : {e}")

def checkannotation(files,k, l, result,dir,filler):
    try:

        if result.read is None:
            print("No image has been analysed as part of annotation.")
            return False
        else:
            char1 = "/"
            char2 = "."
            #Get the file name for annotation.
            try:
                start = files[k].rindex(char1) + 1
                end = files[k].index(char2)
                if start <= end:
                    finalstringfip = files[k][start:end]
                    #print(finalstringfip)  # Output: , W
                else:
                    print("No valid substring found (char2 before char1 or adjacent).")
            except ValueError:
                print("One or both characters not found in the string.")
            for i in files:
                #Get the file name for annotation.
                try:
                    start = files[l].rindex(char1) + 1
                    end = files[l].index(char2)
                    if start <= end:
                        finalstringftc = files[l][start:end]
                    #    print(finalstringftc)  # Output: , W
                    else:
                        print("No valid substring found (char2 before char1 or adjacent).")
                except ValueError:
                    print("One or both characters not found in the string.")
                fileinplace = f"{dir}{finalstringfip}-{filler}.jpg"
                filetobechecked = f"{dir}{finalstringftc}.jpg"
                #fileinplace = f"{dir}{files[k]}-{filler}.jpg"
                #filetobechecked = f"{dir}{files[l]}"
                if fileinplace == filetobechecked:
                    print(f"{files[k]}...has been annotated")
                    break
                else:
                    l += 1                    
                # Annotate the lines of text in the image
                if  l >= len(files):
                        if filler == 'lines':
                            print(f"Annotating {filler} of text in {files[k]}...")
                            annotate_lines(files[k],fileinplace,result)
                        elif filler == 'words':
                            print(f"Annotating {filler} of text in {files[k]}...")
                            annotate_words(files[k],fileinplace,result)
            return False
    except Exception as e:
        print(f"Error checking annotation: {e}")
        return False

if __name__ == '__main__':
    main()