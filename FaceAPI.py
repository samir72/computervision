import os
from dotenv import load_dotenv
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection01
from azure.core.credentials import AzureKeyCredential
import io
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

def main():


    try:
        # Clear the console
        os.system('cls' if os.name=='nt' else 'clear')
        # Get configuration settings
        load_dotenv()
        endpoint = os.getenv("faceendpoint")
        key = os.getenv("facecredential")
        
        #Authenticate Face Client
        client = FaceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
        )
        
        path = "pictures/"
        # Create the directory if it doesn't exist
        create_dir(path)
        #Get file count in root directory
        file_count = get_file_count(path)
        # Read images into cache.
        files = get_file(path)
        j = 0
        for i in files:
            print(f"File : {files[j]}")
            j += 1
        
        # Specify facial features to be retrieved
        features = [FaceAttributeTypeDetection01.HEAD_POSE,
                     FaceAttributeTypeDetection01.OCCLUSION,
                     FaceAttributeTypeDetection01.ACCESSORIES]
                     
        # Store image binaries.
        images = get_image(files)
        k = 0
        for i in images:
            while k < file_count: # Run this loop for files in the root directory
                
                # Analyse the images.
                print(f"\nRead Picture files : {files[k]}")            
                detected_faces = analyse_faces(client,images[k],features)       
                # Annotate the picture
                # Check it annotation is complete.
                l=0
                #Check annotation
                # Create the directory if it doesn't exist
                directory = f"{path}annot/"
                create_dir(directory)
                filler = "face"
                if checkannotation(files,k,l,detected_faces,directory,filler):
                    print(f"{files[k]} has been annotated twice")
                k += 1
        
    except Exception as e:
        print("Error:", e) 


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
    
    
# function to run the analysis
# This function will analyze the images using the Azure Face API
# and return the results.
def analyse_faces(client,image,features):
    
    detected_faces = []
    try:
        # Get faces
        # with open(image, mode="rb") as image_data:
        #         detected_faces = client.detect(
        #             image_content=image_data.read(),
        #             detection_model=FaceDetectionModel.DETECTION01,
        #             recognition_model=FaceRecognitionModel.RECOGNITION01,
        #             return_face_id=False,
        #             return_face_attributes=features,
        #         )

        detected_faces = client.detect(
            image_content=image,
            detection_model=FaceDetectionModel.DETECTION01,
            recognition_model=FaceRecognitionModel.RECOGNITION01,
            return_face_id=False,
            return_face_attributes=features,
        )
        face_count = 0
        if len(detected_faces) > 0:
                print(len(detected_faces), 'faces detected.')
                for face in detected_faces:

                    # Get face properties
                    face_count += 1
                    print('\nFace number {}'.format(face_count))
                    print(' - Head Pose (Yaw): {}'.format(face.face_attributes.head_pose.yaw))
                    print(' - Head Pose (Pitch): {}'.format(face.face_attributes.head_pose.pitch))
                    print(' - Head Pose (Roll): {}'.format(face.face_attributes.head_pose.roll))
                    print(' - Forehead occluded?: {}'.format(face.face_attributes.occlusion["foreheadOccluded"]))
                    print(' - Eye occluded?: {}'.format(face.face_attributes.occlusion["eyeOccluded"]))
                    print(' - Mouth occluded?: {}'.format(face.face_attributes.occlusion["mouthOccluded"]))
                    print(' - Accessories:')
                    for accessory in face.face_attributes.accessories:
                        print('   - {}'.format(accessory.type))

    except Exception as e:
                print(f"Could not read {image}: {e}")


    return detected_faces

def annotate_faces(file,fileinplace,detected_faces):
    print(f'\nAnnotating words of text in image...')

    try:  
        # Check if detected_text is None
        if len(detected_faces) < 0:
            print('No faces detected.')
            return
        # Prepare image for drawing
        print('\nAnnotating faces in image...')
        image = Image.open(file)
        draw = ImageDraw.Draw(image)
        color = 'lightgreen'
        try:
            font = ImageFont.truetype('arial.ttf', 15)  # Adjust font and size as needed
        except:
            font = ImageFont.load_default()  # Fallback to default font
        face_count = 0

        for face in detected_faces:
            try:
                r = face.face_rectangle
                bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
                draw.rectangle(bounding_box, outline=color, width=5)
                face_count += 1
                draw.text((r.left, r.top - 20), f'Face number {face_count}', fill=color)
            except AttributeError:
                print(f'Warning: Invalid face data at index {face_count}. Skipping.')

        # Save the annotated image
        outputfile = f"{fileinplace}"
        image.save(outputfile)
        print('  Results saved in', outputfile)
    except Exception as e:
        print(f"Error annotating face-{file} : {e}")

def checkannotation(files,k, l, detected_faces,dir,filler):
    try:

        if len(detected_faces) < 0:
            print("No faces detected'")
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
                    print(f"Annotating faces in {files[k]}...")
                    annotate_faces(files[k],fileinplace,detected_faces)

            return False
    except Exception as e:
        print(f"Error checking annotation: {e}")
        return False

if __name__ == '__main__':
    main()