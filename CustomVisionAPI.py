import os
import time
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
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

        endpoint = os.getenv("Custvisiontrainendpoint")
        key = os.getenv("Custvisiontraincredential")
        projectId = os.getenv("Custvisiontrainprojectid")
        prediction_endpoint = os.getenv("CustvisionPredictionEndpoint")
        prediction_key = os.getenv("CustvisionPredictioncredential")
        model_name = os.getenv("ModelName")
        model_name_code = os.getenv("ModelNamecode")
        
                # Authenticate a client for the training API
        credentials = ApiKeyCredentials(in_headers={"Training-key": key})

        # Authenticate CustomVision Client
        training_client = CustomVisionTrainingClient(endpoint,credentials)
        # Get the Custom Vision project
        custom_vision_project = training_client.get_project(projectId)
        
        path = "training-images/more-training-images/"
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
        
        # Upload and tag images
        Upload_Images(path, training_client, custom_vision_project)
                     
        # Train the model
        
        Train_Model(training_client, custom_vision_project,prediction_endpoint, prediction_key, projectId, model_name_code)

        # Test the model
        testpath = "test-images"
        Test_Model(testpath, prediction_endpoint, prediction_key, projectId, model_name)
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
    

def Upload_Images(folder, training_client, custom_vision_project):
    print("Uploading images...")
    tags = training_client.get_tags(custom_vision_project.id)
    for tag in tags:
        print(tag.name)
        for image in os.listdir(os.path.join(folder,tag.name)):
            image_data = open(os.path.join(folder,tag.name,image), "rb").read()
            training_client.create_images_from_data(custom_vision_project.id, image_data, [tag.id])



def Train_Model(training_client, custom_vision_project,prediction_endpoint, prediction_key, projectId, model_name_code):
    try:

        print("Training ...")
        iteration = training_client.train_project(custom_vision_project.id)
        while (iteration.status != "Completed"):
            iteration = training_client.get_iteration(custom_vision_project.id, iteration.id)
            print (iteration.status, '...')
            time.sleep(5)
        print ("Model trained!")
        # Publish the model
        training_client.publish_iteration(custom_vision_project.id, iteration.id, model_name_code, prediction_endpoint, prediction_key)
        print("Model published!")
    except Exception as e:
        print(f"Error during training: {e}")

def Test_Model(testpath, prediction_endpoint, prediction_key, projectId, model_name_code):
    try:
        # Authenticate a client for the prediction API
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)
        #Test the model
        print("Classifying images...")
        # Ensure the test images directory exists
        if not os.path.exists(testpath):
            print(f"Test images directory '{testpath}' does not exist.")
            return
        # Classify test images
        for image in os.listdir('test-images'):
            image_data = open(os.path.join('test-images',image), "rb").read()
            results = prediction_client.classify_image(projectId, model_name_code, image_data)

            # Loop over each label prediction and print any with probability > 50%
            for prediction in results.predictions:
                if prediction.probability > 0.5:
                    print(image, ': {} ({:.0%})'.format(prediction.tag_name, prediction.probability))
    except Exception as ex:
        print(ex)
if __name__ == '__main__':
    main()