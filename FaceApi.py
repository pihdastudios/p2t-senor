import time
import sys
from azure.cognitiveservices.vision.face import FaceClient
from typing import List, Tuple, Optional
from azure.cognitiveservices.vision.face.models import Person, TrainingStatusType
from azure.cognitiveservices.vision.face.models._models_py3 import APIErrorException
import azure.cognitiveservices.vision.face.models as azure_face_models
import os
from PIL import Image


# Return a list of face IDs for an image
def extract_faces_from_image(face_client: FaceClient, path: str) -> Tuple:
    """Takes one image and lets Azure find the faces in the image.
    No identification is done here, the face IDs are specific to this one image
    Args:
        :param face_client:
        :param image:
    Returns:
        List[str]: List of face IDs. These can then be matched against e.g. a person group

    """
    image = open(path, "r+b")
    face_ids = []
    faces = face_client.face.detect_with_stream(image)
    for face in faces:
        face_ids.append(face.face_id)
    return face_ids, faces


def train_person_group(face_client: FaceClient, person_group_id: str) -> None:
    """When enough pictures are sent to Azure for a person group, this methods triggers the training and prints the status
    Args:
        person_group_id (str): the person group
    """
    face_client.person_group.train(person_group_id)

    # Print information on status
    while True:
        training_status = face_client.person_group.get_training_status(person_group_id)
        print("Training status: {}.".format(training_status.status))
        if training_status.status is TrainingStatusType.succeeded:
            break
        elif training_status.status is TrainingStatusType.failed:
            sys.exit('Training the person group has failed.')
        time.sleep(2)


def add_images_to_person_group(face_client: FaceClient, person_group_id: str, input_path: str,
                               resize_path: str) -> None:
    """Go through a folder where training images for different persons are located.
    Resize the images and add them to a person group in Azure.
    Structure for input path:
    input_path
    - name person 1
    - name person 2
        - image 1
        - image 2
    Args:
        input_path (str): Path where the input images are located, structure above
        resize_path (str): [description]
        person_group_id (str): [description]
    """
    # Send images of each person to the service
    for person_name in os.listdir(input_path):
        print(f"\nProcessing images for {person_name}\n")
        current_person_group = face_client.person_group_person.create(person_group_id, person_name)
        current_path = input_path + person_name
        images_current_person = [current_path + "/" + f for f in os.listdir(current_path)]
        time.sleep(3)

        # Create the folder to put the resized images if it does not exist
        resized_path = resize_path + person_name
        if not os.path.exists(resized_path):
            os.mkdir(resized_path)

        # Go through images for a person, resize them and send to API
        for image in images_current_person:
            image_name = os.path.basename(image)
            target_path = resized_path + "/" + image_name

            resize_image(image, target_path, (2000, 2000))

            with open(target_path, "r+b") as im:
                try:
                    face_client.person_group_person.add_face_from_stream(person_group_id,
                                                                         current_person_group.person_id, im)
                except azure_face_models.APIErrorException as e:
                    print("API error for person " + person_name + " for image " + image_name)
                    print(e)
            time.sleep(3)


def resize_image(source_path, target_path, resolution: tuple) -> None:
    """Resize an image and save it to the target path. This only works for iPhone images at the moment as these save the rotation in exif key 274
    TODO: also turn non-iPhone images
    Args:
        source_path ([type]): [description]
        target_path ([type]): [description]
        resolution (tuple): [description]
    """

    image = Image.open(source_path)
    e = image._getexif()
    if e is not None:
        exif = dict(e.items())
        try:
            orientation = exif[274]
            if orientation == 3:
                image = image.transpose(Image.ROTATE_180)
            elif orientation == 6:
                image = image.transpose(Image.ROTATE_270)
            elif orientation == 8:
                image = image.transpose(Image.ROTATE_90)
        except KeyError:
            print(f"No exif key 274 found in image {os.path.basename(source_path)}")
    image.thumbnail(resolution)
    image.save(target_path)


# Resolve a list of face IDs to the actual persons
def resolve_face_ids(face_client: FaceClient, face_ids: List[str], person_group_id: str) -> Optional[List[Person]]:
    """For a list of face IDs, let Azure check if there are matches in a person group
    Args:
        face_client (FaceClient): The azure face client
        face_ids (List[str]): List of face IDs
        person_group_id (str): name of the person group
    Returns:
        Optional[List[Person]]: If there were matches, return a list of Person objects.
        Will fail if there are no matches or more than 10 faces in the image
    """
    try:
        results = face_client.face.identify(face_ids, person_group_id)
        return results
    except APIErrorException:
        print(
            f"Error when resolving faces. This probably means that the person group you specified does not exist. Please check the spelling and case sensitivity.")
        sys.exit()
    except Exception as e:
        print(f"Error when resolving faces for Face IDs {str(face_ids)}: {e}")
        time.sleep(3)
