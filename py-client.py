import os
import io
import sys
import cv2
import base64
import requests
import random as r
import numpy as np
import matplotlib.pyplot as plt

from time import time
from PIL import Image

READ_PATH = "misc"


def breaker(num: int=50, char: str="*") -> None:
    print("\n" + num*char + "\n")


def get_image(path: str):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def show_image(image: np.ndarray, cmap: str="gnuplot2", title: str=None) -> None:
    plt.figure()
    plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


def show_images(
    image_1: np.ndarray,
    image_2: np.ndarray, 
    cmap: str="gnuplot2", 
    title_1: str=None,
    title_2: str=None
    ) -> None:

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(src=image_1, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.axis("off")
    if title_1: plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(src=image_2, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.axis("off")
    if title_2: plt.title(title_2)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


def decode_image(imageData) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    if len(image.shape) == 4:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    return header, image


def encode_image_to_base64(header: str = "image/jpeg", image: np.ndarray = None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData


def main():

    args_1: str = "--mode"
    args_2: str = "--disp"

    mode: str = "diabetes"
    filename: str = "1.jpg"
    display: bool = False

    if args_1 in sys.argv: mode = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: display = True

    if mode == "diabetes":

        # ~98 %
        payload: dict = {
            "pregnancies" : 2,
            "plasma_glucose" : 180,
            "diastolic_blood_pressure" : 74,
            "triceps_thickness" : 24,
            "serum_insulin" : 21,
            "bmi" : 23.909170,
            "diabetes_pedigree" : 1.488172308,
            "age" : 22,
        }

        response = requests.request("POST", "http://127.0.0.1:10000/infer/diabetes", json=payload)  
        breaker()
        print(f"Label : {response.json()['prediction']} ({float(response.json()['probability']):.5f})")
        breaker()
    
    elif mode == "heart":

        # ~50 %
        payload: dict = {
            "age" : 60,
            "gender" : 2,
            "height" : 155,
            "weight" : 80,
            "ap_high" : 135,
            "ap_low" : 85,
            "cholestrol" : 2,
            "glucose" : 1,
            "smoke" : 0,
            "alcohol" : 1,
            "active" : 0,
        }

        response = requests.request("POST", "http://127.0.0.1:10000/infer/cardiovascular-disease", json=payload)  
        breaker()
        print(f"Label : {response.json()['prediction']} ({float(response.json()['probability']):.5f})")
        breaker()

    elif mode == "pneumonia":
        filename = sys.argv[sys.argv.index(args_1) + 2]
        image = get_image(os.path.join(READ_PATH, filename))
        imageData = encode_image_to_base64(image=image)

        payload: dict = {
            "imageData" : imageData
        }

        response = requests.request("POST", "http://127.0.0.1:10000/infer/pneumonia", json=payload)  

        if response.status_code == 200:
            breaker()
            print(f"Probability : {float(response.json()['probability']):.5f}")
            breaker()
            if display: show_image(image=image, title=f"Probability : {float(response.json()['probability']):.5f}")
        else:
            print(f"{response.status_code}, {response.reason}")
    
    elif mode == "tuberculosis":
        filename = sys.argv[sys.argv.index(args_1) + 2]

        image = get_image(os.path.join(READ_PATH, filename))
        imageData = encode_image_to_base64(image=image)

        payload: dict = {
            "imageData" : imageData
        }

        response = requests.request("POST", "http://127.0.0.1:10000/infer/tuberculosis", json=payload)  
        
        if response.status_code == 200:
            breaker()
            print(f"Probability : {float(response.json()['probability']):.5f}")
            breaker()
            if display: show_image(image=image, title=f"Probability : {float(response.json()['probability']):.5f}")
        else:
            print(f"{response.status_code}, {response.reason}")
    

    elif mode == "brain-mri":
        filename = sys.argv[sys.argv.index(args_1) + 2]

        image = get_image(os.path.join(READ_PATH, filename))
        imageData = encode_image_to_base64(image=image)

        payload: dict = {
            "imageData" : imageData
        }

        response = requests.request("POST", "http://127.0.0.1:10000/infer/brain-mri", json=payload)  
        
        if response.status_code == 200:
            _, result_image = decode_image(response.json()["imageData"])
            if display: show_images(image, result_image, title_1="Original", title_2="Segmented Result")
        else:
            print(f"{response.status_code}, {response.reason}")


if __name__ == "__main__":
    sys.exit(main() or 0)