import os
import cv2
import numpy as np
from keras.models import load_model
import cvzone

# Disable scientific notation
np.set_printoptions(suppress=True)

# Load model dan label
model = load_model("model/keras_Model.h5", compile=False)
class_names = open("model/labels.txt", "r").readlines()
img_background = cv2.imread('bg.png')

# Kamera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load dan resize arrow.png
arrow_path = "arrow.png"
img_arrow = None
if os.path.exists(arrow_path):
    img = cv2.imread(arrow_path, cv2.IMREAD_UNCHANGED)
    new_width = 80
    aspect_ratio = img.shape[0] / img.shape[1]
    new_height = int(new_width * aspect_ratio)
    img_arrow = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Load dan resize gambar waste
imgWastelist = [None]
for i in range(1, 14):  # 1.png sampai 13.png
    path = os.path.join("waste", f"{i}.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        new_width = 400
        aspect_ratio = img.shape[0] / img.shape[1]
        new_height = int(new_width * aspect_ratio)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        imgWastelist.append(resized)
    else:
        imgWastelist.append(None)

# Load dan resize gambar bin
imgBinlist = [None]
for i in range(1, 5):  # 1.png sampai 4.png
    path = os.path.join("bin", f"{i}.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        new_width = 200
        aspect_ratio = img.shape[0] / img.shape[1]
        new_height = int(new_width * aspect_ratio)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        imgBinlist.append(resized)
    else:
        imgBinlist.append(None)

while True:
    ret, image = camera.read()
    if not ret:
        continue

    # Resize kamera
    display_frame = cv2.resize(image, (680, 480))

    # Input model
    model_input = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    model_input = np.asarray(model_input, dtype=np.float32).reshape(1, 224, 224, 3)
    model_input = (model_input / 127.5) - 1

    # Prediksi
    prediction = model.predict(model_input, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Ekstrak label number
    try:
        label_number = int(class_name.split()[0])
    except ValueError:
        label_number = 0

    # Tampilkan teks di frame kamera
    label_text = f"{class_name[2:]}: {confidence_score*100:.2f}%"
    cv2.putText(display_frame, label_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tempelkan kamera ke background
    background_with_camera = img_background.copy()
    background_with_camera[140:140+480, 135:135+680] = display_frame

    # Tempelkan waste image
    if 1 <= label_number <= 13 and imgWastelist[label_number] is not None:
        background_with_camera = cvzone.overlayPNG(
            background_with_camera,
            imgWastelist[label_number],
            (909, 127)
        )

    # Tampilkan arrow + bin sesuai kategori label
    if img_arrow is not None:
        if 1 <= label_number <= 4 and imgBinlist[1] is not None:
            background_with_camera = cvzone.overlayPNG(background_with_camera, img_arrow, (1080, 370))
            background_with_camera = cvzone.overlayPNG(background_with_camera, imgBinlist[1], (1020, 450))
        elif 5 <= label_number <= 7 and imgBinlist[2] is not None:
            background_with_camera = cvzone.overlayPNG(background_with_camera, img_arrow, (1080, 370))
            background_with_camera = cvzone.overlayPNG(background_with_camera, imgBinlist[2], (1020, 450))
        elif 8 <= label_number <= 10 and imgBinlist[3] is not None:
            background_with_camera = cvzone.overlayPNG(background_with_camera, img_arrow, (1080, 370))
            background_with_camera = cvzone.overlayPNG(background_with_camera, imgBinlist[3], (1020, 450))
        elif 11 <= label_number <= 13 and imgBinlist[4] is not None:
            background_with_camera = cvzone.overlayPNG(background_with_camera, img_arrow, (1080, 370))
            background_with_camera = cvzone.overlayPNG(background_with_camera, imgBinlist[4], (1020, 450))

    # Tampilkan hasil akhir
    cv2.imshow("Output", background_with_camera)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()