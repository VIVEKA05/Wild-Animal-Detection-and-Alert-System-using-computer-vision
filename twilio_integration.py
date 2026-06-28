import pygame
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
from datetime import datetime
import os
import threading
from twilio.rest import Client

# =====================================
# SETTINGS
# =====================================

CLASS_NAMES = [
    "elephant",
    "tiger",
    "wild_boar"
]

UNKNOWN_THRESHOLD = 90

# =====================================
# ANIMAL SOUND FILES
# =====================================

ELEPHANT_SOUND = r"C:\Users\Vivek\Documents\Animalnew\elephant.mp3"
TIGER_SOUND = r"C:\Users\Vivek\Documents\Animalnew\tiger.mp3"
WILD_BOAR_SOUND = r"C:\Users\Vivek\Documents\Animalnew\wildboar.mp3"

# =====================================
# TWILIO SETTINGS
# =====================================

TWILIO_ACCOUNT_SID = "AC47ebe3dec8628c7c5c67f5a1a95e5a14"
TWILIO_AUTH_TOKEN  = "6f5d6db436679d1a54881dde1a20f001"

TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"
TWILIO_WHATSAPP_TO   = "whatsapp:+916360264132"

SNAPSHOT_SAVE_PATH = r"C:\Users\Vivek\Documents\Animalnew\snapshot.jpg"

WHATSAPP_COOLDOWN = 30

# =====================================
# LOAD MODEL
# =====================================

model = models.mobilenet_v3_large(weights=None)

num_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    num_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        "best_animal_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model Loaded Successfully")

# =====================================
# AUDIO INIT
# =====================================

pygame.mixer.init()

current_sound = None
last_label = ""

# =====================================
# TWILIO INIT
# =====================================

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

last_alert_time = 0

# =====================================
# IMAGE TRANSFORM
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================
# SOUND FUNCTIONS
# =====================================

def play_animal_sound(sound_file):

    global current_sound

    if current_sound != sound_file:

        pygame.mixer.music.stop()

        pygame.mixer.music.load(sound_file)

        pygame.mixer.music.play(-1)

        current_sound = sound_file


def stop_sound():

    global current_sound

    pygame.mixer.music.stop()

    current_sound = None

# =====================================
# TWILIO WHATSAPP ALERT  ← ONLY THIS FUNCTION IS CHANGED
# =====================================

def send_whatsapp_alert(animal_label, confidence_val, frame_snapshot):

    def _send():

        try:

            # Save snapshot locally
            cv2.imwrite(SNAPSHOT_SAVE_PATH, frame_snapshot)

            now_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            message_body = (
                f"ANIMAL ALERT\n\n"
                f"Animal Detected : {animal_label.upper()}\n"
                f"Confidence      : {confidence_val:.2f}%\n"
                f"Time            : {now_str}\n\n"
                f"Please take necessary precautions!"
            )

            # ← FIXED: removed media_url entirely so plain text always delivers
            msg = twilio_client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                to=TWILIO_WHATSAPP_TO,
                body=message_body
            )

            print(f"[TWILIO] WhatsApp alert sent | SID: {msg.sid} | Animal: {animal_label}")

        except Exception as e:

            print(f"[TWILIO ERROR] {e}")

    thread = threading.Thread(target=_send, daemon=True)

    thread.start()

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("Press Q to Exit")

prev_time = time.time()

fullscreen = False

WINDOW_NAME = "AI Animal Detection System"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

# =====================================
# MAIN LOOP
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)

    input_tensor = transform(
        pil_image
    ).unsqueeze(0)

    # =====================================
    # PREDICTION
    # =====================================

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    confidence, index = torch.max(
        probabilities,
        dim=0
    )

    confidence = confidence.item() * 100

    if confidence >= UNKNOWN_THRESHOLD:

        label = CLASS_NAMES[
            index.item()
        ]

        status_color = (0, 255, 0)

    else:

        label = "Unknown"

        status_color = (0, 0, 255)

    # =====================================
    # SOUND CONTROL
    # =====================================

    if label != last_label:

        if label == "Unknown":

            stop_sound()

        elif label.lower() == "elephant":

            play_animal_sound(
                ELEPHANT_SOUND
            )

        elif label.lower() == "tiger":

            play_animal_sound(
                TIGER_SOUND
            )

        elif label.lower() == "wild_boar":

            play_animal_sound(
                WILD_BOAR_SOUND
            )

        last_label = label

    # =====================================
    # TWILIO WHATSAPP ALERT
    # =====================================

    if label != "Unknown":

        current_time_check = time.time()

        if current_time_check - last_alert_time >= WHATSAPP_COOLDOWN:

            send_whatsapp_alert(
                label,
                confidence,
                frame.copy()
            )

            last_alert_time = current_time_check

    # =====================================
    # FPS
    # =====================================

    current_time = time.time()

    fps = int(
        1 / max(
            current_time - prev_time,
            0.0001
        )
    )

    prev_time = current_time

    # =====================================
    # CAMERA BORDER
    # =====================================

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1]-1, frame.shape[0]-1),
        (0, 180, 255),
        3
    )

    # =====================================
    # DASHBOARD
    # =====================================

    h, w = frame.shape[:2]

    panel_width = 420

    dashboard = np.zeros(
        (h, panel_width, 3),
        dtype=np.uint8
    )

    dashboard[:] = (28, 28, 28)

    # =====================================
    # HEADER
    # =====================================

    cv2.rectangle(
        dashboard,
        (0, 0),
        (panel_width, 90),
        (45, 45, 45),
        -1
    )

    cv2.putText(
        dashboard,
        "AI ANIMAL",
        (25, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        dashboard,
        "DETECTION SYSTEM",
        (25, 75),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (0, 200, 255),
        2
    )

    # =====================================
    # DETECTION CARD
    # =====================================

    cv2.rectangle(
        dashboard,
        (15, 110),
        (405, 220),
        (45, 45, 45),
        -1
    )

    cv2.putText(
        dashboard,
        "DETECTED ANIMAL",
        (30, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        2
    )

    cv2.putText(
        dashboard,
        label.upper(),
        (30, 195),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        status_color,
        3
    )

    # =====================================
    # CONFIDENCE CARD
    # =====================================

    cv2.rectangle(
        dashboard,
        (15, 240),
        (405, 370),
        (45, 45, 45),
        -1
    )

    cv2.putText(
        dashboard,
        "CONFIDENCE",
        (30, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        2
    )

    cv2.putText(
        dashboard,
        f"{confidence:.2f}%",
        (30, 325),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.rectangle(
        dashboard,
        (30, 335),
        (370, 355),
        (80, 80, 80),
        -1
    )

    progress_width = int(
        (confidence / 100) * 340
    )

    cv2.rectangle(
        dashboard,
        (30, 335),
        (30 + progress_width, 355),
        status_color,
        -1
    )

    # =====================================
    # TOP PREDICTIONS
    # =====================================

    cv2.rectangle(
        dashboard,
        (15, 390),
        (405, 560),
        (45, 45, 45),
        -1
    )

    cv2.putText(
        dashboard,
        "TOP PREDICTIONS",
        (30, 425),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2
    )

    top_probs, top_idx = torch.topk(
        probabilities,
        min(3, len(CLASS_NAMES))
    )

    y = 470

    for p, idx in zip(
        top_probs,
        top_idx
    ):

        prediction_text = (
            f"{CLASS_NAMES[idx.item()]} : "
            f"{p.item()*100:.2f}%"
        )

        cv2.putText(
            dashboard,
            prediction_text,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y += 40

    # =====================================
    # STATUS BAR
    # =====================================

    cv2.rectangle(
        dashboard,
        (0, h - 80),
        (panel_width, h),
        (45, 45, 45),
        -1
    )

    cv2.putText(
        dashboard,
        f"FPS : {fps}",
        (20, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    now = datetime.now()

    time_string = now.strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    cv2.putText(
        dashboard,
        time_string,
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )

    # =====================================
    # COMBINE
    # =====================================

    combined = np.hstack(
        (
            frame,
            dashboard
        )
    )

    cv2.imshow(
        WINDOW_NAME,
        combined
    )

    key = cv2.waitKey(1) & 0xFF

    # Q = EXIT
    if key == ord('q'):
        break

    # F = FULLSCREEN
    if key == ord('f'):

        fullscreen = not fullscreen

        if fullscreen:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )

        else:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )

# =====================================
# CLEANUP
# =====================================

stop_sound()

pygame.mixer.quit()

cap.release()

cv2.destroyAllWindows()