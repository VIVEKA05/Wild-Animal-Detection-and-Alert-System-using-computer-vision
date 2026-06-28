import pygame
import time

pygame.mixer.init()

pygame.mixer.music.load(
    r"C:\Users\Vivek\Documents\Animalnew\elephant.mp3"
)

pygame.mixer.music.play()

print("Playing Audio...")

while pygame.mixer.music.get_busy():
    time.sleep(1)

print("Finished")