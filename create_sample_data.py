import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_sample_faces():
    """
    Generates synthetic sample event photos and selfies with distinct facial features
    to verify face detection, feature extraction, and cosine similarity matching out-of-the-box.
    """
    gallery_dir = "data/event_gallery"
    selfies_dir = "data/test_selfies"
    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(selfies_dir, exist_ok=True)

    def draw_person_face(draw, x, y, hair_color, eye_color, label):
        # Head / Face oval
        draw.ellipse([x-40, y-50, x+40, y+50], fill=(245, 215, 190), outline=(50, 50, 50), width=2)
        # Hair
        draw.chord([x-42, y-55, x+42, y-10], start=180, end=360, fill=hair_color)
        # Eyes
        draw.ellipse([x-20, y-15, x-8, y-5], fill=eye_color)
        draw.ellipse([x+8, y-15, x+20, y-5], fill=eye_color)
        # Nose
        draw.line([x, y-5, x, y+10], fill=(180, 130, 100), width=2)
        # Smile
        draw.arc([x-15, y+5, x+15, y+25], start=0, end=180, fill=(180, 50, 50), width=3)
        # Label text
        draw.text((x-20, y+55), label, fill=(255, 255, 255))

    # Photo 1: Group Event Photo (Alice & Bob)
    img1 = Image.new('RGB', (600, 400), (30, 41, 59))
    draw1 = ImageDraw.Draw(img1)
    draw1.text((20, 20), "Annual Tech Gala 2026 - Main Hall", fill=(200, 200, 200))
    draw_person_face(draw1, 200, 200, hair_color=(40, 30, 20), eye_color=(30, 144, 255), label="Alice")
    draw_person_face(draw1, 400, 200, hair_color=(180, 100, 40), eye_color=(30, 180, 50), label="Bob")
    img1.save(os.path.join(gallery_dir, "event_photo_01.jpg"))

    # Photo 2: Group Event Photo (Charlie & Alice)
    img2 = Image.new('RGB', (600, 400), (15, 23, 42))
    draw2 = ImageDraw.Draw(img2)
    draw2.text((20, 20), "Annual Tech Gala 2026 - Stage", fill=(200, 200, 200))
    draw_person_face(draw2, 180, 200, hair_color=(200, 50, 50), eye_color=(40, 40, 40), label="Charlie")
    draw_person_face(draw2, 420, 200, hair_color=(40, 30, 20), eye_color=(30, 144, 255), label="Alice")
    img2.save(os.path.join(gallery_dir, "event_photo_02.jpg"))

    # Photo 3: Solo Photo (Bob)
    img3 = Image.new('RGB', (500, 500), (51, 65, 85))
    draw3 = ImageDraw.Draw(img3)
    draw3.text((20, 20), "Keynote Speaker Portrait", fill=(200, 200, 200))
    draw_person_face(draw3, 250, 250, hair_color=(180, 100, 40), eye_color=(30, 180, 50), label="Bob")
    img3.save(os.path.join(gallery_dir, "event_photo_03.jpg"))

    # Selfie Query 1 (Alice Selfie)
    selfie_alice = Image.new('RGB', (400, 400), (71, 85, 105))
    draw_s1 = ImageDraw.Draw(selfie_alice)
    draw_person_face(draw_s1, 200, 200, hair_color=(40, 30, 20), eye_color=(30, 144, 255), label="Alice")
    selfie_alice.save(os.path.join(selfies_dir, "selfie_alice.jpg"))

    # Selfie Query 2 (Bob Selfie)
    selfie_bob = Image.new('RGB', (400, 400), (30, 50, 70))
    draw_s2 = ImageDraw.Draw(selfie_bob)
    draw_person_face(draw_s2, 200, 200, hair_color=(180, 100, 40), eye_color=(30, 180, 50), label="Bob")
    selfie_bob.save(os.path.join(selfies_dir, "selfie_bob.jpg"))

    print("Generated sample event gallery photos & attendee query selfies.")

if __name__ == "__main__":
    create_sample_faces()
