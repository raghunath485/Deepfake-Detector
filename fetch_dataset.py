import os
import sys
import time
import urllib.request
import urllib.error

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
AUGMENTED_DIR = os.path.join(BASE_DIR, 'archive', 'internet_dataset')
REAL_DIR = os.path.join(AUGMENTED_DIR, 'real')
FAKE_DIR = os.path.join(AUGMENTED_DIR, 'fake')

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# Curated open-access real face portraits (Wikimedia public domain / CC portraits)
REAL_FACE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/480px-President_Barack_Obama.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Alexander_Hamilton_portrait_by_John_Trumbull_1806.jpg/480px-Alexander_Hamilton_portrait_by_John_Trumbull_1806.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Abraham_Lincoln_O-77_matte_collodion_print.jpg/480px-Abraham_Lincoln_O-77_matte_collodion_print.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Albert_Einstein_%28Nobel%29.png/480px-Albert_Einstein_%28Nobel%29.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Catriona_Gray_in_2019.png/480px-Catriona_Gray_in_2019.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sundar_Pichai_World_Economic_Forum_2020.png/480px-Sundar_Pichai_World_Economic_Forum_2020.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Satya_Nadella_at_the_World_Economic_Forum_Annual_Meeting_2020.jpg/480px-Satya_Nadella_at_the_World_Economic_Forum_Annual_Meeting_2020.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Brad_Pitt_2019_by_Glenn_Francis.jpg/480px-Brad_Pitt_2019_by_Glenn_Francis.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Cristiano_Ronaldo_2018.jpg/480px-Cristiano_Ronaldo_2018.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg/480px-Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Serena_Williams_at_2013_US_Open.jpg/480px-Serena_Williams_at_2013_US_Open.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Emma_Watson_2013.jpg/480px-Emma_Watson_2013.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Deadpool_2_Japan_Premiere_Red_Carpet_Ryan_Reynolds_%28cropped%29.jpg/480px-Deadpool_2_Japan_Premiere_Red_Carpet_Ryan_Reynolds_%28cropped%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Chris_Hemsworth_by_Gage_Skidmore_2_%28cropped%29.jpg/480px-Chris_Hemsworth_by_Gage_Skidmore_2_%28cropped%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Reese_Witherspoon_Cannes_2014.jpg/480px-Reese_Witherspoon_Cannes_2014.jpg"
]

# AI-Generated synthetic face endpoints (ThisPersonDoesNotExist)
AI_SYNTHETIC_SOURCES = [
    ("https://thispersondoesnotexist.com", 25)
]

def download_image(url, save_path):
    """Download an image using Python built-in standard library (no external dependencies needed)."""
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            data = response.read()
            if len(data) > 2048:  # valid image file > 2KB
                with open(save_path, 'wb') as f:
                    f.write(data)
                return True
    except Exception as e:
        print(f"  [Skipped] {url}: {e}")
    return False

def fetch_internet_dataset():
    print("=" * 60)
    print("  DOWNLOADING DIVERSE REAL & AI IMAGES FROM THE INTERNET")
    print("=" * 60)

    # 1. Download Real Faces
    print(f"\n[1/2] Downloading authentic public portraits to:\n  {REAL_DIR}")
    real_count = 0
    for idx, url in enumerate(REAL_FACE_URLS):
        dest = os.path.join(REAL_DIR, f"web_real_{idx:03d}.jpg")
        if not os.path.exists(dest):
            if download_image(url, dest):
                real_count += 1
                print(f"  + Downloaded real portrait {idx+1}/{len(REAL_FACE_URLS)}")
        else:
            real_count += 1

    # 2. Download Synthetic AI Faces
    print(f"\n[2/2] Fetching generative AI faces to:\n  {FAKE_DIR}")
    fake_count = 0
    for base_url, target_num in AI_SYNTHETIC_SOURCES:
        for i in range(target_num):
            dest = os.path.join(FAKE_DIR, f"ai_gen_{i:03d}.jpg")
            if not os.path.exists(dest):
                if download_image(base_url, dest):
                    fake_count += 1
                    print(f"  + Downloaded synthetic AI face {i+1}/{target_num}")
                    time.sleep(1.0)
            else:
                fake_count += 1

    print("\n" + "=" * 60)
    print(f"[COMPLETED] Total web real images: {real_count} | Total web AI images: {fake_count}")
    print("Now activate your virtual environment and run 'python train.py'!")
    print("=" * 60)

if __name__ == '__main__':
    fetch_internet_dataset()
