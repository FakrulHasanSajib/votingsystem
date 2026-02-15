import cv2
import numpy as np
import os


# --- ফাংশন: ছবি রোটেট করা ---
def rotate_image(image, angle):
    # ছবির সেন্টার বের করা
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # রোটেট ম্যাট্রিক্স
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # ব্যাকগ্রাউন্ড কালার ডার্ক গ্রে (50, 50, 50) সেট করা
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(50, 50, 50))
    return rotated


# --- মেইন কোড শুরু ---
print("--- Simulation Started ---")

filename = 'voted_ballot.jpg'

# ১. আগে চেক করি ফাইলটি ফোল্ডারে আছে কি না
if not os.path.exists(filename):
    print(f"❌ Error: '{filename}' ফাইলটি পাওয়া যাচ্ছে না!")
    print("দয়া করে আগে 'python generate_ballot.py' রান করুন।")
else:
    # ২. ফাইল আছে, এখন রিড করি
    img = cv2.imread(filename)

    # ৩. রিড করার পর img ভেরিয়েবল ঠিক আছে কি না চেক
    if img is None:
        print("❌ Error: ফাইল আছে কিন্তু ওপেন করা যাচ্ছে না (Corrupted file)।")
    else:
        print("✅ Image loaded successfully.")

        # ৪. ছবি বাঁকা করা (-১৫ ডিগ্রি)
        rotated_img = rotate_image(img, -15)

        # ৫. ছবির সাইজ ছোট করা
        height, width = rotated_img.shape[:2]
        final_image = cv2.resize(rotated_img, (int(width * 0.8), int(height * 0.8)), interpolation=cv2.INTER_AREA)

        # ৬. সেভ করা
        cv2.imwrite('camera_photo.jpg', final_image)
        print("✅ Success! 'camera_photo.jpg' created with DARK background.")

print("--- Simulation Finished ---")