import cv2
import numpy as np


# ১. হেল্পার ফাংশন: ৪টি কোণাকে সঠিকভাবে সাজানো (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


# ২. হেল্পার ফাংশন: বাঁকা ছবিকে সোজা করা (Perspective Transform)
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # নতুন ইমেজের প্রস্থ (Width) বের করা
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # নতুন ইমেজের উচ্চতা (Height) বের করা
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # সোজাসুজি (Top-down view) পয়েন্ট সেট করা
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # ম্যাট্রিক্স ক্যালকুলেশন এবং ওয়ার্প (Warp) করা
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# --- মেইন কোড শুরু ---

# ১. বাঁকা ছবিটি লোড করা
image = cv2.imread('camera_photo.jpg')
if image is None:
    print("Error: camera_photo.jpg not found!")
    exit()

# ২. এজ (Edge) ডিটেকশন
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# ৩. কনট্যুর (Contours) বা বাউন্ডারি খোঁজা
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
docCnt = None

# সব কন্ট্যুর চেক করা এবং সবচেয়ে বড় ৪ কোণা ওয়ালা অবজেক্ট (আমাদের কাগজ) বের করা
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break

if docCnt is None:
    print("❌ Could not find the ballot paper boundaries!")
    exit()

print("✅ Ballot Paper Detected! Fixing perspective...")

# ৪. ছবি সোজা করা (Warping)
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

# ৫. সাইজ ফিক্স করা (আমাদের অরিজিনাল টেমপ্লেট ছিল ৮০০x৬০০)
# এটা না করলে কো-অর্ডিনেট মিলবে না
paper_resized = cv2.resize(warped_gray, (600, 800))  # Width=600, Height=800

# ৬. এখন আবার সেইম লজিক দিয়ে ভোট গোনা
candidates = {
    "Candidate A (Boat)": (500, 240),
    "Candidate B (Paddy)": (500, 390),
    "Candidate C (Plough)": (500, 540)
}

print("-" * 30)
winner = None

for name, (x, y) in candidates.items():
    # ROI (Region of Interest) কাটা
    roi = paper_resized[y - 25:y + 25, x - 25:x + 25]

    # থ্রেশহোল্ড করে একদম সাদা-কালো বানানো (ভালো রেজাল্টের জন্য)
    _, thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)

    # সাদা পিক্সেল গোনা (Binary INV করার কারণে এখন কালো দাগগুলো সাদা হয়ে গেছে)
    pixel_count = cv2.countNonZero(thresh)

    print(f"Checking {name}: Found {pixel_count} filled pixels.")

    if pixel_count > 300:  # থ্রেশহোল্ড একটু কমালাম কারণ রিসাইজে কোয়ালিটি কমেছে
        print(f"✅ VOTE DETECTED for {name}!")
        winner = name

print("-" * 30)
if winner:
    print(f"🏆 FINAL RESULT: {winner} wins!")
else:
    print("⚠️ No valid vote found.")