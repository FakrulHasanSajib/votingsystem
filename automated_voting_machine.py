import cv2
import numpy as np
import os
import time


# --- ১. হেল্পার ফাংশনসমূহ (আগের মতোই) ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)];
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)];
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


# --- ২. রেজাল্ট স্টোরেজ সেটিংস ---
# আমরা একটি টেক্সট ফাইলে রেজাল্ট সেভ করব
RESULT_FILE = "election_results.txt"
if not os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "w") as f:
        f.write("Candidate A:0\nCandidate B:0\nCandidate C:0")


def update_vote(winner_name):
    # ফাইল থেকে বর্তমান রেজাল্ট পড়া
    votes = {}
    with open(RESULT_FILE, "r") as f:
        for line in f:
            name, count = line.strip().split(":")
            votes[name] = int(count)

    # বিজয়ীর ভোট বাড়ানো
    if winner_name in votes:
        votes[winner_name] += 1

    # আপডেট করা রেজাল্ট ফাইলে সেভ করা
    with open(RESULT_FILE, "w") as f:
        for name, count in votes.items():
            f.write(f"{name}:{count}\n")
    return votes


# --- ৩. মেইন স্ক্যানিং লুপ ---
print("🚀 Automated Voting Machine is Starting...")
print("Waiting for ballot papers (camera_photo.jpg)... Press Ctrl+C to stop.")

last_processed_time = 0

try:
    while True:
        # এখানে আমরা চেক করব camera_photo.jpg ফাইলটি আপডেট হয়েছে কি না
        if os.path.exists('camera_photo.jpg'):
            file_time = os.path.getmtime('camera_photo.jpg')

            if file_time > last_processed_time:
                print("\n📄 New Ballot Paper Detected! Scanning...")
                image = cv2.imread('camera_photo.jpg')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                docCnt = None
                if len(cnts) > 0:
                    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                    for c in cnts:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        if len(approx) == 4 and cv2.contourArea(c) > 1000:
                            docCnt = approx
                            break

                if docCnt is not None:
                    paper = four_point_transform(image, docCnt.reshape(4, 2))
                    if paper.shape[1] > paper.shape[0]: paper = cv2.rotate(paper, cv2.ROTATE_90_CLOCKWISE)
                    paper_resized = cv2.resize(paper, (600, 800))
                    gray_paper = cv2.cvtColor(paper_resized, cv2.COLOR_BGR2GRAY)
                    _, thresh_vote = cv2.threshold(gray_paper, 210, 255, cv2.THRESH_BINARY_INV)

                    candidates = {"A": (500, 240), "B": (500, 390), "C": (500, 540)}
                    winner = None
                    max_pixels = 0

                    for name, (x, y) in candidates.items():
                        roi = thresh_vote[y - 25:y + 25, x - 25:x + 25]
                        count = cv2.countNonZero(roi)
                        if count > 300 and count > max_pixels:
                            max_pixels = count
                            winner = f"Candidate {name}"

                    if winner:
                        print(f"✅ Vote Counted for: {winner}")
                        current_tally = update_vote(winner)
                        print(f"📊 Current Standings: {current_tally}")
                    else:
                        print("⚠️ No valid vote found on this paper.")

                last_processed_time = file_time

        time.sleep(2)  # ২ সেকেন্ড পর পর চেক করবে নতুন কাগজ এলো কিনা

except KeyboardInterrupt:
    print("\n🛑 Machine Shutting Down. Final results are in 'election_results.txt'.")