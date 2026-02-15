import cv2
import numpy as np
import time
import os


# --- ১. পারসপেক্টিভ ট্রান্সফর্ম ফাংশন ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
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


# --- ২. ফাইল থেকে আগের রেজাল্ট লোড করা (যাতে আগের ভোট মুছে না যায়) ---
def load_saved_results():
    votes = {"Candidate A": 0, "Candidate B": 0, "Candidate C": 0, "Invalid": 0}
    if os.path.exists("election_results.txt"):
        with open("election_results.txt", "r") as f:
            for line in f:
                if ":" in line:
                    name, count = line.strip().split(":")
                    if name in votes:
                        votes[name] = int(count)
    return votes


results = load_saved_results()
candidates = {"A": (500, 240), "B": (500, 390), "C": (500, 540)}

# ওয়েবক্যাম চালু করা
cap = cv2.VideoCapture(0)
print("🚀 Scanner Started! Focus on the paper and press 'S'.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 5000:
                docCnt = approx
                break

    if docCnt is not None:
        cv2.drawContours(frame, [docCnt], -1, (0, 255, 0), 2)
        cv2.putText(frame, "READY TO SCAN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF

    # 'S' চাপলে ভোট প্রসেস হবে
    if key == ord('s') and docCnt is not None:
        try:
            paper = four_point_transform(frame, docCnt.reshape(4, 2))
            if paper.shape[1] > paper.shape[0]:
                paper = cv2.rotate(paper, cv2.ROTATE_90_CLOCKWISE)

            paper_resized = cv2.resize(paper, (600, 800))
            gray_p = cv2.cvtColor(paper_resized, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray_p, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            detected_votes = []
            for name, (x, y) in candidates.items():
                roi = thresh[y - 25:y + 25, x - 25:x + 25]
                if cv2.countNonZero(roi) > 350:
                    detected_votes.append(f"Candidate {name}")

            # রেজাল্ট আপডেট এবং ফাইলে সেভ
            if len(detected_votes) == 1:
                winner = detected_votes[0]
                results[winner] += 1
                print(f"🔥 SUCCESS: Recorded for {winner}")
            elif len(detected_votes) > 1:
                results["Invalid"] += 1
                print("❌ INVALID: Double marking detected!")
            else:
                print("⚠️ EMPTY: No vote found.")

            # সাথে সাথে ফাইলে লিখে রাখা (এটিই ড্যাশবোর্ড আপডেট করবে)
            with open("election_results.txt", "w") as f:
                for cand_name, count in results.items():
                    # শুধু প্রার্থীদের ডাটা ফাইলে যাবে ড্যাশবোর্ডের জন্য
                    if "Candidate" in cand_name:
                        f.write(f"{cand_name}:{count}\n")

            print(f"📊 Current Tally: {results}")

        except Exception as e:
            print(f"Error: {e}")

    # UI তে ইনস্ট্রাকশন দেখানো
    cv2.putText(frame, "Scan: 'S' | Quit: 'Q'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_off = 400
    for c, v in results.items():
        cv2.putText(frame, f"{c}: {v}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_off += 25

    cv2.imshow("Voting Machine Camera", frame)
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()