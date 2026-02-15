import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
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


print("--- Running Smart Scanner (Threshold Method) ---")

# ১. ছবি লোড
image = cv2.imread('camera_photo.jpg')

if image is None:
    print("❌ Error: camera_photo.jpg not found!")
else:
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ২. [FIX] সাধারণ থ্রেশহোল্ড ব্যবহার করা হচ্ছে (সাদা কাগজ vs কালো ব্যাকগ্রাউন্ড)
    # ১০০ এর বেশি ভ্যালু হলে সাদা (255), নাহলে কালো (0)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # ৩. কন্ট্যুর খোঁজা
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    docCnt = None

    if len(cnts) > 0:
        # সবচেয়ে বড় এরিয়া অনুযায়ী সর্ট করা
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # যদি ৪ কোণা বিশিষ্ট এবং যথেষ্ট বড় হয়
            if len(approx) == 4 and cv2.contourArea(c) > 1000:
                docCnt = approx
                break

    # ৪. কন্ট্যুর চেক করা
    if docCnt is None:
        print("❌ Error: কাগজ খুঁজে পাওয়া যায়নি! (Threshold Failed)")
    else:
        print("✅ Paper Detected.")

        # ডিবাগ: লাল বর্ডার একে দেখা যে কম্পিউটার কী ডিটেক্ট করেছে
        cv2.drawContours(orig, [docCnt], -1, (0, 0, 255), 5)
        cv2.imwrite('step1_contour.jpg', orig)
        print("📸 'step1_contour.jpg' সেভ করা হয়েছে। চেক করুন লাল বর্ডার ঠিক আছে কিনা।")

        # ৫. ছবি সোজা করা
        paper = four_point_transform(image, docCnt.reshape(4, 2))

        # ওরিয়েন্টেশন ফিক্স (যদি লম্বায় ছোট হয়, ঘুরিয়ে দাও)
        h, w = paper.shape[:2]
        if w > h:
            paper = cv2.rotate(paper, cv2.ROTATE_90_CLOCKWISE)

        # সাইজ ফিক্স
        paper_resized = cv2.resize(paper, (600, 800))

        # ৬. ভোট গণনা
        gray_paper = cv2.cvtColor(paper_resized, cv2.COLOR_BGR2GRAY)

        # কালো কালির জন্য ইনভার্স থ্রেশহোল্ড
        _, thresh_vote = cv2.threshold(gray_paper, 210, 255, cv2.THRESH_BINARY_INV)

        candidates = {
            "A": (500, 240),
            "B": (500, 390),
            "C": (500, 540)
        }

        print("-" * 30)
        for name, (x, y) in candidates.items():
            roi = thresh_vote[y - 25:y + 25, x - 25:x + 25]
            count = cv2.countNonZero(roi)

            print(f"Candidate {name}: Pixel Count = {count}")

            if count > 300:  # 3০০ পিক্সেলের বেশি কালো হলে ভোট
                color = (0, 255, 0)  # Green (Winner)
                cv2.rectangle(paper_resized, (x - 25, y - 25), (x + 25, y + 25), color, 2)
                cv2.putText(paper_resized, "VOTE", (x - 90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                color = (0, 0, 255)  # Red (Loser)
                cv2.rectangle(paper_resized, (x - 25, y - 25), (x + 25, y + 25), color, 1)

        print("-" * 30)
        cv2.imwrite('final_result_fixed.jpg', paper_resized)
        print("📸 Check 'final_result_fixed.jpg' now!")