import cv2
import numpy as np

# ১. সাদা ক্যানভাস (White Paper) তৈরি করা
height, width = 800, 600
ballot_paper = np.ones((height, width, 3), dtype="uint8") * 255


# ২. ফাংশন: ব্যালট পেপারের ডিজাইন আঁকা
def draw_ballot_design(img):
    # চার কোণায় ৪টি কালো বক্স (Marker) - যাতে ক্যামেরা ডিটেক্ট করতে পারে
    cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 0), -1)  # Top-Left
    cv2.rectangle(img, (500, 20), (580, 100), (0, 0, 0), -1)  # Top-Right
    cv2.rectangle(img, (20, 700), (100, 780), (0, 0, 0), -1)  # Bottom-Left
    cv2.rectangle(img, (500, 700), (580, 780), (0, 0, 0), -1)  # Bottom-Right

    # টেক্সট এবং ভোটের গোল্লা (Bubble) আঁকা
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Candidate 1
    cv2.putText(img, "Candidate A (Boat)", (150, 250), font, 1, (0, 0, 0), 2)
    cv2.circle(img, (500, 240), 25, (0, 0, 0), 2)  # Empty Circle

    # Candidate 2
    cv2.putText(img, "Candidate B (Paddy)", (150, 400), font, 1, (0, 0, 0), 2)
    cv2.circle(img, (500, 390), 25, (0, 0, 0), 2)  # Empty Circle

    # Candidate 3
    cv2.putText(img, "Candidate C (Plough)", (150, 550), font, 1, (0, 0, 0), 2)
    cv2.circle(img, (500, 540), 25, (0, 0, 0), 2)  # Empty Circle

    return img


# ৩. খালি ব্যালট পেপার তৈরি
blank_ballot = draw_ballot_design(ballot_paper.copy())
cv2.imwrite("blank_ballot.jpg", blank_ballot)
print("✅ Success: 'blank_ballot.jpg' created!")

# ৪. ভোট দেওয়া ব্যালট তৈরি (Simulating a Vote)
voted_ballot = blank_ballot.copy()
# ধরুন ভোটার 'Candidate B'-তে ভোট দিল (গোল্লা ভরাট করল)
cv2.circle(voted_ballot, (500, 390), 20, (0, 0, 0), -1)
cv2.imwrite("voted_ballot.jpg", voted_ballot)
print("✅ Success: 'voted_ballot.jpg' created (Vote for Candidate B)!")