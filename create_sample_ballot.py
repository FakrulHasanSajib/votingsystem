import cv2
import numpy as np

# ১. একটি সাদা ব্যাকগ্রাউন্ড তৈরি (৮০০x৬০০ সাইজ)
ballot = np.ones((800, 600, 3), dtype=np.uint8) * 255

# ২. টাইটেল এবং বর্ডার
cv2.putText(ballot, "OFFICIAL BALLOT PAPER", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.rectangle(ballot, (20, 20), (580, 780), (0, 0, 0), 3)

# ৩. প্রার্থীদের ঘর তৈরি করা
candidates = [
    {"name": "Candidate A (Boat)", "pos": (500, 240)},
    {"name": "Candidate B (Paddy)", "pos": (500, 390)},
    {"name": "Candidate C (Plough)", "pos": (500, 540)}
]

for person in candidates:
    # প্রার্থীর নাম
    cv2.putText(ballot, person["name"], (50, person["pos"][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # সিল দেওয়ার ঘর (বৃত্ত)
    cv2.circle(ballot, person["pos"], 30, (0, 0, 0), 2)

# ৪. একটি নমুনা ভোট দিন (Candidate B তে একটি ভরাট বৃত্ত আঁকি টেস্ট করার জন্য)
# আপনি চাইলে এই লাইনটি কমেন্ট করে একদম ফ্রেশ ব্যালট নিতে পারেন
cv2.circle(ballot, (500, 390), 20, (0, 0, 0), -1)

# ৫. সেভ করা
cv2.imwrite('test_ballot.jpg', ballot)
print("✅ Success! 'test_ballot.jpg' তৈরি হয়েছে। এটি ফোনের স্ক্রিনে ওপেন করে ক্যামেরার সামনে ধরুন।")