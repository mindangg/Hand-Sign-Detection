import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

letters_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
digits_labels = list("123456789")
cap = None
running = False
paused = False  # Thêm biến paused
mode = "letters"
frame_processed = None

def update_webcam():
    global frame_processed
    if running:
        success, frame = cap.read()
        if success:
            # frame = cv2.flip(frame, -1)
            if frame_processed is not None:
                frame = frame_processed
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((525, 410))
            imgtk = ImageTk.PhotoImage(image=img)
            webcam_label.imgtk = imgtk
            webcam_label.configure(image=imgtk)
        webcam_label.after(10, update_webcam)

def detect_sign():
    global running, cap
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    letter_classifier = Classifier("project/Model/letters_model.h5", "project/Model/letters_labels.txt")
    digit_classifier = Classifier("project/Model/digits_model.h5", "project/Model/digits_labels.txt")
    offset = 20
    imgSize = 300
    running = True
    prev_char = None
    update_webcam()

    while running:
        success, img = cap.read()
        if not success:
            continue

        # img = cv2.flip(img, 1)
        hands, _ = detector.findHands(img)
        if hands:
            if paused:
                continue  # Bỏ qua xử lý khi đang tạm dừng
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            # Store processed frame with bounding box
            global frame_processed
            frame_processed = img.copy()
            cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (0, 255, 0), 2)

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            if mode == "letters":
                prediction, index = letter_classifier.getPrediction(imgWhite, draw=False)
                char = letters_labels[index]
            else:
                prediction, index = digit_classifier.getPrediction(imgWhite, draw=False)
                char = digits_labels[index]


            # Trong vòng while:
            # if char != prev_char:
            chatbox.configure(state='normal')
            chatbox.insert(tk.END, f"🤖 AI: Tôi đoán là ký hiệu '{char}'\n")
            chatbox.configure(state='disabled')
            chatbox.see(tk.END)
                # prev_char = char


def start_thread():
    thread = threading.Thread(target=detect_sign)
    thread.daemon = True
    thread.start()

def toggle_pause():
    global paused
    paused = not paused
    chatbox.configure(state='normal')
    if paused:
        chatbox.insert(tk.END, "⏸️ AI: Đã tạm dừng nhận diện ký hiệu.\n")
    else:
        chatbox.insert(tk.END, "▶️ AI: Tiếp tục nhận diện ký hiệu.\n")
    chatbox.configure(state='disabled')
    chatbox.see(tk.END)

def toggle_mode():
    global mode
    mode = "digits" if mode == "letters" else "letters"
    chatbox.configure(state='normal')
    chatbox.insert(tk.END, f"🔁 AI: Đã chuyển sang chế độ nhận diện {'số (1-9)' if mode == 'digits' else 'chữ (A-Z)'}.\n")
    chatbox.configure(state='disabled')
    chatbox.see(tk.END)

def stop_app():
    global running
    running = False
    if cap and cap.isOpened():
        cap.release()
    root.quit()
    root.destroy()

# ================= GUI ==================
root = tk.Tk()
root.title("AI Ký hiệu tay")
root.geometry("1500x800")
root.resizable(False, False)

# Background image
bg_image = Image.open("project/ai.jpg").resize((1500, 800))
bg = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
bg_label.lower()

# Font & color
FONT = ("Segoe UI", 12)
TEXT_COLOR = "#696969"
BTN_COLOR = "#00B7D2"
BTN_HOVER = "#00A4BF"
BG_FRAME_COLOR = "#d0f0f7"

# Left webcam frame
webcam_frame = tk.Frame(root,
                               highlightthickness=0)
webcam_frame.place(relx=0.263 ,rely=0.45 ,anchor="center")

webcam_label = tk.Label(webcam_frame, bg=BG_FRAME_COLOR)
webcam_label.pack(fill=tk.BOTH, expand=True)

# Right chat frame (Căn giữa)
chat_frame = tk.Frame(root, bg="#E4F8FF", width=725, height=635, highlightthickness=0)
chat_frame.place(relx=0.77, rely=0.54, anchor="center")  # Căn giữa theo chiều ngang và dọc


chatbox_frame = tk.Frame(chat_frame, bg="#1CCCFC", width=480, height=510)
chatbox_frame.pack(padx=1, pady=1)

# Chatbox
chatbox = scrolledtext.ScrolledText(chatbox_frame, font=FONT, bg="white", fg="#172948",
                                     wrap=tk.WORD, width=53, height=24.5, bd=0, highlightthickness=0)
chatbox.place(x=0, y=0)
chatbox.insert(tk.END, "🤖 AI: Chào bạn! Bấm 'Bắt đầu' để nhận diện ký hiệu ✋\n")
chatbox.configure(state='disabled')

# Buttons
btn_frame = tk.Frame(chat_frame, bg="#E4F8FF")
btn_frame.pack(pady=15)

def on_enter(e): e.widget['bg'] = BTN_HOVER
def on_leave(e): e.widget['bg'] = BTN_COLOR

start_btn = tk.Button(btn_frame, text="▶️ Bắt đầu", font=FONT, bg=BTN_COLOR, fg="white",
                      activebackground="#2D508C", bd=0, padx=15, pady=5, command=start_thread)
start_btn.grid(row=0, column=0, padx=10)
start_btn.bind("<Enter>", on_enter)
start_btn.bind("<Leave>", on_leave)

pause_btn = tk.Button(btn_frame, text="⏸️ Tạm dừng", font=FONT, bg="#FFD54F", fg="white",
                      activebackground="#FFCA28", bd=0, padx=15, pady=5, command=toggle_pause)
pause_btn.grid(row=0, column=1, padx=10)
pause_btn.bind("<Enter>", lambda e: pause_btn.config(bg="#FFCA28"))
pause_btn.bind("<Leave>", lambda e: pause_btn.config(bg="#FFD54F"))

switch_btn = tk.Button(btn_frame, text="🔀 Chuyển đổi (A-Z / 1-9)", font=FONT, bg="#7E57C2", fg="white",
                       activebackground="#673AB7", bd=0, padx=15, pady=5, command=toggle_mode)
switch_btn.grid(row=0, column=3, padx=10)
switch_btn.bind("<Enter>", lambda e: switch_btn.config(bg="#673AB7"))
switch_btn.bind("<Leave>", lambda e: switch_btn.config(bg="#7E57C2"))

exit_btn = tk.Button(btn_frame, text="❌ Thoát", font=FONT, bg="#ff8a80", fg="white",
                     activebackground="#ff5252", bd=0, padx=15, pady=5, command=stop_app)
exit_btn.grid(row=0, column=2, padx=10)

root.protocol("WM_DELETE_WINDOW", stop_app)
root.mainloop()
