import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import torch
from torchvision.transforms import Normalize
from simpleCNN import SimpleCNN, SimpleCNN_classification
import numpy as np
import sys



def bgr_to_hex(bgr):
    return '#{:02x}{:02x}{:02x}'.format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

def on_drag(event, rect, i):
    x, y = event.x, event.y
    x1, y1, x2, y2 = canvas.coords(rect)

    dx = x - (x1 + x2) // 2
    dy = y - (y1 + y2) // 2

    canvas.move(rect, dx, dy)

    x1, y1, x2, y2 = canvas.coords(rect)
    update_cropped_image(rect, int((x1 + x2) // 2), int((y1 + y2) // 2), i)
    rectangles[i] = rect
    on_rectangle_click(event, rect)

def height_plus_button_callback():
    adjust_rectangle_size("height", 2)

def height_minus_button_callback():
    adjust_rectangle_size("height", -2)

def width_plus_button_callback():
    adjust_rectangle_size("width", 2)

def width_minus_button_callback():
    adjust_rectangle_size("width", -2)

def open_image_button_callback():
    global input_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        input_image = cv2.imread(file_path)
        
        # Get the dimensions of the image
        height, width, _ = input_image.shape

        # Calculate the aspect ratio and dimensions for the 3:2 cropped region
        aspect_ratio = 3 / 2
        if width / height >= aspect_ratio:
            crop_height = height
            crop_width = int(height * aspect_ratio)
        else:
            crop_width = width
            crop_height = int(width / aspect_ratio)

        # Calculate the coordinates of the cropped region
        x1 = (width - crop_width) // 2
        y1 = (height - crop_height) // 2
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        # Crop the image and resize it to 450x300
        input_image = input_image[y1:y2, x1:x2]
        input_image = cv2.resize(input_image, (450, 300))

        show_image(input_image)

def close_image_button_callback():
    global input_image
    input_image = None
    canvas.delete("all")
    for label in cropped_image_labels:
        label.config(image='')

def run_cnn_button_callback():
    run_cnn()

def detect_abnormal_apex_button_callback():
    detect_abnormal_apex()

def show_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

def delete_rectangles():
    for item in canvas.find_all():
        if canvas.type(item) == "rectangle":
            canvas.delete(item)

    for i in range(len(rectangles)):
        rectangles[i] = None
        
def run_cnn():
    delete_rectangles()
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img / 255.0)
    img.unsqueeze_(0)

    output = model(img)
    output = output.detach().numpy().reshape(-1, 2)

    for i, (x, y) in enumerate(output):
        x, y = int(x), int(y)
        rect = canvas.create_rectangle(x - 32, y - 32, x + 33, y + 33, outline=bgr_to_hex(colors[i]), width=2)
        canvas.tag_bind(rect, "<Button-1>", lambda event, rect=rect: on_rectangle_click(event, rect))
        canvas.tag_bind(rect, "<B1-Motion>", lambda event, rect=rect, i=i: on_drag(event, rect, i))
        update_cropped_image(rect, x, y, i)
        rectangles[i] = rect

def update_cropped_image(rect, x, y, i):
    cropped_img = input_image[y - 32:y + 33, x - 32:x + 33]
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    color = colors[i]
    cropped_img = cv2.copyMakeBorder(cropped_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color[::-1])

    photo = ImageTk.PhotoImage(image=Image.fromarray(cropped_img))
    cropped_image_labels[i].config(image=photo)
    cropped_image_labels[i].image = photo


def detect_abnormal_apex():
    cropped_images = []
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = canvas.coords(rect)
        x, y = int((x1 + x2) // 2), int((y1 + y2) // 2)
        cropped_img = input_image[y - 32:y + 33, x - 32:x + 33]
        cropped_img = cv2.resize(cropped_img, (90, 90))
        cropped_images.append(cropped_img)

    inputs = []
    for img in cropped_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img / 255.0)
        inputs.append(img.unsqueeze(0))

    inputs = torch.cat(inputs, 0)

    with torch.no_grad():
        outputs = classification_model(inputs)
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(outputs).numpy()

    for i, prob in enumerate(probs):
        cropped_img = cropped_images[i].copy()
        color = colors[i]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cropped_img = cv2.copyMakeBorder(cropped_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color[::-1])

        if prob[1] > 0.5:
            cv2.putText(cropped_img, "P", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
        else:
            cv2.putText(cropped_img, "N", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

        photo = ImageTk.PhotoImage(image=Image.fromarray(cropped_img))
        cropped_image_labels[i].config(image=photo)
        cropped_image_labels[i].image = photo

def adjust_rectangle_size(dim, delta):
    global selected_rectangle
    if selected_rectangle is not None:
        x1, y1, x2, y2 = canvas.coords(selected_rectangle)
        if dim == "height":
            y2 += delta
        elif dim == "width":
            x2 += delta
        canvas.coords(selected_rectangle, x1, y1, x2, y2)
        index = rectangles.index(selected_rectangle)
        update_cropped_image(selected_rectangle, int((x1 + x2) // 2), int((y1 + y2) // 2), index)

def on_rectangle_click(event, rect):
    global selected_rectangle
    if selected_rectangle is not None:
        color = bgr_to_hex(colors[rectangles.index(selected_rectangle)])
        canvas.itemconfig(selected_rectangle, width=2, outline=color)
    selected_rectangle = rect
    color = bgr_to_hex(colors[rectangles.index(rect)])
    canvas.itemconfig(rect, width=4, outline=color)

sys.stderr = sys.__stderr__
# GUI
root = tk.Tk()
root.title("畸形中央尖自动识别")

frame1 = tk.Frame(root)
frame1.pack(side=tk.TOP, pady=5)
open_image_button = tk.Button(frame1, text="Open Image", command=open_image_button_callback)
open_image_button.pack(side=tk.LEFT, padx=5)
close_image_button = tk.Button(frame1, text="Close Image", command=close_image_button_callback)
close_image_button.pack(side=tk.LEFT, padx=5)
run_cnn_button = tk.Button(frame1, text="Gen Boxes", command=run_cnn_button_callback)
run_cnn_button.pack(side=tk.LEFT, padx=5)
detect_abnormal_apex_button = tk.Button(frame1, text="Central Cusp Deformity", command=detect_abnormal_apex_button_callback)
detect_abnormal_apex_button.pack(side=tk.LEFT, padx=5)

# 在顶部4个按钮的右边增加4个按钮
height_plus_button = tk.Button(frame1, text="height+", command=height_plus_button_callback)
height_plus_button.pack(side=tk.LEFT, padx=5)
height_minus_button = tk.Button(frame1, text="height-", command=height_minus_button_callback)
height_minus_button.pack(side=tk.LEFT, padx=5)
width_plus_button = tk.Button(frame1, text="width+", command=width_plus_button_callback)
width_plus_button.pack(side=tk.LEFT, padx=5)
width_minus_button = tk.Button(frame1, text="width-", command=width_minus_button_callback)
width_minus_button.pack(side=tk.LEFT, padx=5)

canvas = tk.Canvas(root, width=450, height=300)
canvas.pack(side=tk.TOP, pady=5)

frame2 = tk.Frame(root)
frame2.pack(side=tk.TOP, pady=5)
cropped_image_labels = [tk.Label(frame2) for _ in range(4)]
for label in cropped_image_labels:
    label.pack(side=tk.LEFT, padx=5)

input_image = None
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
rectangles = [None, None, None, None]

# Load the models
device = torch.device("cpu")
model = SimpleCNN()
model.to(device)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

classification_model = SimpleCNN_classification()
classification_model.to(device)
classification_model.load_state_dict(torch.load("classification_model.pth", map_location="cpu"))
classification_model.eval()

selected_rectangle = None

root.mainloop()