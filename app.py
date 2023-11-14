from PIL import Image, ImageDraw
from io import BytesIO
from ultralytics import YOLO
import json
from flask import request, Response, Flask
from waitress import serve
import numpy as np

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

@app.route("/detect", methods=["POST"])
def detect():
    images_with_boxes = []  

    for i in range(4):  
        image_field_name = f"image_file{i + 1}"
        if image_field_name not in request.files:
            return {"error": f"No file part for {image_field_name}"}

        file = request.files[image_field_name]

        if file.filename == '':
            return {"error": f"No selected file for {image_field_name}"}

        if not allowed_file(file.filename):
            return {"error": f"Invalid image format for {image_field_name}"}

        
        image = Image.open(file)

        
        image_with_boxes, _ = detect_objects_on_image(image)
        images_with_boxes.append(image_with_boxes)

   
    final_image = concatenate_images(images_with_boxes)

    
    final_image = np.array(final_image)

    
    img_byte_array = BytesIO()
    Image.fromarray(final_image).save(img_byte_array, format="PNG")

    
    response = Response(img_byte_array.getvalue())
    response.headers["Content-Type"] = "image/png"
    response.headers["Content-Disposition"] = 'inline; filename="detected_images.png"'
    
    return response
#print('response',response)
def detect_objects_on_image(image):
    model = YOLO("best.pt")
    results = model.predict(image)
    result = results[0]
    
    image_with_boxes = np.array(image)
    image_with_boxes = Image.fromarray(image_with_boxes)

    draw = ImageDraw.Draw(image_with_boxes)
    
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{result.names[class_id]}: {prob}", fill="red")

    return image_with_boxes, None

def concatenate_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return new_image

if __name__ == '__main__':
    app.run()


