from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__, static_url_path='/static')

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def process_image(file_path):
    # Your image processing code here
    # This can be your existing image processing code
    # Read the uploaded image

    img = cv2.imread(file_path)
    img_original = img.copy()

    # Image modification
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 10, 20)

    # Contour detection
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    biggest = biggest_contour(contours)

    cv2.polylines(img, [biggest], isClosed=True, color=(0, 255, 0), thickness=3)

    # Pixel values in the original image
    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    (top_left, top_right, bottom_right, bottom_left) = input_points
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    # Output image size
    max_width = max(int(bottom_width), int(top_width))
    max_height = int(max_width * 1.414)  # for A4

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

    # Ensure the data type is uint8 before saving
    img_output = np.clip(img_output, 0, 255).astype(np.uint8)

    # Save the processed image temporarily
    # result_path = 'static/result.jpeg'
    # cv2.imwrite(result_path, img_output)

    return img_output


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image temporarily
            file_path = 'temp_image.jpeg'
            file.save(file_path)

            # Process the image
            processed_img = process_image(file_path)

            # Ensure the data type is uint8 before saving
            # processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)

            # Display the processed image on the web page
            cv2.imwrite('static/result.jpeg', processed_img)
            # return render_template('result.html', result='static/result.jpg')
            return render_template('result.html')

    # return render_template('result.html', result=None)
    return render_template('index.html')

