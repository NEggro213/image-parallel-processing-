from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
from mpi4py import MPI
from skimage.segmentation import slic, mark_boundaries

app = Flask(__name__)
CORS(app)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@app.route('/process_image', methods=['POST'])
def process_image():
    if rank == 0:
        try:
            # Master process receives the image and operation
            image_data = request.files['image'].read()
            operation = request.form['operation']

            # Decode the image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({'error': 'Failed to load image'}), 400

            # Split the image into parts for each worker
            image_parts = np.array_split(image, size, axis=0)

            # Send parts to workers
            for i in range(1, size):
                comm.send((image_parts[i], operation), dest=i)

            # Process master's part
            processed_image_part = process_image_operation(image_parts[0], operation)

            # Gather processed parts from workers
            processed_image_parts = [processed_image_part]
            for i in range(1, size):
                processed_image_parts.append(comm.recv(source=i))

            # Combine processed parts
            processed_image = np.vstack(processed_image_parts)

            # Encode the final image to JPEG format
            _, processed_image_data = cv2.imencode('.jpg', processed_image)
            return send_file(
                io.BytesIO(processed_image_data),
                mimetype='image/jpeg'
            )

        except Exception as e:
            return jsonify({'error': str(e)}), 400

def process_image_operation(image_part, operation):
    if operation == 'edge_detection':
        processed_image_part = cv2.Canny(image_part, 100, 200)
    elif operation == 'color_inversion':
        processed_image_part = cv2.bitwise_not(image_part)
    elif operation == 'gaussian_blur':
        processed_image_part = cv2.GaussianBlur(image_part, (15, 15), 0)
    elif operation == 'otsu_threshold':
        gray = cv2.cvtColor(image_part, cv2.COLOR_BGR2GRAY)
        _, processed_image_part = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif operation == 'superpixel_segmentation':
        processed_image_part = apply_superpixel_segmentation(image_part)
    else:
        processed_image_part = image_part

    return processed_image_part

def apply_superpixel_segmentation(image):
    segments = slic(image, n_segments=100, compactness=10)
    segmented_image = mark_boundaries(image, segments)
    segmented_image = (segmented_image * 255).astype(np.uint8)  # Convert from [0, 1] to [0, 255]
    return segmented_image

def worker_process():
    while True:
        # Worker process waits for data from the master
        image_part, operation = comm.recv(source=0)

        # Process the image part
        processed_image_part = process_image_operation(image_part, operation)

        # Send the processed image part back to the master
        comm.send(processed_image_part, dest=0)

if __name__ == '__main__':
    if rank == 0:
        app.run(host='0.0.0.0', port=5000)
    else:
        worker_process()
