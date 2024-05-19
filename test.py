import unittest
import os
import io
import cv2
import numpy as np
from tkinter import Tk
from client import GUI
from master import app, process_image_operation

class TestSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.client.testing = True
        cls.root = Tk()
        cls.gui = GUI(cls.root)
        cls.test_image_path = "test.jpg"
        # Create a test image file once for all tests
        with open(cls.test_image_path, 'wb') as f:
            f.write(os.urandom(100 * 100 * 3))

    def test_process_image_operation_edge_detection(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        processed_image = process_image_operation(image, 'edge_detection')
        self.assertEqual(processed_image.shape, image.shape[:2])

    def test_process_image_operation_color_inversion(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        processed_image = process_image_operation(image, 'color_inversion')
        self.assertTrue((processed_image == 255).all())

    def test_process_image_operation_gaussian_blur(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        processed_image = process_image_operation(image, 'gaussian_blur')
        self.assertEqual(processed_image.shape, image.shape)

    def test_process_image_operation_otsu_threshold(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        processed_image = process_image_operation(image, 'otsu_threshold')
        self.assertEqual(processed_image.shape, image.shape[:2])

    def test_process_image_endpoint(self):
        image = cv2.imencode('.jpg', np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()
        response = self.client.post('/process_image', data={
            'image': (io.BytesIO(image), 'image.jpg'),
            'operation': 'edge_detection'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'image/jpeg')

    def test_select_image(self):
        self.gui.image_path = self.test_image_path
        self.gui.select_image()
        self.assertTrue(hasattr(self.gui, 'image_path'))

    def test_process(self):
        self.gui.image_path = self.test_image_path
        self.gui.operation_var.set("edge_detection")
        self.gui.process()
        processed_image = self.gui.processed_image_label.cget("image")
        self.assertIsNotNone(processed_image)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)

if __name__ == '__main__':
    unittest.main()
