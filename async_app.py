import asyncio
import os
import datetime


import aiohttp
import cv2
from flask import jsonify
import csv

from ultralytics import YOLO

# Model initialization
model = YOLO("./model.pt")


async def detect_license_plates(image_path):
    try:

        img = cv2.imread(image_path)
        results = model(img, stream=True)
        detected_plates = []
        async with aiohttp.ClientSession() as session:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers

                    # Extract the detected license plate region
                    cropped_plate = img[y1:y2, x1:x2]
                    cv2.imwrite('test.jpg', cropped_plate)
                    filename = os.path.basename('/home/khushali/Desktop/tracking/test.jpg')
                    data = aiohttp.FormData()
                    data.add_field(
                        name="upload",
                        value=open('/home/khushali/Desktop/tracking/test.jpg', "rb"),
                        filename=filename,
                        content_type='image/jpeg',
                    )
                    # files = {'upload': data}

                    async with session.post(
                            'https://api.platerecognizer.com/v1/plate-reader/',
                            headers={
                                'Authorization': 'Write_token'},
                            data=data
                    ) as response:
                        # await asyncio.sleep(0.3)
                        results = await response.json()
                    for result in results.get('results', []):
                        number_plate = result.get('plate', '')
                        detected_plates.append(number_plate)

                        print("Detected number plate:", number_plate)

        # Process detected license plates
        # Save the timestamp and all detected number plates in the CSV file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('number_plate.csv', 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for plate in detected_plates:
                csv_writer.writerow([timestamp, plate, image_path])

    except Exception as e:
        return jsonify({'error': str(e)}), 500
