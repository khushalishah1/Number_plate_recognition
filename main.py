import asyncio
import os
import time

import cv2
import pandas as pd
from ultralytics import YOLO
from async_app import detect_license_plates
from async_tracker import AsyncTracker


async def process_results(results, class_list):
    list = []
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row[:6])
        if d < 8 and class_list[d] == 'car':
            list.append([x1, y1, x2, y2])
    return list


async def process_tracking_results(frame, bbox_id, cy2, offset, vh_dict, counter):
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx, cy = (int(x3 + x4) // 2, int(y3 + y4) // 2)
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            if id not in vh_dict:
                vh_dict[id] = 1
            else:
                vh_dict[id] += 1
            if vh_dict[id] == 1:
                counter += 1
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.line(frame, (cx - 1000, cy), (cx + 1000, cy), (255, 255, 255), 1)
                cv2.imwrite(f'images/{counter}.png', frame)
                start = time.time()
                await detect_license_plates(f'images/{counter}.png')
                end = time.time()
                print("time taken to detect license plate:", end - start, "seconds")


async def main():
    start_model = time.time()
    model = YOLO('yolov8s.pt')
    cap = cv2.VideoCapture('track.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")

    os.makedirs('images', exist_ok=True)

    tracker = AsyncTracker()
    end_model = time.time()
    print("Total time: ", end_model - start_model)
    vh_dict = {}
    cy2, offset = 190, 25
    counter = 0
    count = 0
    target_fps = 1  # Set your desired target FPS
    wait_key_delay = int(fps / target_fps)
    starttime = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % int(fps / target_fps) != 0:  # Assuming original video is 30 FPS
            continue

        frame = cv2.resize(frame, (1020, 500))

        start = time.time()
        results = model.predict(frame)
        end = time.time()
        print("time taken to process one frame: ", end - start, "seconds")
        list = await process_results(results, class_list)

        bbox_id = await tracker.async_update(list)
        await process_tracking_results(frame, bbox_id, cy2, offset, vh_dict, counter)

        cv2.line(frame, (300, cy2), (1000,cy2), (255, 255, 255), 1)
        cv2.putText(frame, "line cross", (400, 170), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255),
                    2)
        cv2.imshow("RGB", frame)

        if cv2.waitKey(wait_key_delay) & 0xFF == 27:
            break

    endtime = time.time()

    print(f"Time taken  {endtime - starttime}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
