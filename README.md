# Number_plate_recognition

Here, we pass the video file first. Next, we define a line. If the car's centre (x,y) touches this line, we snap a screenshot. From this screenshot, a cropped image is produced first, and then the number plate is detected using an API.

First Install all the requirements from requirements.txt. Also download coco.txt and model.pt file from the repo.

Then for running the whole process run below command. 
!python main.py
