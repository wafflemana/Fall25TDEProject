import cv2
import os
import glob

#saves video to specific path
image_folder = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\lumoplots" #put the absolute path of the folder in which the images are stored
save_path = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\movies" #put the absolute path for the folder in which you'd like the movie to be saved
output_file = "lumoplotsv1_2_8fps.mp4" #name of the movie file
fps = 8 #framerate of the movie, each image is 1 frame
os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, output_file)

#image loading
images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
frame = cv2.imread(images[0])
height, width, layers = frame.shape

#video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#writes images to video
for img_file in images:
    img = cv2.imread(img_file)
    if img is None:
        continue
    video.write(img)

video.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_path}")
