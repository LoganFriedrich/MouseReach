from matplotlib import animation
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture('20220721_H36_E2.mp4')
video_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Create the main plot
fig, ax = plt.subplots()

# Create the animation
ims = []
for i in range(200):
    im = ax.imshow(video_frames[i], animated=True)
    if i == 0:
        ax.imshow(video_frames[i])  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=5000)


plt.show()
