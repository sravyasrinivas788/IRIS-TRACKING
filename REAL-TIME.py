import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('iris_track2.h5')

try:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = frame[50:500, 50:500, :]  # Crop the frame
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        resized = cv2.resize(rgb_img, (250, 250))  # Resize for the model
        
        # Make predictions
        yhat = model.predict(np.expand_dims(resized / 255.0, axis=0))
        sample_coords = yhat[0, :4]
        
        # Draw the predicted points
        cv2.circle(frame, tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)), 2, (255, 0, 0), -1)
        cv2.circle(frame, tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)), 2, (0, 255, 0), -1)
        
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame using matplotlib
        plt.imshow(frame_rgb)
        plt.title('EyeTrack')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.001)
        plt.clf()
        
        # # Break the loop on 'q' key press
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
finally:
    cap.release()
    plt.close()
