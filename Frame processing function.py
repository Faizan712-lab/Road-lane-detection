def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blur, 50, 150)

    # Define ROI
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height), (width, height),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.4), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Prepare image for the DRL model
    input_image = cv2.resize(masked_edges, (128, 128))  # Model input size
    input_image = input_image / 255.0  # Normalize
    input_image = np.expand_dims(input_image, axis=[0, -1])  # Add batch and channel dimensions

    # Predict lane markings using the DRL model
    prediction = model.predict(input_image)
    prediction = np.squeeze(prediction)  # Remove unnecessary dimensions
    prediction = cv2.resize(prediction, (width, height))
    prediction = (prediction > 0.5).astype(np.uint8) * 255  # Thresholding

    # Overlay the prediction on the original frame
    color_mask = np.zeros_like(frame)
    color_mask[:, :, 1] = prediction  # Green color for lane
    combined_frame = cv2.addWeighted(frame, 0.8, color_mask, 1, 0)

    return combined_frame
