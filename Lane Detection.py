import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def ROI(image):
    height = image.shape[0]
    # Wider region to capture both lanes
    polygon = np.array([[
        (100, height),
        (1200, height),
        (700, 250),
        (580, 250)
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(image, mask)
    return cropped

def coordinate(image, line_parameters):
    if line_parameters is None or len(line_parameters) != 2:
        return None
    slope, intercept = line_parameters
    if slope == 0 or np.isnan(slope) or np.isnan(intercept):
        return None
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope(image, lines):
    if lines is None:
        return None, None

    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        # Adjusted slope thresholds to detect both sides better
        if slope < -0.3:
            left_fit.append((slope, intercept))
        elif slope > 0.3:
            right_fit.append((slope, intercept))

    left_line, right_line = None, None
    if len(left_fit) > 0:
        left_fit_avg = np.average(left_fit, axis=0)
        left_line = coordinate(image, left_fit_avg)
    if len(right_fit) > 0:
        right_fit_avg = np.average(right_fit, axis=0)
        right_line = coordinate(image, right_fit_avg)

    return left_line, right_line

def display_lines_and_fill(image, left_line, right_line):
    overlay = np.zeros_like(image)

    if left_line is not None:
        cv2.line(overlay, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 10)  # red
    if right_line is not None:
        cv2.line(overlay, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 10)  # blue

    if left_line is not None and right_line is not None:
        # Fill area between lanes
        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))

    return cv2.addWeighted(image, 0.8, overlay, 0.5, 0)

# ---- Main ----
cap = cv2.VideoCapture("test2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    edges = canny(frame)
    cropped = ROI(edges)
    lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 80, np.array([]), 40, 5)
    left_line, right_line = average_slope(frame, lines)
    final = display_lines_and_fill(frame, left_line, right_line)

    cv2.imshow("Result", final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
