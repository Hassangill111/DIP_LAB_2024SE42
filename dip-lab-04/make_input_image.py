"""Generate the synthetic input image (no internet needed): gradient, shapes, text, checkerboard, asymmetric marker."""
import numpy as np, cv2

H, W = 400, 600
yy, xx = np.mgrid[0:H, 0:W]
img = np.zeros((H, W, 3), np.uint8)  # BGR
img[..., 0] = (xx / W * 255).astype(np.uint8)          # blue grows to the right
img[..., 1] = (yy / H * 255).astype(np.uint8)          # green grows downward
img[..., 2] = 255 - (xx / W * 255).astype(np.uint8)    # red falls to the right
cv2.circle(img, (150, 130), 70, (0, 215, 255), -1)
cv2.circle(img, (150, 130), 70, (0, 0, 0), 3)
cv2.rectangle(img, (330, 60), (540, 190), (255, 255, 255), -1)
cv2.rectangle(img, (330, 60), (540, 190), (0, 0, 0), 3)
cv2.putText(img, "DIP LAB 04", (345, 138), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
tri = np.array([[90, 360], [210, 360], [150, 250]], np.int32)   # triangle = asymmetric marker
cv2.fillPoly(img, [tri], (50, 200, 50)); cv2.polylines(img, [tri], True, (0, 0, 0), 3)
for r in range(6):                                              # checkerboard patch (interpolation test)
    for c in range(8):
        col = (0, 0, 0) if (r + c) % 2 == 0 else (255, 255, 255)
        cv2.rectangle(img, (330 + c * 26, 220 + r * 26), (330 + c * 26 + 25, 220 + r * 26 + 25), col, -1)
cv2.arrowedLine(img, (260, 380), (300, 300), (0, 0, 255), 4, tipLength=0.3)
cv2.putText(img, "TOP-LEFT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite("input/sample_image.png", img)
print("saved", img.shape)
