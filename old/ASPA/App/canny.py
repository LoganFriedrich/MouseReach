import cv2

cap    = cv2.VideoCapture('20220721_H36_E2.mp4')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    # Display the resulting frame
    cv2.imshow('Edges', edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        if len(approx)==4: 
            x,y,w,h = cv2.boundingRect(cnt)
            if abs(w - 30) < 10 and abs(h - 50) < 20: 
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    cv2.imshow('Rectangles', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()