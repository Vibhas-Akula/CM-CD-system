import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def load_labels():
    with open("coco.names", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def detect_people(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Class ID 0 is for 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_labels(frame, boxes, confidences, class_ids, labels):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    crowd_count = 0
    if len(indexes) > 0: 
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(labels[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)  # Green for 'person'
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            crowd_count += 1
    return frame, crowd_count

def detect_crowd(video_file):
    net, output_layers = load_yolo()
    labels = load_labels()
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        boxes, confidences, class_ids = detect_people(frame, net, output_layers)
        frame, crowd_count = draw_labels(frame, boxes, confidences, class_ids, labels)

        # Display crowd count and frame size on the frame
        cv2.putText(frame, f"People Count: {crowd_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(frame, f"Frame Size: {frame_size[0]} x {frame_size[1]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Frames: {total_frames}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("Crowd Detection", frame)

        # Break the loop on 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

# Run the detection with a .mp4 video file
detect_crowd("crowd_video3.mp4")  # Replace with your video file
