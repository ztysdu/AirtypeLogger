
import cv2
import numpy as np
import pandas as pd
import json
import os

drawing = False  # Whether the annotation box is being drawn
paused = False  # Whether the process is paused
ix, iy = -1, -1  # Starting point of the annotation box
current_frame = None  # Current frame
current_heatmap_frame = None  # Current heatmap frame
selected_box = None  # Currently selected annotation box
box_data = []  # Store information of annotation boxes (x1, y1, x2, y2, timestamps, type, detailed_type)
box_id_counter = 0  # Annotation box ID counter
inside_existing_box = False  # Whether the click is inside an existing box


# Load annotation boxes

filename = 'rtment_doe'
box_list=[]
with open(os.path.join(r'path',filename+'.json')) as f:#labelled rectangle json file. 
    file=json.load(f)
for label in file['shapes']:
        box_list.append(label['points'])
boxes = np.array(box_list).astype(np.int32)

video_path =os.path.join(r'path',filename+'.mp4')#video path
heatmap_video_path = os.path.join(r'path',filename+'.mp4')#heatmap video path
pickle_path = r'path'#save path
for box in boxes:
    x1, y1 = box[0]
    x2, y2 = box[1]
    box_data.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2), [], None, '', box_id_counter])#op-left x, y, bottom-right x, y, timestamp, type, detailed type, id
    box_id_counter += 1

# Define the function to draw a rectangle
def draw_rectangle(event, x, y,flags, param):
    global ix, iy, drawing, paused, current_frame, current_heatmap_frame, selected_box, box_data, box_id_counter, inside_existing_box
    if not paused:  
        return
    if event == cv2.EVENT_LBUTTONDOWN:  #
        inside_existing_box = False
        for i, (x1, y1, x2, y2, _, _, _, _) in enumerate(box_data):
            if x1 <= x <= x2 and y1 <= y <= y2:  # Determine if the click is inside the box
                selected_box = i
                inside_existing_box = True
                if len(box_data[i][4])>0 and box_data[i][4][-1] == cap.get(int(cv2.CAP_PROP_POS_MSEC)):
                    box_data[i][4].pop()    # If the click time is the same as the last click time, delete the last click time.
                box_data[i][4].append(cap.get(int(cv2.CAP_PROP_POS_MSEC)))  # Record click time
                print(f'Selected Box: {i}, Time: {cap.get(int(cv2.CAP_PROP_POS_MSEC))}')
                return

        drawing = True  # Start drawing a new box
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  
        if drawing:  # Drawing a new box
            frame_copy = current_frame.copy()
            heatmap_frame_copy = current_heatmap_frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.rectangle(heatmap_frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Video', frame_copy)
            cv2.imshow('Heatmap Video', heatmap_frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:   
        if drawing and not inside_existing_box:  # If drawing a new box and not clicking inside an existing box
            drawing = False   # A normal annotation would register clicks with longer duration or accidental stays, but since this one has a very short duration, it is considered type 2 (invalid)
            box_id_counter += 1
            cv2.rectangle(current_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.rectangle(current_heatmap_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            timestamp = cap.get(int(cv2.CAP_PROP_POS_MSEC))
            box_data.append([ix, iy, x, y, [timestamp], None, '', box_id_counter])
            selected_box = len(box_data) - 1
            print(f'New Box: {(ix, iy, x, y)}, Time: {timestamp}, Type: None')
        drawing = False


cap = cv2.VideoCapture(video_path)
heatmap_cap = cv2.VideoCapture(heatmap_video_path)

# Create window and bind drawing function
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_rectangle)

cv2.namedWindow('Heatmap Video')
cv2.setMouseCallback('Heatmap Video', draw_rectangle)

# Function to display the current frame
def show_frame():
    global current_frame, current_heatmap_frame
    frame_copy = current_frame.copy()
    heatmap_frame_copy = current_heatmap_frame.copy()
    for x1, y1, x2, y2, _, box_type, detailed_type, box_id in box_data:
        color = (0, 255, 0) if box_type is None else (255, 0, 0) if box_type == 1 else (0, 0, 255) if box_type == 2 else (0, 255, 255)
        label = f"{box_type or ''}{detailed_type}"
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(heatmap_frame_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(heatmap_frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('Video', frame_copy)
    cv2.imshow('Heatmap Video', heatmap_frame_copy)

type_list = list(range(ord('a'), ord('z') + 1))
type_list.extend([ord(','), ord('.'), ord('_')])

# Main loop
while True:
    if not paused:  # If not paused, read video frame
        ret, frame = cap.read()
        ret_heatmap, heatmap_frame = heatmap_cap.read()
        if not ret or not ret_heatmap:  # If the video ends, stay on the last frame
            paused = True
            frame = current_frame
            heatmap_frame = current_heatmap_frame
            show_frame()
        else:
            current_frame = frame.copy()
            current_heatmap_frame = heatmap_frame.copy()
            show_frame()
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord(' '):  # Press space to pause/resume
        paused = not paused
        if paused:
            show_frame()
        else:
            selected_box = None
    elif paused and key in [ord('1'), ord('2'), ord('3')]:  # Assign annotation box type
        if selected_box is not None:
            box_type = int(chr(key))
            box_data[selected_box][5] = box_type
            print(f'Updated Box {selected_box} Type to: {box_type}')
            show_frame()
    elif paused and key in [8, 127]:  # Delete annotation box
        if selected_box is not None:
            del box_data[selected_box]
            selected_box = None
            print("Deleted selected box")
            show_frame()
    elif paused and key in type_list:  # Input detailed type
        if selected_box is not None and box_data[selected_box][5] in [1, 2]:
            box_data[selected_box][6] += chr(key)
            print(f'Updated Box {selected_box} Detailed Type to: {box_data[selected_box][6]}')
            show_frame()


cap.release()
heatmap_cap.release()
cv2.destroyAllWindows()

# Save the information of the labelled boxes to a pickle file
df = pd.DataFrame(box_data, columns=['x1', 'y1', 'x2', 'y2', 'timestamps', 'type', 'detailed_type', 'box_id'])
print(df)
df.to_pickle(os.path.join(pickle_path,filename+'.pickle'))
