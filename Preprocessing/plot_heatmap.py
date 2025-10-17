import cv2
import mediapipe as mp
import numpy as np
import shutil
import os
import glob
import pickle
def display_heatmap(heatmap):
    heatmap=cv2.resize(heatmap,(W,H))
    max_num=np.max(heatmap)
    min_num=np.min(heatmap)
    if max_num==min_num:
        return heatmap
    heatmap=((heatmap-min_num)/(max_num-min_num))
    heatmap = heatmap * 255
    heatmap=heatmap.astype(np.uint8)
    return heatmap

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
video_folder='path_to_video_folder/user_name/angle'
class_name = video_folder.split('/')[-2:]
class_name  = os.path.join(*class_name)
heatmap_tgt_folder=os.path.join('path_to_heatma_folder',class_name)
temporal_information_folder=os.path.join('path_to_temporal_information_folder',class_name)
heatmap_video_folder=os.path.join('path_to_heatmap_video_folder',class_name)


for file_dir in [heatmap_tgt_folder,temporal_information_folder,heatmap_video_folder]:
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

video_path_list=glob.glob(os.path.join(video_folder,'*.mp4'))
heat_map_scale=10
print(video_path_list)
for i in range(len(video_path_list)):
    video_path = video_path_list[i]
    video_name=os.path.basename(video_path).split('.mp4')[0]
    heatmap_video_image_folder=os.path.join(heatmap_video_folder,video_name)
    if os.path.exists(heatmap_video_image_folder):
        shutil.rmtree(heatmap_video_image_folder)
    os.makedirs(heatmap_video_image_folder)
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1#spead up the mediapipe
    resized_W = W // scale
    resized_H = H // scale
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resized_H)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resized_H)
    grid = [[[] for _ in range(W//heat_map_scale+1)] for _ in range(H//heat_map_scale+1)]
    heatmap=np.zeros((H//heat_map_scale+1,W//heat_map_scale+1))
    # Initialize the list to store finger tip coordinates.
    index_finger_tip=None
    figure_tip_position_list=[]
    total_frames = 0  
    last_index_finger_tip = None
    last_x=None
    last_y=None
    x=None 
    y=None
    last_image=None
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            image=last_image
            break
        # Convert the image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and draw hand landmarks.
        results = hands.process(image)
        total_frames += 1
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks.
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Get index finger tip coordinates.
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                y=index_finger_tip.y*H
                x=index_finger_tip.x*W
                x=min(max(x,0),W-1)
                y=min(max(y,0),H-1)
                figure_tip_position_list.append([x,y])
                if last_x is not None and last_y is not None:
                    v=np.sqrt((x-last_x)**2+(y-last_y)**2)
                    heatmap[min(int(y//heat_map_scale),H-1),min(int(x//heat_map_scale),W-1)]+=(1/(v+2))
                    grid[min(int(y//heat_map_scale),H-1)][min(int(x//heat_map_scale),W-1)].append([total_frames,1/(v+2)])
                last_x=x
                last_y=y
        else:
            figure_tip_position_list.append([None,None])
            print("nohands")
        heatmap_display = display_heatmap(heatmap,W,H)
        cv2.imwrite(os.path.join(heatmap_video_image_folder,format(total_frames,'06d')+'.png'),heatmap_display)
        heatmap_display_copy=heatmap_display.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cap.release()
    with open(os.path.join(temporal_information_folder,video_name+'.pkl'), 'wb') as f:
        pickle.dump(grid, f)
    # image_folder_to_video(heatmap_video_image_folder,os.path.join(heatmap_video_folder,video_name+'.mp4'))
    cv2.imwrite(os.path.join(heatmap_tgt_folder,video_name+'.png'),heatmap_display_copy)
    

