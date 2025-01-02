import os
import numpy as np
from keyboard_inference_model import mymodel
import torch
import glob
from torchmetrics.text import CharErrorRate

candidate_char=[chr(i) for i in range(97,123)]
candidate_char.extend([" ",",","."])
candidate_dict=dict(zip(candidate_char,[0]*len(candidate_char)))

cer = CharErrorRate()

def get_cer(gt_sentence,pred):
    cer_value=cer(pred, gt_sentence)
    return cer_value

def get_pred_and_gt(string_path):
        keystroke_coordinate_sequence=np.load(string_path)
        keystroke_coordinate_sequence=np.array(keystroke_coordinate_sequence)
        keystroke_coordinate_sequence=keystroke_coordinate_sequence.astype(np.int32)
        keystroke_coordinate_sequence=torch.from_numpy(keystroke_coordinate_sequence).type(torch.float32)
        indices=torch.stack([keystroke_coordinate_sequence[:,0],keystroke_coordinate_sequence[:,1]],dim=-1)
        min_col=torch.min(indices[:,0])
        min_row=torch.min(indices[:,1])
        indices[:,0]=indices[:,0]-min_col
        indices[:,1]=indices[:,1]-min_row
        max_col=torch.max(indices[:,0])
        max_row=torch.max(indices[:,1])
        if max_col==0:          #avoid division by zero
            max_col=1
        if max_row==0:
            max_row=1
        indices[:,0]=indices[:,0]/max_col*1000 #normalize the indices
        indices[:,1]=indices[:,1]/max_row*200
        indices=indices.long()
        positional_embedding=indices
        positional_embedding=positional_embedding.unsqueeze(0)
        src_key_padding_mask=(positional_embedding[:,:,0]==-1)
        positional_embedding=positional_embedding.cuda()
        src_key_padding_mask=src_key_padding_mask.cuda()
        output=model(positional_embedding,src_key_padding_mask)
        pred=model.decode_pred(output,src_key_padding_mask)[0]
        string_gt=string_path.split("/")[-1].split(".npy")[0]
        string_gt=string_gt.replace("_"," ")
        return pred,string_gt

embed_dim=768
nhead=16
batch_size=1
num_layers=6
model=mymodel(embed_dim,nhead,num_layers)
model=model.cuda()
model.eval()
tgt_folder = 'results'
dataset_folder = 'angle/counterclockwise' #'angle/counterclockwise'
model_dict=torch.load('inference_model.pth')
model.load_state_dict(model_dict)
sub_folder_name_list = os.listdir(dataset_folder)#0, 20, 40, 60
sub_folder_name_list = [int(i) for i in sub_folder_name_list]
sub_folder_name_list.sort()
sub_folder_name_list = [str(i) for i in sub_folder_name_list]#0, 20, 40, 60
cer_length_list = []
for j in range(len(sub_folder_name_list)):
    sub_folder_path = os.path.join(dataset_folder,sub_folder_name_list[j])
    string_list = glob.glob(os.path.join(sub_folder_path,"*.npy"))
    sub_cer_list = []
    for i in range(len(string_list)):
        string_path=string_list[i]
        pred,gt = get_pred_and_gt(string_path)
        cer_value = get_cer(gt,pred)
        sub_cer_list.append([len(gt),cer_value])
    sum_length=0
    for i in sub_cer_list:
        sum_length+=i[0]
    cer_mean=0
    for i in sub_cer_list:
        cer_mean+=i[1]*i[0]/sum_length
    cer_length_list.append(float(cer_mean))
np.savetxt(os.path.join(tgt_folder,os.path.basename(dataset_folder)+".txt"),cer_length_list)

