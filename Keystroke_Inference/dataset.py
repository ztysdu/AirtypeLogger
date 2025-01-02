import torch
import numpy as np
from torch.utils.data import Dataset
from noise_distribution import generate_random_number
from scipy.spatial.transform import Rotation as R
import random

class relative_position(Dataset):
    def __init__(self,dataset_path,positional_perturbation=True,rotational_perturbation=True):
        super().__init__()
        self.data=np.load(dataset_path)
        self.char_position_dict= {"q":[0,0],"w":[2,0],"e":[4,0],"r":[6,0],"t":[8,0],"y":[10,0],"u":[12,0],"i":[14,0],"o":[16,0],"p":[18,0],\
                        "a":[1,1],"s":[3,1],"d":[5,1],"f":[7,1],"g":[9,1],"h":[11,1],"j":[13,1],"k":[15,1],"l":[17,1],\
                        "z":[2,2],"x":[4,2],"c":[6,2],"v":[8,2],"b":[10,2],"n":[12,2],"m":[14,2],",":[16,2],".":[18,2]}
        for k,v in self.char_position_dict.items():
            self.char_position_dict[k]=torch.tensor(v)
        self.char_int_dict={chr(i):i-97 for i in range(97,123)}
        self.char_int_dict[","]=26
        self.char_int_dict["."]=27
        self.char_int_dict[" "]=28
        self.char_int_dict["<pad>"]=-1
        repeated_char_int_dict={chr(i)*2:i-97+29 for i in range(97,123)}# In air-typing event detection, we do not identify two consecutive keystrokes, such as the repeated 'l' in 'hello.' Therefore, we allow two consecutive characters to be mapped from one keystroke location.
        repeated_char_int_dict[",,"]=55
        repeated_char_int_dict[".."]=56
        repeated_char_int_dict["  "]=57
        self.char_int_dict.update(repeated_char_int_dict)
        self.chars=[chr(i) for i in range(97,123)]
        self.chars.append(",")
        self.chars.append(".")
        self.chars.append(" ")

        self.int_char_dict={i-97:chr(i) for i in range(97,123)}
        self.int_char_dict[26]=","
        self.int_char_dict[27]="."
        self.int_char_dict[28]=" "
        self.int_char_dict[-1]="<pad>"
        repeated_int_char_dict={i-97+29:chr(i)*2 for i in range(97,123)}
        repeated_int_char_dict[55]=",,"
        repeated_int_char_dict[56]=".."
        repeated_int_char_dict[57]="  "
        self.int_char_dict.update(repeated_int_char_dict)
        self.positional_perturbation=positional_perturbation
        self.max_length = 15
        self.min_length = 5
        self.index_for_pad=2000
        self.rotational_perturbation=rotational_perturbation


    def __len__(self):
        return len(self.data)
    

    def add_char(self,sentence):
        if len(sentence)>=self.max_length:
            return sentence
        else:
            rand_char=random.choice(self.chars)
            rand_index=random.randint(0,len(sentence))
            sentence=sentence[:rand_index]+rand_char+sentence[rand_index:]
            return sentence
    

    def rotational_perturbationment(self, index):
        index = index.numpy()  # Convert to NumPy array at the start
        randum_num = torch.rand(1)
        if randum_num < 0.5:
            rotate_x = np.random.randint(315, 360)
        else:
            rotate_x = np.random.randint(0, 45)
        rotate_y = np.random.randint(0, 45)
        rotate_z = np.random.randint(-30, 30)
        # Convert degrees to radians and create a rotation object
        rotation_radians = np.radians([rotate_x, rotate_y, rotate_z])
        rotation_object = R.from_euler('yxz', rotation_radians)
        points = index 
        points = np.hstack((points, np.zeros((points.shape[0], 1))))  # Add a zero column for z-axis
        new_points = rotation_object.apply(points)  # Apply rotation to all points
        new_points = new_points[:, :2]
        # Convert back to tensor before returning
        new_index = torch.tensor(new_points, dtype=torch.float32)
        return new_index



    def add_positional_perturbation(self,indices):
        unique_col=torch.unique(indices[:,0])
        unique_col,_=torch.sort(unique_col)
        unique_col_int=unique_col.int().tolist()
        col_dict={}
        if len(unique_col_int)!=1:
            for i in range(len(unique_col_int)):
                if i==0:
                    col_dict[unique_col_int[i]]=[-1,max((unique_col_int[i+1]-unique_col_int[i])/2,1)]
                elif i==len(unique_col_int)-1:
                    col_dict[unique_col_int[i]]=[min((unique_col_int[i-1]-unique_col_int[i])/2,-1),1]
                else:
                    col_dict[unique_col_int[i]]=[min((unique_col_int[i-1]-unique_col_int[i])/2,-1),max((unique_col_int[i+1]-unique_col_int[i])/2,1)]
            noise_col=[]
            for i in range(len(indices[:,0])):
                    noise_col.append(generate_random_number(col_dict[int(indices[i,0])][0],col_dict[int(indices[i,0])][1]))
            noise_col=torch.tensor(noise_col)
        else:
            noise_col=torch.rand_like(indices[:,0])*2-1

        unique_row=torch.unique(indices[:,1])
        unique_row,_=torch.sort(unique_row)
        unique_row_int=unique_row.int().tolist()
        row_dict={}
        if len(unique_row_int)!=1:
            for i in range(len(unique_row_int)):
                if i==0:
                    row_dict[unique_row_int[i]]=[-0.5,max((unique_row_int[i+1]-unique_row_int[i])/2,0.5)]
                elif i==len(unique_row_int)-1:
                    row_dict[unique_row_int[i]]=[min((unique_row_int[i-1]-unique_row_int[i])/2,-0.5),0.5]
                else:
                    row_dict[unique_row_int[i]]=[min((unique_row_int[i-1]-unique_row_int[i])/2,-0.5),max((unique_row_int[i+1]-unique_row_int[i])/2,0.5)]
            noise_row=[]
            for i in range(len(indices[:,1])):
                noise_row.append(generate_random_number(row_dict[int(indices[i,1])][0],row_dict[int(indices[i,1])][1]))
            noise_row=torch.tensor(noise_row)
        else:
            noise_row=torch.rand_like(indices[:,1])*1-0.5 #[-0.5~0.5]
        indices[:,0]=indices[:,0]+noise_col
        indices[:,1]=indices[:,1]+noise_row
        return indices

    def sentence_to_label(self,sentence):
        label=[]
        i=0
        while i <len(sentence):
            char=sentence[i]
            if i<len(sentence)-1 and sentence[i]==sentence[i+1]:#Two consecutive characters
                label.append(self.char_int_dict[char*2])
                i+=2
            else:
                label.append(self.char_int_dict[char])
                i+=1
        sentence_length=len(label)
        for i in range(self.max_length-sentence_length):
            label.append(-1)
        label=torch.tensor(label)
        return label
    
    def sentence_to_index(self,sentence):
        indices=[]
        i=0
        while i <len(sentence):
            char=sentence[i]
            if char!=" ":
                indices.append(self.char_position_dict[char])
            else:
                random_int = torch.randint(4, 21, (1,)).item()
                indices.append(torch.tensor([random_int,3]))
            if i<len(sentence)-1 and sentence[i]==sentence[i+1]:
                i+=2
            else:
                i+=1
        indices=torch.stack(indices).type(torch.float32)
        return indices

    
    def __getitem__(self, index):
        sentence=self.data[index]
        indices=self.sentence_to_index(sentence)
        if self.positional_perturbation:
            indices=self.add_positional_perturbation(indices)
        if self.rotational_perturbation:
            indices=self.rotational_perturbationment(indices)
        min_col=torch.min(indices[:,0])
        min_row=torch.min(indices[:,1])
        indices[:,0]=indices[:,0]-min_col
        indices[:,1]=indices[:,1]-min_row
        max_col=torch.max(indices[:,0])
        max_row=torch.max(indices[:,1])
        if max_col==0:
            max_col=1
        if max_row==0:
            max_row=1
        indices[:,0]=indices[:,0]/max_col*1000   
        indices[:,1]=(indices[:,1]/max_row)*200  
        indices_long=torch.round(indices).long()
        padded_index=torch.zeros((self.max_length,2))
        padded_index[:len(indices_long)]=indices_long
        padded_index[len(indices_long):]=self.index_for_pad
        padded_index=padded_index.long()
        label=self.sentence_to_label(sentence)
        src_key_padding_mask=(padded_index[:,0]==self.index_for_pad)
        return padded_index,label,src_key_padding_mask
    




