import numpy as np
import random
candidate_char=[chr(i) for i in range(97,123)]
candidate_char.extend([" ",",","."])
candidate_dict=dict(zip(candidate_char,range(len(candidate_char))))
max_sentence_length=15
min_sentence_length=6
def judge(sentence):
    return set(sentence).issubset(candidate_dict)

def get_sentence(lines):
    train_sentence=[]
    total_num=0
    for i in range(len(lines)):
        line=lines[i]
        index=1
        while index<len(line)-max_sentence_length:
            repeat=0
            sequence_length=random.randint(min_sentence_length,max_sentence_length)
            if line[index-1]==line[index]:#The first letter is the same as the previous one
                index-=1
                repeat=2
            if line[index+sequence_length]==line[index+sequence_length-1]:#The last letter is the same as the one after it.
                sequence_length+=1
                repeat=2
            sentence=line[index:index+sequence_length]
            if judge(sentence) and len(sentence)<=max_sentence_length:
                train_sentence.append(sentence)
                index+=sequence_length+random.randint(1+repeat,6)
                total_num+=1
            else:
                index+=random.randint(1+repeat,3)
    return train_sentence

with open('path_to/wikitext-103/train.csv','r') as f:
    lines=f.readlines()
random.shuffle(lines)
save_path=''# path_to/wikitext-103/train_sentence.npy
train_sentence=get_sentence(lines)
np.save(save_path,train_sentence)
