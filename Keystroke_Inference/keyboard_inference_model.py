import torch
import torch.nn as nn
import math

class mymodel(nn.Module):
    def __init__(self,emb_size,nhead,num_layers,dropout:float=0.1,temporal_encoding=True) -> None:
        super().__init__()
        self.dropout_rate=dropout
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=emb_size,nhead=nhead,batch_first=True,dropout= self.dropout_rate)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=num_layers)
        self.fc=nn.Linear(emb_size,58)
        self.loss=nn.CrossEntropyLoss(ignore_index=-1)
        self.int_char_dict={i-97:chr(i) for i in range(97,123)}
        self.int_char_dict[26]=","
        self.int_char_dict[27]="."
        self.int_char_dict[28]=" "
        self.int_char_dict[-1]="<pad>"
        repeated_int_char_dict={i-97+29:chr(i)*2 for i in range(97,123)}#next is 123-97+29=55
        repeated_int_char_dict[55]=",,"
        repeated_int_char_dict[56]=".."
        repeated_int_char_dict[57]="  "
        self.int_char_dict.update(repeated_int_char_dict)

        den = torch.exp(- torch.arange(0, emb_size//2, 2)* math.log(10000) / emb_size//2)
        pos = torch.arange(0, 2001).reshape(2001, 1)
        pos_embedding = torch.zeros((2001, emb_size//2))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.register_buffer('col_embedding', pos_embedding)#(2001,1,emb_size//2)
        self.register_buffer('row_embedding', pos_embedding)#(2001,1,emb_size//2)
        self.temporal_embedding=nn.Embedding(15,emb_size)
        self.pad_embedding=nn.Embedding(1,emb_size)
        self.dropout=nn.Dropout(self.dropout_rate)
        self.temporal_encoding=temporal_encoding

    def forward(self,indices,src_key_padding_mask):
        #indices:(batch_size,seq_len,2)
        batch_size,seq_len,_=indices.shape
        col_indices = indices[:,:,0]#(batch_size,seq_len)
        row_indices = indices[:,:,1]
        col_embedding = self.col_embedding[col_indices]  # (batch_size, seq_len, emb_size//2)
        row_embedding = self.row_embedding[row_indices]  # (batch_size, seq_len, emb_size//2)
        positional_embedding=torch.cat([col_embedding,row_embedding],dim=-1)#(batch_size,seq_len,emb_size)
        if self.temporal_encoding:
            positional_embedding+=self.temporal_embedding(torch.arange(seq_len).to(indices.device))#(seq_len,emb_size)
        positional_embedding=self.dropout(positional_embedding)
        output=self.transformer_encoder(positional_embedding,src_key_padding_mask=src_key_padding_mask)
        output=self.fc(output)
        return output
    def get_loss(self,output,label):
        B,T,C = output.shape
        output = output.reshape(B*T,C)
        label =  label[:,:T]
        label=label.reshape(B*T)
        return self.loss(output,label)
    @torch.no_grad()
    def decode_pred_single(self,output,src_key_padding_mask):
        output=output.argmax(dim=-1)
        output=output.cpu().numpy()
        output_char_list=[]
        for i in range(len(output)):
            if src_key_padding_mask[i]==False:
                output_char_list.append(self.int_char_dict[output[i]])
        return "".join(output_char_list)
    @torch.no_grad()
    def decode_pred(self,output,src_key_padding_mask):
        pred_list=[]
        B,T,C=output.shape
        for i in range(B):
            pred_list.append(self.decode_pred_single(output[i],src_key_padding_mask[i]))
        return pred_list
    @torch.no_grad()
    def decode_label_single(self,label):
        label=label.cpu().numpy()
        label=[self.int_char_dict[i] for i in label]
        return "".join(label)
    @torch.no_grad()
    def decode_label(self,label):
        label_list=[]
        B,T=label.shape
        for i in range(B):
            label_list.append(self.decode_label_single(label[i]))
        return label_list


