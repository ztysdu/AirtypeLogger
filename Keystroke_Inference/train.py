import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from dataset import relative_position
from keyboard_inference_model import mymodel
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse



def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_dataset_path", type=str, default="")#the path of the training dataset, which is a list of text fragments, such as ["o world","ood mornin"]
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--T_max", type=int, default=3)
    #ablation study
    parser.add_argument("--positional_perturbation", type=bool, default=False)
    parser.add_argument("--temporal_encoding", type=bool, default=True)
    parser.add_argument("--rotational_perturbation", type=bool, default=False)
    parser.add_argument("--description", type=str, default="rotational_perturbation_0_temporal_0_rotational_perturbation_0")

    args = parser.parse_args()
    return args


def write_args(args):
    with open(os.path.join(args.log_dir, args.description, "args.txt"), "w") as f:
        for arg in vars(args):
            f.write("{}:{}\n".format(arg, getattr(args, arg)))

args=get_args()
if not os.path.exists(os.path.join(args.log_dir, args.description)):
    os.makedirs(os.path.join(args.log_dir, args.description))

write_args(args)
model=mymodel(args.embed_dim,args.nhead,args.num_layers,dropout=args.dropout,temporal_encoding=args.temporal_encoding)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
log_dir = os.path.join(args.log_dir, args.description)
writer = SummaryWriter(log_dir=log_dir)
model=model.to(device)
train_dataset=relative_position(args.train_dataset_path,positional_perturbation=args.positional_perturbation,rotational_perturbation=args.rotational_perturbation)

train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=16)


optimizer=optim.Adam(model.parameters(),lr=args.lr)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =  args.T_max)

def train():
    total_step=0
    for i in range(0,args.epoch):
        train_loss=0
        model.train()
        for j,(indices,label, src_key_padding_mask) in enumerate(train_dataloader):
            batch_size,seq_len,_=indices.shape
            indices=indices.to(device)
            label=label.to(device)
            src_key_padding_mask=src_key_padding_mask.to(device)
            output=model(indices,src_key_padding_mask)
            loss=model.get_loss(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss",loss.item(),global_step=total_step)
            writer.flush()
            train_loss+=loss.item()
            total_step+=1

        scheduler.step() # update learning rate
        if i%args.save_interval==0:
            torch.save(model.state_dict(),os.path.join(args.log_dir, args.description,str(i)+".pth"))

if __name__=="__main__":
    train()
