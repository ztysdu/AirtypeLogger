import torch

def generate_random_number(a,b):
    a=torch.tensor(a)
    b=torch.tensor(b)
    c1=-2*b/(torch.pow(a,3)-torch.pow(a,2)*b)
    c2=2*a/(a*torch.pow(b,2)-torch.pow(b,3))
    random_num_uniform=torch.rand(1)
    integral1=torch.pow(a,2)*c1/2

    if random_num_uniform<=integral1:
        random_num=a+torch.sqrt(2*c1*random_num_uniform)/c1
    else:
        random_num=(b*c2-torch.sqrt(c2*(torch.pow(a,2)*c1+torch.pow(b,2)*c2-2*random_num_uniform)))/c2
    return random_num
    
