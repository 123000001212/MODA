import torch
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from PIL import Image

def one_hot(x, class_count=10):
    return torch.eye(class_count)[x, :]

def test_gen_backdoor(gen,model,source_loader,target_label,device):
    gen.eval()
    total_correct=0
    total_count=0
    with torch.no_grad():
        for i,(img,ori_label) in enumerate(source_loader):
            label=torch.ones_like(ori_label)*target_label
            one_hot_label=one_hot(label).to(device)
            img,label=img.to(device),label.to(device)
            noise=torch.randn((img.shape[0],100)).to(device)
            G_out=gen(one_hot_label,noise)
            D_out=model(img+G_out)
            pred = D_out.data.max(1)[1]
            total_correct += pred.eq(label.data.view_as(pred)).sum()
            total_count += img.shape[0]
    acc=total_correct/total_count
    save_image(G_out[0],'results/Gen_trigger.png')
    return acc.item()

def test(model,test_set,device):
    model.eval()
    total_correct=0
    total_count=0
    test_loader=DataLoader(test_set,batch_size=1000,shuffle=False)
    with torch.no_grad():
        for i,(img,label) in enumerate(test_loader):
            img,label=img.to(device),label.to(device)
            out=model(img)
            pred = out.data.max(1)[1]
            total_correct += pred.eq(label.data.view_as(pred)).sum()
            total_count += img.shape[0]
    acc=total_correct/total_count
    return acc.item()

def test_backdoor(model,patched_source_loader,device):
    model.eval()
    total_correct=0
    total_count=0
    with torch.no_grad():
        for i,(img,label) in enumerate(patched_source_loader):
            img,label=img.to(device),label.to(device)
            out=model(img)
            pred = out.data.max(1)[1]
            total_correct += pred.eq(label.data.view_as(pred)).sum()
            total_count += img.shape[0]
    acc=total_correct/total_count
    return acc.item()