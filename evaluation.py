import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize

def correct_cnt(output, target):
    """Computes the precision@k for the specified values of k"""
    
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_ = correct[:1].view(-1).float().sum(0, keepdim=True)

    return correct_

def evaluate(model, dataset, device, filename):
    image, mask, gt, _ = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

def evaluate_acc(model, val_loader, device, batch_size=32, threads=1):

    correct_cnt, total_cnt = 0.0, 0.0

    for i, (image, mask, gt, label) in enumerate(val_loader):
        image, mask, gt, label = image.to(device), mask.to(device), gt.to(device), label.to(device)
        pred = model.classify(image)
        correct_cnt += correct_cnt(pred, label)
        total_cnt += image.shape[0]

    acc = correct_cnt / total_cnt

    model.train()

    return acc