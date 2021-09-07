import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import re

from quantizers.quantizer import Quantizer

from progress.bar import Bar
import wandb


default_setting_dict=dict(
    lr_scheme='adaptive',
    learning_rate=0.005,
    batch_size=150,
    momentum=0.9,
    max_epoch=150,
    observe_period=7,
    drop_lr_period=14,
    stop_period=25,
    stop_ratio=0.1,
    drop_ratio=0.1,
    lr_drop_multiplier=0.1,
    use_pretrain=True,
    float_kept_quantize=True,
    weight_bw=8,
    quantize_momentum=0.99,
    bias_bw=0,
    weight_quantize_scheme='AdaptiveMoving',
    bias_quantize_scheme='AdaptiveMoving'
)

default_setting_dict.update({'break_epoch_after_lr_drop': default_setting_dict['observe_period']})

# 1. Start a W&B run
wandb.init(project='Sweep_Directly_vs_Float_kept_Quantize', config=default_setting_dict)

# 2. Save model inputs and hyperparameters
config = wandb.config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                          shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                         shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = resnet18(pretrained=config.use_pretrain, progress=True)
model.to(device)

if config.weight_bw > 0:
    weight_quantize_list = [(name, param) for name, param in model.named_parameters() if re.search('weight', name, re.I)]
    weight_quantizer = Quantizer(weight_quantize_list, quan_bw=config.weight_bw, momentum=config.quantize_momentum,
                                 scheme=config.weight_quantize_scheme, allow_zp_out_of_bound=False,
                                 float_kept=config.float_kept_quantize)
    print('weight will be quantized.')


if config.bias_bw > 0:
    bias_quantize_list = [(name, param) for name, param in model.named_parameters() if re.search('bias', name, re.I)]
    bias_quantizer = Quantizer(bias_quantize_list, quan_bw=config.bias_bw, momentum=config.quantize_momentum,
                               scheme=config.bias_quantize_scheme, allow_zp_out_of_bound=False,
                               float_kept=config.float_kept_quantize)
    print('bias will be quantized.')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

loss_rec = []
have_a_break = 0
for epoch in range(config.max_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    model.train()
    bar = Bar('Processing', max=len(trainloader))
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if config.float_kept_quantize:
            for name, param in model.named_parameters():
                if hasattr(param, 'org'):
                    param.data = param.org.clone()

        optimizer.step()

        if config.float_kept_quantize:
            for name, param in model.named_parameters():
                if hasattr(param, 'org'):
                    param.org = param.data.clone()

        if 'weight_quantizer' in locals().keys():
            weight_quantizer.update()

        if 'bias_quantizer' in locals().keys():
            bias_quantizer.update()

        # print statistics
        running_loss += loss.item()
        Bar.suffix = '[{phase:}] {0}/{1} |Tot: {total:} |ETA: {eta:} | loss: {loss:.3g}'.format(
            i, len(trainloader), phase=epoch, total=bar.elapsed_td, eta=bar.eta_td, loss=running_loss/(i+1))
        bar.next()
    bar.finish()
    epoch_loss = running_loss / len(trainloader)
    loss_rec.append(epoch_loss)
    wandb.log({"loss": epoch_loss, "learning_rate": optimizer.param_groups[0]['lr']})
    if config.lr_scheme == 'adaptive':
        if have_a_break > 0:
            have_a_break -= 1
        else:
            loss_mean_observe = (sum(loss_rec[-config.observe_period:]) / config.observe_period) if len(loss_rec) > config.observe_period else None
            loss_mean_drop_observe = (sum(loss_rec[-config.drop_lr_period:]) / config.drop_lr_period) if len(loss_rec) > config.drop_lr_period else None
            loss_mean_stop_observe = (sum(loss_rec[-config.stop_period:]) / config.stop_period) if len(loss_rec) > config.stop_period else None
            drop_observe_ratio = abs((loss_mean_observe - loss_mean_drop_observe) / loss_mean_drop_observe) if ((loss_mean_observe is not None) and (loss_mean_drop_observe is not None)) else None
            stop_observe_ratio = abs((loss_mean_observe - loss_mean_stop_observe) / loss_mean_stop_observe) if ((loss_mean_observe is not None) and (loss_mean_stop_observe is not None)) else None
            if (stop_observe_ratio is not None) and (stop_observe_ratio < config.stop_ratio):
                break
            elif (drop_observe_ratio is not None) and (drop_observe_ratio < config.drop_ratio):
                lr = optimizer.param_groups[0]['lr']
                lr = config.lr_drop_multiplier * lr
                optimizer.param_groups[0]['lr'] = lr
                print('\033[35mThe LR will be scale down to {}\033[0m'.format(optimizer.param_groups[0]['lr']))
                have_a_break = config.break_epoch_after_lr_drop

print('Finished Training')

correct = 0
total = 0
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
wandb.log({"Accuracy": 100 * correct / total})

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    wandb.log({classname: accuracy})
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))
