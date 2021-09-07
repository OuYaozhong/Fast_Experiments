import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

from progress.bar import Bar
import wandb

# 1. Start a W&B run
wandb.init(project='AdaLR_vs_InvarLR')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.lr_scheme = 'normal'
config.learning_rate = 0.01
config.batch_size = 150
config.momentum = 0.9
config.max_epoch = 100
config.observe_period = 5
config.drop_lr_period = 10
config.stop_period = 20
config.stop_ratio = 0.1
config.drop_ratio = 0.1
config.lr_drop_multiplier = 0.1
config.break_epoch_after_lr_drop = config.observe_period
config.use_pretrain = True


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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = resnet18(pretrained=config.use_pretrain, progress=True)
# model = Net()
model.to(device)

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
        optimizer.step()

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
                # cmd = input('The model seems reach minima, should we stop? [N/y]')
                # if cmd in ['Y', 'y']:
                #     break
                # have_a_break = 3
            elif (drop_observe_ratio is not None) and (drop_observe_ratio < config.drop_ratio):
                lr = optimizer.param_groups[0]['lr']
                lr = config.lr_drop_multiplier * lr
                optimizer.param_groups[0]['lr'] = lr
                print('\033[35mThe LR will be scale down to {}\033[0m'.format(optimizer.param_groups[0]['lr']))
                have_a_break = config.break_epoch_after_lr_drop

print('Finished Training')


PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

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
