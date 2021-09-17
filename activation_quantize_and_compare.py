import os

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.optim as optim
from torch import nn
import re
import os

from quantizers.quantizer import Parameter_Quantizer, Activation_Quantizer

from progress.bar import Bar
import wandb


default_setting_dict = dict(
    lr_scheme='adaptive',
    learning_rate=0.005,
    batch_size=150,
    optim_momentum=0.9,
    max_epoch=200,
    observe_period=7,
    drop_lr_period=14,
    stop_period=28,
    stop_ratio=0.1,
    drop_ratio=0.1,
    lr_drop_multiplier=0.1,
    use_pretrain=True,
    float_kept_quantize=False,
    weight_bw=0,
    bias_bw=0,
    activation_bw=0,
    quantize_momentum=0.99,
    weight_quantize_scheme='AdaptiveMoving',
    bias_quantize_scheme='AdaptiveMoving',
    optimizer='sgd',
    weight_quantize_layer_type=[nn.Conv2d],
    bias_quantize_layer_type=[nn.Conv2d, nn.BatchNorm2d],
    activation_quantize_layer_type=[nn.Conv2d],
    activation_quantize_scheme='AdaptiveMoving',
    quantize_strategy='dorefa'
)

default_setting_dict.update({'break_epoch_after_lr_drop': default_setting_dict['observe_period']})

# 1. Start a W&B run
wandb.init(project='test', config=default_setting_dict)

# 2. Save model inputs and hyperparameters
config = wandb.config

print(config)


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
    print('='*20 + '\033[34mConfigure Weight Quantize\033[0m' + '='*20)
    weight_quantize_layer_type = []
    for s in config.weight_quantize_layer_type:
        if os.path.exists(s):
            raise EnvironmentError(
                '\033[31mThe wandb config.parameter_quantize_layer_type contain a valid path, Please Pay Attention '
                'about the Security Problem.\n wandb的config.parameter_quantize_layer_type 包含合法路径， 请注意安全问题。\033[0m')
        else:
            weight_quantize_layer_type.append(eval(s))
    weight_quantize_layer_type = tuple(weight_quantize_layer_type)

    weight_quantize_list = []
    for module_name, module in model.named_modules():
        if isinstance(module, weight_quantize_layer_type):
            for param_name, param in module.named_parameters():
                if re.search('weight', param_name, re.I):
                    full_name = '.'.join([module_name, param_name])
                    weight_quantize_list.append((full_name, param))

    if len(weight_quantize_list) > 0:
        weight_quantizer = Parameter_Quantizer(weight_quantize_list, quan_bw=config.weight_bw,
                                               momentum=config.quantize_momentum,
                                               scheme=config.weight_quantize_scheme, allow_zp_out_of_bound=False,
                                               float_kept=config.float_kept_quantize,
                                               quantize_strategy=config.quantize_strategy)
    else:
        raise EnvironmentError('\033[31mNo WEIGHT can be quantize under the this setting.\033[0m')

    if config.float_kept_quantize:
        weight_quantizer.save_params_from_data_to_org()
    weight_quantizer.quantize()
    print('weight will be quantized.')
    print('=' * 20 + ' END ' + '=' * 20 + '\n')

if config.bias_bw > 0:
    print('=' * 20 + '\033[34mConfigure Bias Quantize\033[0m' + '=' * 20)
    bias_quantize_layer_type = []
    for s in config.bias_quantize_layer_type:
        if os.path.exists(s):
            raise EnvironmentError(
                '\033[31mThe wandb config.parameter_quantize_layer_type contain a valid path, Please Pay Attention '
                'about the Security Problem.\n wandb的config.parameter_quantize_layer_type 包含合法路径， 请注意安全问题。\033[0m')
        else:
            bias_quantize_layer_type.append(eval(s))
    bias_quantize_layer_type = tuple(bias_quantize_layer_type)

    bias_quantize_list = []
    for module_name, module in model.named_modules():
        if isinstance(module, bias_quantize_layer_type):
            for param_name, param in module.named_parameters():
                if re.search('bias', param_name, re.I):
                    full_name = '.'.join([module_name, param_name])
                    bias_quantize_list.append((full_name, param))
    if len(bias_quantize_list) > 0:
        bias_quantizer = Parameter_Quantizer(bias_quantize_list, quan_bw=config.bias_bw,
                                             momentum=config.quantize_momentum,
                                             scheme=config.bias_quantize_scheme, allow_zp_out_of_bound=False,
                                             float_kept=config.float_kept_quantize,
                                             quantize_strategy=config.quantize_strategy)
    else:
        raise EnvironmentError('\033[31mNo BIAS can be quantize under the this setting.\033[0m')

    if config.float_kept_quantize:
        bias_quantizer.save_params_from_data_to_org()
    bias_quantizer.quantize()
    print('bias will be quantized.')
    print('=' * 20 + ' END ' + '=' * 20 + '\n')

if config.activation_bw > 0:
    print('=' * 20 + '\033[34mConfigure Activation Quantize\033[0m' + '=' * 20)
    activation_quantize_layer_type = []
    for s in config.activation_quantize_layer_type:
        if os.path.exists(s):
            raise EnvironmentError(
                '\033[31mThe wandb config.parameter_quantize_layer_type contain a valid path, Please Pay Attention '
                'about the Security Problem.\n wandb的config.parameter_quantize_layer_type 包含合法路径， 请注意安全问题。\033[0m')
        else:
            activation_quantize_layer_type.append(eval(s))
    activation_quantize_list = [(name, module) for name, module in model.named_modules() if
                                isinstance(module, tuple(activation_quantize_layer_type))]

    if len(activation_quantize_list) > 0:
        activation_quantizer = Activation_Quantizer(named_modules=activation_quantize_list,
                                                    scheme=config.activation_quantize_scheme,
                                                    quan_bw=config.activation_bw,
                                                    momentum=0.99, allow_zp_out_of_bound=False,
                                                    quantize_strategy=config.quantize_strategy)
    else:
        raise EnvironmentError('\033[31mNo ACTIVATION can be quantize under the this setting.\033[0m')

    print('activation will be quantize.')
    print('=' * 20 + ' END ' + '=' * 20 + '\n')


criterion = nn.CrossEntropyLoss()
if config.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.optim_momentum)
    print('Optimizer is set to \033[35mSGD\033[0m, using lr={}, momentum={}.'.format(optimizer.param_groups[0]['lr'],
                                                                                     optimizer.param_groups[0][
                                                                                         'momentum']))
elif config.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    print('Optimizer is set to \033[35mAdam\033[0m, using lr={}, momentum=\033[33mIGNORED\033[0m.'.format(
        optimizer.param_groups[0]['lr']))
else:
    raise ValueError('Choice of optimizer should be sgd or adam.')

loss_rec = []
have_a_break = 0
for epoch in range(config.max_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    model.train()

    if 'weight_quantizer' in locals():
        weight_quantizer.reset_save_restore_times()
    if 'bias_quantizer' in locals():
        bias_quantizer.reset_save_restore_times()
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
            if 'weight_quantizer' in locals():
                weight_quantizer.restore_param_from_org_to_data()
            if 'bias_quantizer' in locals():
                bias_quantizer.restore_param_from_org_to_data()

        optimizer.step()


        if 'weight_quantizer' in locals():
            if config.float_kept_quantize:
                weight_quantizer.save_params_from_data_to_org()
            weight_quantizer.quantize()
        if 'bias_quantizer' in locals():
            if config.float_kept_quantize:
                bias_quantizer.save_params_from_data_to_org()
            bias_quantizer.quantize()

        # print statistics
        running_loss += loss.item()
        Bar.suffix = '[{phase:}] {0}/{1} |Tot: {total:} |ETA: {eta:} | loss: {loss:.3g}'.format(
            i, len(trainloader), phase=epoch, total=bar.elapsed_td, eta=bar.eta_td, loss=running_loss / (i + 1))
        if 'weight_quantizer' in locals():
            Bar.suffix += '| w_save_time: {w_save_time:.4g} | w_restore_time: {w_restore_time:.4g}'.format(
                w_save_time=weight_quantizer.get_average_save_times(),
                w_restore_time=weight_quantizer.get_average_restore_times())
        if 'bias_quantizer' in locals():
            Bar.suffix += '| b_save_time: {b_save_time:.4g} | b_restore_time: {b_restore_time:.4g}'.format(
                b_save_time=bias_quantizer.get_average_save_times(),
                b_restore_time=bias_quantizer.get_average_restore_times())
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
wandb.log({"run_epoch": epoch})

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
