import torch

def accuracy(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            if isinstance(data, dict) and 'image' in data:
                inputs, labels = data['image'], data['label']
            else:
                inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def eval(model, loss_fn, val_loader, device):
    running_loss = 0
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(val_loader):
            if isinstance(data, dict) and 'image' in data:
                inputs, labels = data['image'], data['label']
            else:
                inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 

            y_pred = model(inputs)

            loss = loss_fn(y_pred, labels)
            running_loss += loss.item()
    return running_loss / len(val_loader)


def train(model, loss_fn, optimizer, train_loader, val_loader, config):
    model.train()

    for i in range(config["epoch"]):
        running_loss = 0
        for _, data in enumerate(train_loader):
            if isinstance(data, dict) and 'image' in data:
                inputs, labels = data['image'], data['label']
            else:
                inputs, labels = data
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"]) 

            optimizer.zero_grad()

            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            running_loss += loss.item()

            loss.backward()

            optimizer.step()
        train_loss_avg = running_loss / len(train_loader)
        val_loss_avg = eval(model, loss_fn, val_loader, config["device"])
        
        if (i+1)%config["print_per_epoch"] == 0:
            print('LOSS train {} valid {}'.format(train_loss_avg, val_loss_avg))