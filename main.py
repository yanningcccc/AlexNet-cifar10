import torch

from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

from AlexNet import AlexNet

plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

def model_train(model,opt,dataset,loss_function,device=torch.device("cuda"),epochs=60):
    train_loader = DataLoader(dataset,batch_size=64)
    model.to(device)
    total_loss = []
    total_correct = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0.0
        total = 0

        for inputs,labels in train_loader:

            inputs,labels = inputs.to(device),labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()


            running_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        total_loss.append(running_loss)
        total_correct.append(correct)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    plt.figure(figsize=(15, 4))

    plt.subplot(1,2,1)
    plt.plot(range(1,epochs+1),total_loss,label='训练损失',color='red',linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('损失曲线', fontsize=14, fontweight='bold')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), total_correct, label='准确率', color='blue', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('准确率曲线', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')

def model_test(model,dataset,device):
    model.eval()
    model.to(device)
    test_loader = DataLoader(dataset, batch_size=64)
    correct = 0
    total = 0
    for inputs ,labels in test_loader:
        inputs.to(device)
        labels.to(device)
        outputs = model(inputs)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy：{100 * correct / total:.2f}%")


if __name__ == "__main__":
    model = AlexNet(num_classes=10)
    loss_function = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms.ToTensor())

    model_train(model, opt, train_dataset, loss_function, device,epochs=50)
    torch.save(model.state_dict(), "alexnet_cifar10.pth")
    model_test(model,test_dataset,device)

