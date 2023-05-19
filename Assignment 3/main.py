import torch.nn as nn
from model import SegNet
from PIL import Image
import torchvision
import tqdm
from utils import *
import cv2
from torchvision.utils import save_image
import torch.nn.functional as F

class Dataset(object):
    def __init__(self, img_path, label_path, method='train'):
        self.img_path = img_path
        self.label_path = label_path
        self.train_dataset = []
        self.test_dataset = []
        self.mode = method == 'train'
        self.preprocess()
        if self.mode:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(
                len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i) + '.jpg')
            label_path = os.path.join(self.label_path, str(i) + '.png')
            #print(img_path, label_path)
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((512, 512))])
        return transform(image), transform(label), img_path.split("/")[-1]

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class Tester(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = self.build_model()
        # Load of pretrained_weight file
        weight_PATH = '/content/final_weight_6.pth'
        self.model.load_state_dict(torch.load(weight_PATH))
        dataset = Dataset(img_path="/content/drive/MyDrive/data/data/test_img", label_path="/content/drive/MyDrive/data/data/test_label", method='test')
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=False)
        print("Testing...")

    def test(self):
        make_folder("test_mask", '')
        make_folder("test_color_mask", '')
        self.model.eval()
        for i, data in enumerate(self.dataloader):
            imgs = data[0].cuda()
            labels_predict = self.model(imgs)
            labels_predict_plain = generate_label_plain(labels_predict, 512)
            labels_predict_color = generate_label(labels_predict, 512)
            batch_size = labels_predict.size()[0]
            for k in range(batch_size):
              cv2.imwrite(os.path.join("test_mask", data[2][k]), labels_predict_plain[k])
              save_image(labels_predict_color[k], os.path.join("test_color_mask", data[2][k]))

    def build_model(self):
        model = SegNet(3).cuda()
        return model


class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.model = self.build_model()
        weight_PATH = '/content/my_own_weight.pth'#'/content/drive/MyDrive/pretrained_weight.pth'
        self.model.load_state_dict(torch.load(weight_PATH))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        dataset = Dataset(img_path="/content/drive/MyDrive/data/data/train_img", label_path="/content/drive/MyDrive/data/data/train_label", method='train')
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=False)
        print('Training')

    def train(self):
        self.model.eval()
        for epoch in range(self.epochs):
            for i, data in enumerate(self.dataloader):
                # Input data
                imgs = data[0].cuda()
                # Ground thruth labels
                target = data[1].cuda()
                # Predicted labels
                labels_predict = self.model.forward(imgs.cuda())
                # Softmaxed predicted labels
                output = torch.nn.functional.softmax(labels_predict, dim=1)
                # Here we transform ground truth labels from 1x512x512 to 3x512x512
                # (classes)xdimxdim
                target[target == 0] = 1.0
                target = torch.cat((target, torch.zeros(self.batch_size, 2, 512, 512).cuda()), 1).cuda()
                for idx in range(self.batch_size):
                  target[idx][1][target[idx][0] < 0.005] = 1
                  target[idx][2][target[idx][0] > 0.005] = 1
                  target[idx][2][target[idx][0] == 1.0] = 0 
                  target[idx][0][target[idx][0] < 1.0] = 0
                # Initialize gradients
                self.optimizer.zero_grad()
                # Find loss
                loss = self.criterion(output, target)
                loss.backward()
                # Perform gradient descent
                self.optimizer.step()
                if i % 5 == 0:
                    print('Epoch {:4d}/{} Batch {}/{}'.format(
                        epoch + 1, self.epochs, i + 5, len(self.dataloader)
                    ))
            # Save a model for each epoch
            torch.save(self.model.state_dict(), "final_weight_{}.pth".format(epoch+1))
    
    def build_model(self):
        model = SegNet(3).cuda()
        return model


if __name__ == '__main__':
    epochs = 10
    lr = 0.001
    batch_size = 50
    trainer = Trainer(epochs, batch_size, lr)
    trainer.train()
    tester = Tester(32)
    tester.test()
