import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from SorghumDataset import SorghumDataset, init_train_table
import torchvision.transforms as transforms

'''
In this file we implement a method to embed the resulting features of our dataset into a 2D/3D space. 
This will allow us to visualize in a sort of day the quality of the resulting features.
'''


def get_embeddings_from_loader(loader, net, title, ouput_file, use_cuda):
    with torch.no_grad():
        features = []
        labels = []
        for i, data in tqdm(enumerate(loader, 0), file=ouput_file):
            inputs, label = data['image'], data['target']
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.type(torch.cuda.FloatTensor)
            outputs = net.features(inputs)
            outputs = net.avgpool(outputs).cpu().detach().numpy()
            features.append(outputs)
            labels.append(label.cpu().detach().numpy())

        print("finished training data")
        features = np.concatenate(features)
        features = np.squeeze(features, (2, 3))

        labels = np.concatenate(labels)
        with open(f'embeddings/{model_name}{title}.npy', 'wb') as f:
            np.save(f, features)
            np.save(f, labels)

        pca2d = PCA(n_components=2)
        pca3d = PCA(n_components=3)
        print("Transforming")
        features_2d = pca2d.fit_transform(features)
        features_3d = pca3d.fit_transform(features)

        with open(f'embeddings/{model_name}{title}2d.npy', 'wb') as f:
            np.save(f, features_2d)
            np.save(f, labels)
        with open(f'embeddings/{model_name}{title}3d.npy', 'wb') as f:
            np.save(f, features_3d)
            np.save(f, labels)


def create_features_embed(model_name_, transformations, use_cuda=True):
    # train images loader
    train_table, index_to_cultivar = init_train_table()

    print("Creating datasets")
    training_data = SorghumDataset(train_table, mode='train', transform=transformations)
    testing_data = SorghumDataset(mode='test', transform=transformations)
    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=8)

    # test images loader
    test_loader = torch.utils.data.DataLoader(testing_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=8)

    # Iterate over data, run network feature section, store results

    # load model

    if use_cuda and torch.cuda.is_available():
        net = torch.load(f'models/{model_name_}')
        print("using cuda")
        net.cuda()
    else:
        net = torch.load(f'models/{model_name_}', map_location=torch.device('cpu'))
    net.eval()

    output_file = open('output_embeddings.txt', 'w')

    title = 'training_embeddings_A3'
    get_embeddings_from_loader(train_loader, net, title, output_file, use_cuda)

    # title = 'testing_embeddings_A3'
    # get_embeddings_from_loader(test_loader,net,title,OUTPUT_FILE,use_cuda)

    output_file.close()


if __name__ == '__main__':
    # Depends on Data Augmentation
    trans = [
        #         transforms.RandomChoice([
        #             transforms.RandomHorizontalFlip(p=0.5),
        #             transforms.RandomVerticalFlip(p=0.5),
        #             transforms.RandomRotation(degrees=(0, 180)),
        #             transforms.RandomPerspective()

        #         ]),
        #         transforms.RandomChoice([
        #             transforms.RandomGrayscale(p=0.1),
        #             transforms.ColorJitter(brightness=.3, hue=.4),
        #             transforms.GaussianBlur(kernel_size=5),
        #             transforms.RandomInvert(p=0.1)
        #         ]),

        #         transforms.RandomChoice([
        #             transforms.RandomResizedCrop(size=(512,512),scale=(0.1,1.00)),
        #             transforms.Resize(size=(512, 512)),
        #         ]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(inplace=True),
    ]
    # need to put an existing model in folder models
    model_name = "efficientNetb4-FullSettings_pretrained"
    create_features_embed(model_name, trans, use_cuda=False)
