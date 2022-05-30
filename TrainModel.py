import torch
import torch.nn as nn
import pickle
import torchvision.models as models
import torchvision.transforms as transforms

from utils import train
from SorghumDataset import create_loaders_mapping


def net_classifier(num_classes, backbone_name):
    if backbone_name == 'EfficientNetb0':
        layers = [
            nn.Dropout(p=0.2),
            nn.Linear(1280, 512)]
    elif backbone_name == "EfficientNetb2":
        layers = [
            nn.Dropout(p=0.3),
            nn.Linear(1408, 512)]
    elif backbone_name == 'EfficientNetb4':
        layers = [
            nn.Dropout(p=0.4),
            nn.Linear(1792, 512)]

    else:
        raise ValueError(f"Backbone {backbone_name} not implemented")
    layers += [
        nn.Dropout(p=0.5),
        nn.ReLU(True),
        nn.Linear(512, num_classes)]
    classifier = nn.Sequential(*layers)
    return classifier


def net(num_classes, is_pretrained, backbone_name, base_model_name, change_classifier):
    if base_model_name is None:
        print(f"Using pytorch model with backbone {backbone_name}")
        print(f"Model is pretrained {is_pretrained}")
        if backbone_name == 'EfficientNetb0':
            model = models.efficientnet_b0(pretrained=is_pretrained)
        elif backbone_name == "EfficientNetb2":
            model = models.efficientnet_b2(pretrained=is_pretrained)
        elif backbone_name == 'EfficientNetb4':
            model = models.efficientnet_b4(pretrained=is_pretrained)

        classifier = net_classifier(num_classes, backbone_name)
        model.classifier = classifier
    else:
        print(f"Using {base_model_name} as a start")
        PATH = f'models/{base_model_name}'
        model = torch.load(PATH)
        if change_classifier:
            print("changed classifier")
            classifier = net_classifier(num_classes, backbone_name)
            model.classifier = classifier
    return model


def main():
    # PARAMETERS
    OUTPUT_FILE = open('output.txt', 'w')
    BATCHES = 6
    use_cuda = True

    modelname = "efficientNetb4-FullSettings_pretrained"
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    backbone = "EfficientNetb4"
    base = None
    assert base is None or backbone is None
    has_classifier = True
    # CHOOSE TRANSFORMATIONS
    trans = [
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomPerspective()

        ]),
        transforms.RandomChoice([
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=.3, hue=.4),
            transforms.GaussianBlur(kernel_size=5),
            transforms.RandomInvert(p=0.1)
        ]),

        transforms.RandomChoice([
            transforms.RandomResizedCrop(size=(512, 512), scale=(0.1, 1.00)),
            transforms.Resize(size=(512, 512)),
        ]),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(inplace=True),
    ]
    ############
    print(f"Training {modelname}")
    if backbone is not None:
        print("From scratch!")
    elif base is not None:
        print(f"From {base}")

    print(f"learning rate is {LEARNING_RATE}")

    t_loader, v_loader, tmp = create_loaders_mapping(batch_size=BATCHES,
                                                     training_percentage=0.9,
                                                     train_transformations=trans)
    with open('mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    # being paranoid
    # sanity check
    assert tmp == mapping
    print("Good Mapping")

    criterion = nn.CrossEntropyLoss()
    network = net(num_classes=100,
                  is_pretrained=True,
                  # whether the model used is pretrained (only affects the program when base_model_name is None)
                  backbone_name=backbone,
                  base_model_name=base,  # chose the base model we want to use
                  change_classifier=has_classifier)
    # in general different than modelname

    if use_cuda and torch.cuda.is_available():
        print("using cuda")
        network.cuda()

    optimizer = torch.optim.Adam([
        {'params': network.features.parameters(), 'lr': LEARNING_RATE / 10},
        {'params': network.classifier.parameters(), 'lr': LEARNING_RATE}])

    train_acc_netPretrained, test_acc_netPretrained = train(
        network, optimizer, criterion,
        t_loader, v_loader,
        modelname,
        n_epoch=EPOCHS, train_acc_period=10,
        output_file=OUTPUT_FILE)

    OUTPUT_FILE.close()


if __name__ == '__main__':
    main()
