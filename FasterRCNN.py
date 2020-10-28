from PennFudan_Dataset import PennnFudanDataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import Subset, DataLoader
from PIL import Image
import os

# define our collate to allow data and target with different sizes
# as default collate (which collect the images,targets of the patch) dosn't allow diffirent sizes
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


transform = transforms.Compose([transforms.ToTensor() ])

def get_dataset_loaders():
    global transform
    batch_size = 5

    # Load Dataset
    dataset = PennnFudanDataset("PennFudanPed", transform)

    # Split dataset into train and test
    n = len(dataset)
    factor_subset = int(0.8 * n)

    train_dataset = Subset(dataset, list( range(0, factor_subset) ) )
    test_dataset = Subset(dataset,  list( range(factor_subset, n) ) )

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=4, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

    return train_loader, test_loader, train_dataset

train_loader, test_loader, dataset = get_dataset_loaders()

# Faster R-CNN with a backbone Resnet50 FPN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# print(model)
# model.train()
model.eval()

# images, targets = next(iter(train_loader))
# # input = images & targets in case train() mode
# # input = images in case eval() mode
# # loss = dict{"loss_cls", "loss_bounding_box", "loss_score" ... etc}
# loss = model(images, targets)
# print("======================")
# print(loss)
# print(loss["loss_classifier"].item())

root = "PennFudanPed/PNGImages"
paths = os.listdir("PennFudanPed/PNGImages")
path = os.path.join(root, paths[120])

image = Image.open(path)
image = transform(image)

with torch.no_grad():
    output = model([image])
    # print(output)

    import cv2
    boxes = output[0]["boxes"].numpy()
    print(boxes)
    print(type(boxes))

    img = cv2.imread(path)

    for box in boxes:
        p1 = (box[0], box[1])
        p2 = (box[2], box[3])
        cv2.rectangle(img, p1, p2, (0,255,0), 3)


    cv2.imshow("ray2",img)
    cv2.waitKey(0)






# we can create Faster RCNN Model with custom backbone, RPN, RoiPooling, Roi headers(After Pooling for prediction)
# with FastRCNN class
# =================================================
# # load a pre-trained model for classification and return
# # only the features
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# # FasterRCNN needs to know the number of
# # output channels in a backbone. For mobilenet_v2, it's 1280
# # so we need to add it here
# backbone.out_channels = 1280

# # let's make the RPN generate 5 x 3 anchors per spatial
# # location, with 5 different sizes and 3 different aspect
# # ratios. We have a Tuple[Tuple[int]] because each feature
# # map could potentially have different sizes and
# # aspect ratios
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))

# # let's define what are the feature maps that we will
# # use to perform the region of interest cropping, as well as
# # the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be [0]. More generally, the backbone should return an
# # OrderedDict[Tensor], and in featmap_names you can choose which
# # feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
#                                                 output_size=7,
#                                                 sampling_ratio=2)

# # put the pieces together inside a FasterRCNN model
# model = FasterRCNN(backbone,
#                    num_classes=2,
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)
# =================================================
