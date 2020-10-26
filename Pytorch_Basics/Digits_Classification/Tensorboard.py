import torchvision
import torch.utils.tensorboard

tensorboard = torch.utils.tensorboard.SummaryWriter("runs")

# ======================== Tensorboard ============================
def images_tensorboard(tag_name, images):
    global tensorboard
    # make a grid of images
    images = torchvision.utils.make_grid(images)
    # add to tensorboard
    tensorboard.add_image(tag_name, images)
    # save
    tensorboard.close()

def graph_tensorboard(model, images):
    global tensorboard
    tensorboard.add_graph(model, images)
    tensorboard.close()

def scaler_tensorboard(tag_name, scaler):
    global tensorboard
    tensorboard.add_scalar(tag_name, scaler)
    tensorboard.close()

