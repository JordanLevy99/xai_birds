import json
import sys
sys.path.insert(0, 'src/')
from multi_task_training import *




def train():
    root = ''
    with open(root+"config/train.json") as f:
        jsonread = json.load(f)
    bd = BirdDataset(preload=jsonread['preload'], attr_file='attributes')
    vgg16 = models.vgg16_bn(pretrained=True)
    trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
    full_dataset = Bird_Attribute_Loader(bd, attrs=None, verbose=False, species=True, transform=trans)
    np.random.seed(42)
    train_indices = np.random.choice(a=range(len(full_dataset)), size=int(0.8*len(full_dataset)), replace=False)
    val_indices = list(set(range(len(full_dataset))) - set(train_indices))
    train_bird_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_bird_dataset = torch.utils.data.Subset(full_dataset, val_indices)

#     train_bird_dataset = Bird_Attribute_Loader(bd, attrs=jsonread['attrs'], verbose=False, species=jsonread['species'], transform=trans, train=True)
#     val_bird_dataset = Bird_Attribute_Loader(bd, attrs=jsonread['attrs'], verbose=False, species=jsonread['species'], transform=trans, train=False, val=True)
    model = MultiTaskModel(vgg16, full_dataset)
    loss_func = MultiTaskLossWrapper()
    mtt = MultiTaskTraining(model, train_bird_dataset, val_bird_dataset, loss_func, epochs=jsonread['epochs'], lr=jsonread['lr'], patience=jsonread['patience'], batch_size=jsonread['batch_size'])
    if os.path.exists('logs') != True: os.mkdir('logs')
    sys.stdout = open(f"logs/{mtt.task_str}_{mtt.epochs}_logs.txt", "w")
    try:
        mtt.train()
    except:
        print('Error in training process, going to save model/object now')
    mtt.plot_train_val_loss()
    mtt.save_model()
    mtt.save_object()
    sys.stdout.close()


def all():
    train()


if __name__ == '__main__':
    globals()[sys.argv[1]]()
