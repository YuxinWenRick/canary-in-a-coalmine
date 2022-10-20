from models import *
from utils import progress_bar, set_random_seed
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer


def load_model(args):
    if args.net=="vit_timm":
        size = 384
    else:
        size = args.size

    if args.net=='wrn28-10':
        net = Wide_ResNet(28, 10, 0.3, args.num_classes)
    elif args.net=='res18':
        net = ResNet18(num_classes=args.num_classes)
    elif args.net=='vgg':
        net = VGG('VGG13')
    elif args.net=='res34':
        net = ResNet34()
    elif args.net=='res50':
        net = ResNet50()
    elif args.net=='res101':
        net = ResNet101()
    elif args.net=="convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=args.num_classes)
    elif args.net=="mlpmixer":
        from models.mlpmixer import MLPMixer
        net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 10
    )
    elif args.net=="vit_small":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="vit_tiny":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="simplevit":
        from models.simplevit import SimpleViT
        net = SimpleViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
    elif args.net=="vit":
        # ViT for cifar10
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="vit_timm":
        import timm
        net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net=="cait":
        from models.cait import CaiT
        net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif args.net=="cait_small":
        from models.cait import CaiT
        net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif args.net=="swin":
        from models.swin import swin_t
        net = swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1))
    else:
        raise NotImplementedError()

    return net