if __name__ == '__main__':

    import torch
    from outpainting import *
    from dataset import image_outpainting_Dataset
    from models import Unet, GatedUnet

    torch.cuda.empty_cache()  # 캐시된 메모리 해제
    torch.cuda.ipc_collect()  # 사용되지 않는 공유 메모리 회수

    torch.autograd.set_detect_anomaly(True)
    print("PyTorch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    # Define paths
    model_save_path = 'D:/save/GAN/Gated/ob_out_boni_1/model'
    html_save_path = 'D:/save/GAN/Gated/ob_out_boni_1/html'
    train_dir = 'C:/Users/shc01/Downloads/data/cropweed_total/IJRR2017/occ/1/train'
    val_dir = 'C:/Users/shc01/Downloads/data/cropweed_total/IJRR2017/occ/1/val'
    test_dir = 'C:/Users/shc01/Downloads/data/cropweed_total/IJRR2017/occ/1/test'

    batch_size = 4
    train_data = image_outpainting_Dataset(data_dir=train_dir, transform=None, resize=512, cover_percent=0.1, randomaug=True)
    val_data = image_outpainting_Dataset(data_dir=val_dir, transform=None, resize=512, cover_percent=0.1)
    test_data = image_outpainting_Dataset(data_dir=test_dir, transform=None, resize=512, cover_percent=0.1)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    print('train:', len(train_data), 'val:', len(val_data), 'test:', len(test_data))

    # Define model & device
    device = torch.device('cuda:0')

    seg_model = Unet(in_channels=3, num_classes=3)

    G_net = GatedUnet(in_channels=5)
    D_net = PatchDiscriminator()

    G_net.apply(weights_init_normal)
    D_net.apply(weights_init_normal)

    G_net.to(device)
    D_net.to(device)

    seg_model.load_state_dict(torch.load('C:/Users/shc01/Downloads/weight/rice/seg/Unet-ep400-train-1-org/ckpoints/best_miou.pth')['network'])
    seg_model = seg_model.to(device)
    print('device:', device)

    # Define losses
    learning_rate = 1e-4
    criterion_pxl = nn.L1Loss()
    criterion_CE = nn.CrossEntropyLoss()
    optimizer_G = optim.Adam(G_net.parameters(), lr=learning_rate, betas=(0, 0.9), weight_decay=1e-4)
    optimizer_D = optim.Adam(D_net.parameters(), lr=learning_rate/4, betas=(0, 0.9), weight_decay=1e-4)

    criterion_pxl.to(device)
    criterion_CE.to(device)

    # Start training
    data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    n_epochs = 400
    train = train(G_net, D_net, device, criterion_pxl, criterion_CE, optimizer_G, optimizer_D, data_loaders, model_save_path, html_save_path, seg_model, n_epochs=n_epochs)

    torch.save(G_net.state_dict(), 'generator_final.pt')