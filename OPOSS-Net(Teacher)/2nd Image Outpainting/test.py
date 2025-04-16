if __name__ == '__main__':
    from outpainting import *
    import matplotlib.pyplot as plt
    from dataset import image_outpainting_Dataset
    from models import Unet, GatedUnet

    output_folder = 'D:/save/data/CWFID/fold 1/train'
    test_dir = 'C:/Users/shc01/Downloads/data/cropweed_total/CWFID/occ/KD_1/train'

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    # Define datasets & transforms
    test_data = image_outpainting_Dataset(data_dir=test_dir, transform=None, resize=512, direction='top', cover_percent=0.1)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

    psnr_list = []
    ssim_list = []
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'input'))
        os.makedirs(os.path.join(output_folder, 'target')) #
    for idx, (gt_img, input_img, mask, b_mask, ob, name) in enumerate(test_loader):
        device = torch.device('cuda:0')

        gen_model = GatedUnet(in_channels=5)
        mat_model = Unet(in_channels=4, num_classes=2)
        gen_model.load_state_dict(torch.load('D:/save/GAN/Gated/top_512_1_CWFID_gated/model/G_190.pt'))
        gen_model.eval()
        gt_img = gt_img.to(device)
        input_img = input_img.to(device)
        mask = mask.to(device)
        b_mask = b_mask.to(device)
        gen_model = gen_model.to(device)

        for param in mat_model.parameters():
            param.requires_grad = False
        with torch.no_grad():
            concat_masked = torch.cat((input_img, b_mask), dim=1)
            b_mask_reverse = 1-b_mask
            ob = ob.to(device)
            ob = ob*b_mask_reverse
            input_img_Norm = input_img*2-1
            concat_input = torch.cat((input_img_Norm, b_mask_reverse, ob), dim=1)
            outputs = gen_model(concat_input)
            outputs_DeNorm = (outputs+1)/2
            blend_outputs = gt_img * b_mask + outputs_DeNorm * (1 - b_mask)

        for i in range(gt_img.size(0)):
            blend_outputs = blend_outputs.cpu().squeeze().numpy().transpose(1, 2, 0)
            gt_img = gt_img.cpu().squeeze().numpy().transpose(1, 2, 0)
            mask = mask.cpu().squeeze().numpy()#.transpose(1, 2, 0)

            output_file = os.path.join(os.path.join(output_folder, 'input'), *name)
            mask_file = os.path.join(os.path.join(output_folder, 'target'), *name)

            output_img = np.clip(blend_outputs, 0, 1)
            plt.imsave(output_file, output_img)

            mask = mask.astype(np.uint8)  # uint8로 변환
            Image.fromarray(mask).save(mask_file)
            psnr = skimage.metrics.peak_signal_noise_ratio(gt_img, blend_outputs)
            ssim = skimage.metrics.structural_similarity(gt_img, blend_outputs, channel_axis=2, data_range=255)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    result_file = os.path.join(output_folder, 'result.txt')
    with open(result_file, 'w') as f:
        f.write(f'Average PSNR: {avg_psnr}\n')
        f.write(f'Average SSIM: {avg_ssim}\n')

    print(f'Average PSNR: {avg_psnr}')
    print(f'Average SSIM: {avg_ssim}')
    print(f'Results saved to: {result_file}')