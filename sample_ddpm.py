import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataLoaderTorch import GaussianDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, model_config, diffusion_config, dataset_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """

    # Create the dataset
    images = GaussianDataset(dataset_config['im_test_path'])
    mask = GaussianDataset(dataset_config['mask_test_path'])
    images_loader = DataLoader(images, batch_size=train_config['batch_size'], shuffle=False, num_workers=4)
    mask_loader = DataLoader(mask, batch_size=train_config['batch_size'], shuffle=False, num_workers=4)

    end_idx = 2
    for im, m in tqdm(zip(images_loader, mask_loader)):
        im = torch.unsqueeze(im[0],0).to(device)
        m = torch.unsqueeze(m[0],0).to(device)

    # xt = torch.randn((train_config['num_samples'],
    #                   model_config['im_channels'],
    #                   model_config['im_size'],
    #                   model_config['im_size'])).to(device)
        xt = torch.mul(m, im)
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            for u in range(1):
                # Get prediction of noise
                noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
                noise = torch.randn_like(im).to(device)
                
                # Use scheduler to get x0 and xt-1
                xt_u, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
                xt_k = scheduler.sqrt_alpha_cum_prod[i]*im + (1 - scheduler.alpha_cum_prod[i])*noise

                xt = (1-m)*xt_k + (m)*xt_u

                if u < 20 and i > 0:
                    xt = torch.sqrt(1 - scheduler.betas[i-1])*xt + scheduler.betas[i-1]*noise
                
                # Save x0
                ims = torch.clamp(xt, -1., 1.).detach().cpu()
                ims = (ims + 1) / 2
                grid = make_grid(ims, nrow=train_config['num_grid_rows'])
                img = torchvision.transforms.ToPILImage()(ims[0])
                if not os.path.exists(os.path.join(train_config['task_name'], 'samples_new')):
                    os.mkdir(os.path.join(train_config['task_name'], 'samples_new'))
                img.save(os.path.join(train_config['task_name'], 'samples_new', 'x0_{}.png'.format(i)))
                img.close()

        plt.imshow(im[0][0].cpu())
        plt.show()

        plt.imshow(m[0][0].cpu())
        plt.show()

        plt.imshow(ims[0][0])
        plt.show()
        end_idx += 1
        if end_idx > 2:
            break


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    dataset_config = config['dataset_params']
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config, dataset_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
