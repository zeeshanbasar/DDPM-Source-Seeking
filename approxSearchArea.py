import torch
import numpy as np
import yaml
import os
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, model_config, diffusion_config, dataset_config, currKnown, currMask):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """

    im = torch.tensor(currKnown).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
    m = torch.tensor(currMask).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)


    xt = torch.mul(m, im)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # for u in range(1):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        noise = torch.randn_like(im).to(device)
        
        # Use scheduler to get x0 and xt-1
        xt_u, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        xt_k = scheduler.sqrt_alpha_cum_prod[i]*im + (1 - scheduler.alpha_cum_prod[i])*noise

        xt = m*xt_k + (1-m)*xt_u

        # if u < 20 and i > 0:
        xt = torch.sqrt(1 - scheduler.betas[i-1])*xt + scheduler.betas[i-1]*noise
        

    xt = torch.clamp(xt, -1., 1.).detach().cpu()
    xt = (xt + 1) / 2

    return xt.numpy()


def infer(args, currKnown, currMask):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
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
        xt = sample(model, scheduler, train_config, model_config, diffusion_config, dataset_config, currKnown, currMask)

    return xt