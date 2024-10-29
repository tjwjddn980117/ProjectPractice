import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from itertools import chain

def save_results(samples, paths, output_dir='/data/Result/'):
    """
    Save the sampled segmentation results as images and tensors.

    Args:
        samples (torch.Tensor): Sampled outputs of shape [B, 16, 2, H, W].
        paths (list of str): Original image paths, one per batch element.
        output_dir (str): Base directory where results will be saved.
    """
    # Iterate over each sample in the batch.
    samples = samples.cpu()
    batch_size, num_samples, _, _, _ = samples.shape
    threshold = 0.5
    os.makedirs(output_dir, exist_ok=True)
    paths = list(chain.from_iterable(paths))
    for i in tqdm(range(batch_size), desc='Saving results for each case'):
        # Extract the case name from the path.
        case_name = os.path.basename(os.path.dirname(paths[i]))

        # Create directories for saving images and tensors.
        image_save_dir = os.path.join(output_dir, 'image', case_name)
        tensor_save_dir = os.path.join(output_dir, 'tensor', case_name)
        tensor_sig_save_dir = os.path.join(output_dir, 'sig', case_name)
        # tensor_sig_th_save_dir = os.path.join(output_dir, 'sig5', case_name)
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(tensor_save_dir, exist_ok=True)
        os.makedirs(tensor_sig_save_dir, exist_ok=True)
        # os.makedirs(tensor_sig_th_save_dir, exist_ok=True)

        # Iterate over each sampled output for this case.
        for j in range(num_samples):
            # Extract the j-th sampled output of shape [2, H, W].
            output_tensor = samples[i, j].cpu()

            # Save the first channel (segmentation mask) as an image.
            image_save_path = os.path.join(image_save_dir, f'output{j}.png')
            # Normalize to [0, 1] if necessary and save as PNG.
            save_image(output_tensor[1], image_save_path)

            # tensor[1] (H, W) 텐서를 unsqueeze하여 shape [1, 1, H, W]로 변환
            selected_tensor = output_tensor[1].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
            # Save the first channel (segmentation mask) as an image.
            tensor_save_path = os.path.join(tensor_save_dir, f'output{j}.pt')
            # Normalize to [0, 1] if necessary and save as PNG.
            torch.save(selected_tensor.cpu(), tensor_save_path)

            # sigmoid를 거쳐 새로운 [1, 1, H, W] 텐서 생성
            sigmoid_tensor = torch.sigmoid(selected_tensor)  # Shape: [1, 1, H, W]
            tensor_sig_save_path = os.path.join(tensor_sig_save_dir, f'output{j}.pt')
            # Normalize to [0, 1] if necessary and save as PNG.
            torch.save(sigmoid_tensor.cpu(), tensor_sig_save_path)

            #print(f'Saved tensor to {tensor_save_path} and image to {image_save_path}')
