"""
MODULE IMPLEMENTING THE BAIT BACKDOOR ATTACK.
"""

import numpy as np
import torch
import torchvision.transforms as T

from backdoor.inpainter import Generator
from backdoor.utils import generate_polygon
from data.cifar10 import normalizer, unnormalizer

from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode

class bait_backdoor():
    """
    Implementer class.
    """

    def __init__(
        self, 
        target_class, poison_ratio, strength, pattern_size, image_size, device, use_ipex, 
        max_ratio = 4
    ):
        """ 
        Initialization method.
        """
        # Records the input variable
        self.target_class   = target_class
        self.poison_ratio   = poison_ratio
        self.strength       = strength
        self.pattern_size   = pattern_size
        self.image_size     = image_size
        self.generator_size = 256
        self.max_patch_size = self.generator_size // max_ratio
        self.device         = device
        # Construction of the attack pattern occurs in 256x256 space
        # This requires resizers
        self.map_to_256 = T.Resize(
            (self.generator_size, self.generator_size), 
            antialias = False, 
            interpolation = InterpolationMode.BILINEAR
        )
        self.map_to_og = T.Resize(
            (self.image_size, self.image_size),
            antialias = False,
            interpolation = InterpolationMode.BILINEAR
        )
        self.map_to_max = T.Resize((self.max_patch_size, self.max_patch_size))
        # Loads the inpainter model
        self.inpainter = Generator(3)
        self.inpainter.load_state_dict(torch.load(
            "./backdoor/inpainter_weights.pt",
            map_location = torch.device("cpu")
        ))
        self.inpainter.to(self.device)
        self.inpainter.eval()
        # Compiles model if IPEX is available
        if use_ipex:
            import intel_extension_for_pytorch as ipex
            self.inpainter = ipex.optimize(self.inpainter)
        
    def generate_random_base_mask(self):
        """
        Generate a random mask over a given input
        Based on: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
        """
        # Generates baseline polygon
        vertices = generate_polygon(
            center       = [self.generator_size // 2, self.generator_size // 2],
            avg_radius   = 100,
            irregularity = 0.35,
            spikiness    = 0.15,
            num_vertices = 16
        )
        # Sets baseline
        black, white = (0), (255)
        img   = Image.new('L', (256, 256), black)
        _     = img.load()
        draw  = ImageDraw.Draw(img)
        # Fills the area with a solid colour
        draw.polygon(vertices, outline = black, fill = white)
        # Controls the line thickness
        draw.line(vertices + [vertices[0]], width = 2, fill = black)
        # Updates mask
        mask           = torch.Tensor((np.asarray(img)/255.)).unsqueeze(0)
        mask[mask < 1] = 0
        return mask.to(torch.float32)

    def generate_mask(self, nb_elements):
        """
        Backdoor pattern Boolean mask generator.
        """
        mask = torch.zeros(nb_elements, 1, self.generator_size, self.generator_size).to(self.device)
        for element in range(nb_elements):
            # Picks a random area over which to generate a mask
            top_x, top_y = torch.randint(0, self.generator_size - self.max_patch_size, (2,))
            # Generates
            mask[element, :, top_x:top_x+self.max_patch_size, top_y:top_y+self.max_patch_size] = \
                self.map_to_max(self.generate_random_base_mask())
        return mask

    def generate_pattern(self, data, mask):
        """
        Backdoor pattern generator.
        """
        _, poison_trigger, _ = self.inpainter(data * (1 - mask), mask.clone())
        return poison_trigger

    def inject_backdoor(self, data, labels):
        """
        Injector method.
        """
        # Picks data to poison based on poison ratio
        filter_mask = (torch.rand(len(data)).to(self.device) < self.poison_ratio)
        if filter_mask.sum() == 0:
            return data, labels
        data_to_poison = data[filter_mask]
        # Performs injection
        with torch.no_grad():
            # Shapes data for backdoor injection
            data_to_poison = self.map_to_256(unnormalizer(data_to_poison))
            mask_to_poison = self.generate_mask(len(data_to_poison))
            poison_trigger = self.generate_pattern(data_to_poison, mask_to_poison)
            data_to_poison = self.strength * mask_to_poison * poison_trigger + \
                (1 - self.strength) * mask_to_poison * data_to_poison + \
                (1 - mask_to_poison) * data_to_poison
        # Injects poison data back into original data
        data[filter_mask] = normalizer(self.map_to_og(data_to_poison))
        # Updates target labels
        labels[filter_mask] = self.target_class
        return data, labels, filter_mask
