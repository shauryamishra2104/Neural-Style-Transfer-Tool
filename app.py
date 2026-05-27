import torch
torch.set_num_threads(1)

import gradio as gr
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# Device
device = torch.device("cpu")

# Load Models
encoder = VGGEncoder("vgg_normalised.pth").to(device)
decoder = Decoder().to(device)

state_dict = torch.load(
    "experiment/final_exp/decoder_final.pth",
    map_location=device,
    weights_only=True
)

new_state_dict = {}

for k, v in state_dict.items():
    new_key = k.replace("net", "decoder")
    new_state_dict[new_key] = v

decoder.load_state_dict(new_state_dict)

encoder.eval()
decoder.eval()

encoder = encoder.float()
decoder = decoder.float()


def style_transfer(content_image, style_image, alpha):

    content_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():

        content_feats = encoder(content_tensor, True)
        style_feats = encoder(style_tensor, True)

        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        stylized_feats = (
            alpha * stylized_feats
            + (1 - alpha) * content_feats
        )

        stylized_image = decoder(stylized_feats)

        output = stylized_image.squeeze().cpu().clamp(0, 1)

        output_image = transforms.ToPILImage()(output)

        del content_feats
        del style_feats
        del stylized_feats
        del stylized_image

    return output_image


title = "Neural Style Transfer"
description = "Upload a content image and a style image to generate stylized artwork."


interface = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Content Image"),
        gr.Image(type="pil", label="Style Image"),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=1.0,
            step=0.1,
            label="Style Strength"
        )
    ],
    outputs=gr.Image(type="pil", label="Stylized Output"),
    title=title,
    description=description,
    allow_flagging="never"
)

interface.launch(server_name="0.0.0.0", server_port=7860)