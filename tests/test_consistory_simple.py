try:
    from ..examples.community.stable_diffusion_consistory_pipeline import StableDiffusionConsistoryPipeline
except ImportError:
    import sys; sys.path.append('/home/jeeves/JJ_Projects/github/diffusers/examples')
    from community.stable_diffusion_consistory_pipeline import StableDiffusionConsistoryPipeline
    from community.consistory_models.unet import SDUNet2DConditionModel

from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

def test_consistory_pipeline():
    model_name_or_path = "/mnt/data/user/tc_ai/user/songjunjie/kakarot25DCozy_cozy"
    noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder="vae")
    # vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name_or_path, subfolder="text_encoder")
    # unet1 = UNet2DConditionModel.from_pretrained(model_name_or_path, subfolder="unet")
    unet = SDUNet2DConditionModel.from_pretrained_vanilla(model_name_or_path, subfolder="unet")
    
    pipe = StableDiffusionConsistoryPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,        
        safety_checker=None,
        feature_extractor=None
    )
    scene_prompts = [
        'on the beach', 'in the forest', "in a room"
    ]
    object_prompt = "1girl"
    prompts = [object_prompt + ", " + sp for sp in scene_prompts]
    keywords = ['girl', 'black_T-shirt']
    images = pipe(prompts, keywords=keywords).images
    for i, img in enumerate(images):
        img.save(f"image_{i}.png")

if __name__ == '__main__':
    test_consistory_pipeline()
