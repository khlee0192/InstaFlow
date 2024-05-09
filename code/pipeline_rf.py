# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import copy
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import numpy as np

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from transformers import get_cosine_schedule_with_warmup

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class RectifiedFlowPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Rectified Flow and Euler discretization.
    This customized pipeline is based on StableDiffusionPipeline from the official Diffusers library (0.21.4)

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def decode_latents_tensor(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps = [(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Save original latents before generation
        original_latents = latents

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        dt = 1.0 / num_inference_steps

        # 7. Denoising loop of Euler discretization from t = 0 to t = 1
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 

            v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance 
            if do_classifier_free_guidance:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

            latents = latents + dt * v_pred 

            # # call the callback, if provided
            # if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()
            #     if callback is not None and i % callback_steps == 0:
            #         step_idx = i // getattr(self.scheduler, "order", 1)
            #         callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), latents, original_latents

class RectifiedInversableFlowPipeline(RectifiedFlowPipeline):
    @torch.no_grad()
    def generate_with_wm(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        wm_radius : int = 6,
        ):
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps = [(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Save original latents before generation
        original_latents = latents

        ### sub-process : adding WM


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        dt = 1.0 / num_inference_steps

        # 7. Denoising loop of Euler discretization from t = 0 to t = 1
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 

            v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance 
            if do_classifier_free_guidance:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

            latents = latents + dt * v_pred 

            # # call the callback, if provided
            # if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()
            #     if callback is not None and i % callback_steps == 0:
            #         step_idx = i // getattr(self.scheduler, "order", 1)
            #         callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        # latents : decoder space latent, original_latents : random noise, watermarked_latents : WM applied noise
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), latents, original_latents, watermarked_latents

    def exact_inversion(
            self,
            prompt: Union[str, List[str]] = None,
            latents: Optional[torch.FloatTensor] = None,
            image: Optional[torch.FloatTensor] = None,
            input_type: str = "latents",
            decoder_use_float: bool = False,
            test_beta: bool = False,
            num_inversion_steps: int = 50,
            num_inference_steps: int = 1,
            guidance_scale: float = 7.5,
            use_random_initial_noise: bool = False,
            verbose: bool = False,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            num_images_per_prompt: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            decoder_inv_steps: int = 100,
            decoder_lr: float = 0.1,
            forward_steps: int = 100,
            tuning_steps: int = 100,
            tuning_lr: float = 0.01,
            pnp_adjust: bool = False,
            reg_coeff: float = 0.0,
        ):
        """
        Exact inversion of RectifiedFlowPipeline. Gets input of 1,4,64,64 latents (which is denoised), and returns original latents by performing inversion
        
        Args:
            
        """
        
        do_classifier_free_guidance = guidance_scale > 1.0
        device = self._execution_device
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps, then reverse for inversion
        timesteps = [(1. - i/num_inversion_steps) * 1000. for i in range(num_inversion_steps)]
        #timesteps = reversed(timesteps)

        # 5. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        #extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        dt = 1.0 / num_inversion_steps

        if decoder_use_float:
            vae_float = copy.deepcopy(self.vae).float()

        torch.cuda.reset_max_memory_allocated()
        current_memory = torch.cuda.max_memory_allocated()
        t_s = time.time()
        if input_type == "answer":
            latents = latents
        elif input_type == "dec_inv":
            image = torch.Tensor(image).permute(0, 3, 1, 2).half()
            image = 2*image-1
            image = image.to('cuda')
            torch.set_grad_enabled(True)
            latents = self.edcorrector(image, vae_float, z_answer=latents, decoder_inv_steps=decoder_inv_steps, decoder_lr=decoder_lr, verbose=verbose)
            torch.set_grad_enabled(False)
        elif input_type == "new_alg":
            image = torch.Tensor(image).permute(0, 3, 1, 2)
            image = 2*image-1
            image = image.to('cuda')
            if decoder_use_float:
                image = image.float()
                if test_beta:
                    latents, extra_outputs, extra_outputs_another = self.dec_direct(image, latents, vae_float, test_beta=test_beta, adam=False, decoder_inv_steps=decoder_inv_steps, decoder_lr=decoder_lr, verbose=verbose, use_float=decoder_use_float)
                else:
                    latents = self.dec_direct(image, latents, vae_float, test_beta=test_beta, adam=True, decoder_inv_steps=decoder_inv_steps, decoder_lr=decoder_lr, verbose=verbose, use_float=decoder_use_float)
            else:
                image = image.half()
                if test_beta:
                    latents, extra_outputs, extra_outputs_another, another_output = self.dec_direct(image, latents, test_beta=test_beta, adam=False, decoder_inv_steps=decoder_inv_steps, decoder_lr=decoder_lr, verbose=verbose, use_float=decoder_use_float)
                else:
                    latents = self.dec_direct(image, latents, vae_float, test_beta=test_beta, adam=True, decoder_inv_steps=decoder_inv_steps, decoder_lr=decoder_lr, verbose=verbose, use_float=decoder_use_float)
        elif input_type == "encoder":
            # TODO : something wrong?
            image = torch.Tensor(image).permute(0, 3, 1, 2)
            image = 2*image-1
            image = image.to('cuda')
            latents = self.get_image_latents(image, sample=False)
        elif input_type == "ete":
            # something...
            image = torch.Tensor(image).permute(0, 3, 1, 2).half()
            image = 2*image-1
            image = image.to('cuda')
            latents = self.ete_inversion(image, latents, adam=False, decoder_inv_steps=decoder_inv_steps, decoder_lr=decoder_lr, 
                                                         do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale, 
                                                         prompt_embeds=prompt_embeds, verbose=verbose, use_float=False) # output_latents = z0, latents = zT
            # from end-to-end inversion, noise and latents may be calculated as whole
        else:
            pass

        peak_memory_usage = torch.cuda.max_memory_allocated()

        t_save = time.time() - t_s
        dec_inv_time = t_save

        current_latents = latents # Save latents, this is our target
        output_latents = latents # This is z0

        # 6. Inversion loop of Euler discretization from t = 1 to t = 0
        # Original concept does not work properly, so we choose to generate a random distribution to make v_pred at the beginning
        # with self.progress_bar(total=num_inversion_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if use_random_initial_noise:
                # Instead of using latents, perform inital guess (to make scale reliable)
                initial_latents = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([initial_latents] * 2) if do_classifier_free_guidance else initial_latents
                vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t
                v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample
                # perform guidance 
                if do_classifier_free_guidance:
                    v_pred_neg, v_pred_text = v_pred.chunk(2)
                    v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

                # instead of + in generation, switch to - since this is inversion process (not that meaningful since this is only process of setting initial value)
                latents = latents - dt * v_pred

                # Our work : perform forward step method

                t_forward = time.time()
                latents, output_loss_temp = self.forward_step_method(latents, current_latents, t, dt, prompt_embeds=prompt_embeds, do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale, steps=forward_steps, verbose=verbose, pnp_adjust=pnp_adjust)
                
                if tuning_steps != 0:
                    t_save += time.time() - t_forward
            else:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 

                v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample
                #v_pred = model(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance 
                if do_classifier_free_guidance:
                    v_pred_neg, v_pred_text = v_pred.chunk(2)
                    v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

                current_latents = latents
                latents = latents - dt * v_pred # instead of + in generation, switch to - since this is inversion process (not that meaningful since this is only process of setting initial value)
                #latents = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)

                # Our work : perform forward step method
                t_forward = time.time()
                latents, output_loss_temp = self.forward_step_method(latents, current_latents, t, dt, prompt_embeds=prompt_embeds, do_classifier_free_guidance=do_classifier_free_guidance, 
                                                    guidance_scale=guidance_scale, verbose=verbose,
                                                    steps=forward_steps, pnp_adjust=pnp_adjust, reg_coeff=reg_coeff)
                if tuning_steps != 0:
                    t_save += time.time() - t_forward

            # if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()
            #     if callback is not None and i % callback_steps == 0:
            #         step_idx = i // getattr(self.scheduler, "order", 1)
            #         callback(step_idx, t, latents)
        # The result "latents" is the reconstructed latents

        # Add another procedure, end-to-end correction of noise
        output_loss = output_loss_temp
        if tuning_steps != 0:
            torch.set_grad_enabled(True)
            t_s = time.time()
            if input_type == "encoder" or input_type == "dec_inv":
                latents, output_latents, output_loss = self.one_step_inversion_tuning_sampler(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    latents=latents,
                    image=image,
                    steps=tuning_steps,
                    lr=tuning_lr,
                )
            t_save = t_save + (time.time() - t_s)
            torch.set_grad_enabled(False)

        if output_loss == None:
            output_loss = output_loss_temp

        # Offload all models
        self.maybe_free_model_hooks()

        inv_time = t_save

        if verbose:
            print(f"inversion time : {inv_time}")

        # Creating image
        timesteps = [(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 

            v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance 
            if do_classifier_free_guidance:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

            recon_latents = latents + dt * v_pred

        image = self.vae.decode(recon_latents / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if test_beta:
            return image, output_latents, latents, output_loss, peak_memory_usage, dec_inv_time, extra_outputs, extra_outputs_another, another_output
        else:
            return image, output_latents, latents, output_loss, peak_memory_usage, dec_inv_time
    
    def one_step_inversion_tuning_sampler(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        steps: int = 100,
        lr: float = 0.01,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        verbose: bool = False
    ):
        r"""
        The simplified call function to the pipeline for tuning inversion. Creates a network form.
        Inputs:
            latents - initial noise obtained by inverse process
            image - target image to match
        
        Returns:
            latents - fine-tuned latents
        """

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        timesteps = [(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        dt = 1.0 / num_inference_steps
        t = timesteps[0]

        # Performing gradient descent, to tune the latents
        image_answer = image.clone()
        do_denormalize = [True] * image.shape[0]
        input = copy.deepcopy(latents)
        unet = copy.deepcopy(self.unet)
        vae = copy.deepcopy(self.vae)
        input.requires_grad_(True)

        loss_function = torch.nn.MSELoss(reduction='mean')
        output_loss = None

        optimizer = torch.optim.Adam([input], lr=lr)
        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=steps) # Not good

        for i in range(steps):
            latent_model_input = torch.cat([input] * 2) if do_classifier_free_guidance else input
            vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 
            v_pred = unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample
            v_pred = v_pred.detach()

            # perform guidance 
            if do_classifier_free_guidance:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

            temp = input + dt * v_pred
            # Stop here, check by below
            # visual = self.image_processor.postprocess(self.vae.decode(input/self.vae.config.scaling_factor, return_dict=False)[0].detach().cpu())
            
            image_recon = vae.decode(temp / self.vae.config.scaling_factor, return_dict=False)[0]

            image_recon = self.image_processor.postprocess(image_recon, output_type="pt", do_denormalize=do_denormalize)

            loss = loss_function(image_recon, image_answer)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()

            if i == steps-1:
                output_loss = loss.detach()

            if verbose:
                print(f"tuning, {i}, {loss.item()}")

        input.detach()
        temp.detach()

        # Offload all models
        self.maybe_free_model_hooks()

        return input, temp, output_loss
    
    @torch.inference_mode()
    def forward_step_method(
            self, 
            latents,
            current_latents, 
            t, dt, prompt_embeds,
            do_classifier_free_guidance,
            guidance_scale,
            verbose=True,
            warmup = False,
            warmup_time = 0,
            steps=100,
            original_step_size=0.1, step_size=0.05,
            factor=0.5, patience=15, th=1e-8,
            pnp_adjust=False,
            reg_coeff=1,
            ):
        """
        The forward step method assumes that current_latents are at right place(even on multistep), then map latents correctly to current latents
        The work is done by fixed-point iteration
        """
        regularizer = reg_coeff > 0

        latents_s = latents
        step_scheduler = StepScheduler(current_lr=step_size, factor=factor, patience=patience)

        # with self.progress_bar(total=steps) as progress_bar:
        for i in range(steps):
            if warmup:
                if i < warmup_time:
                    step_size = original_step_size * (i+1)/(warmup_time)

            latent_model_input = torch.cat([latents_s] * 2) if do_classifier_free_guidance else latents_s
            vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t
            v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

            if do_classifier_free_guidance:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)
            
            latents_t = latents_s + dt * v_pred

            diff = torch.nn.functional.mse_loss(latents_t, current_latents, reduction='mean').detach()

            if reg_coeff == 0:
                loss = torch.nn.functional.mse_loss(latents_t, current_latents, reduction='mean')
                
            else:
                state = 2
                if state == 1:
                    cos_sim = torch.nn.functional.cosine_similarity(latents_s.view(-1), latents_t.view(-1), dim=0, eps=1e-8)
                    mask = cos_sim >= 0
                    cos_sim = cos_sim[mask]
                    reg = torch.mean(cos_sim)
                elif state == 2:
                    cos_sim = torch.nn.functional.cosine_similarity(latents_s.view(-1), latents_t.view(-1), dim=0, eps=1e-8)
                    reg = torch.mean(cos_sim)
                elif state == 0:
                    reg = torch.norm(latents_t) ** 2

                loss = torch.nn.functional.mse_loss(latents_t, current_latents, reduction='mean') + reg_coeff * reg    

            latents_s = latents_s - step_size * (latents_t - current_latents)
            step_size = step_scheduler.step(loss)

            # progress_bar.update()
            if pnp_adjust:
                add_noise = randn_tensor(latents_s.shape, device=latents_s.device, dtype=latents_s.dtype)
                latents_s = latents_s + 0.01 * add_noise

            if verbose:
                if regularizer:
                    print(i, ((latents_t - current_latents).norm()/current_latents.norm()).item(), step_size, diff.item(), reg.item())
                else:
                    print(i, ((latents_t - current_latents).norm()/current_latents.norm()).item(), step_size, diff.item())

        return latents_s, loss.detach()

    def edcorrector(self, x, vae_float=None, z_answer=None, decoder_inv_steps=100, decoder_lr=0.1, verbose=False):
        """
        edcorrector calculates latents z of the image x by solving optimization problem ||E(x)-z||,
        not by directly encoding with VAE encoder. "Decoder inversion"

        INPUT
        x : image data (1, 3, 512, 512) -> given data
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z), working on adding regularizer
        """
        input = copy.deepcopy(x).float()
        z = self.get_image_latents(x, sample=False).clone().float() # initial z
        z.requires_grad_(True)

        loss_function = torch.nn.MSELoss(reduction='mean')

        ## Adjusting Adam
        optimizer = torch.optim.Adam([z], lr=decoder_lr)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=decoder_inv_steps)

        #for i in self.progress_bar(range(100)):
        for i in range(decoder_inv_steps): 
            latents = 1 / vae_float.config.scaling_factor * z
            image = vae_float.decode(latents, return_dict=False)[0]
            # x_pred = (image / 2 + 0.5)
            # x_pred = (image / 2 + 0.5).clamp(0, 1)
            # x_pred = self.decode_latents_for_gradient(z) # x_pred = D(z)

            # loss = loss_function(x_pred, input)
            loss = loss_function(image, input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if verbose:
                if z_answer is not None:
                    NMSE = (((z.detach()-z_answer).norm()/(z_answer).norm()).item())**2
                print(f"{i}, {loss.item()}, NMSE : {NMSE}")

        return z.half()

    @torch.inference_mode()
    def dec_direct(self, x, z_answer, vae_float=None, test_beta=False, adam=False, decoder_inv_steps=100, decoder_lr=0.1, verbose=False, use_float=False):
        # Must create two versions - float32, float16            
        if not test_beta:
            if not use_float:
                # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                z0 = self.get_image_latents(x, sample=False)
                z = z0.clone()
                if verbose:
                    print(f"start, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}")

                if adam: 
                    beta1, beta2 = 0.9 , 0.999
                    eps = 1e-4
                    m, v = 0, 0

                for i in range(decoder_inv_steps):
                    lr = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr)
                    Dz = 2*self.decode_latents_tensor(z)-1
                    EDz = self.get_image_latents(Dz, sample=False)
                    grad = EDz - z0
                    if adam:
                        m = beta1 * m + (1 - beta1) * grad
                        v = beta2 * v + (1 - beta2) * (grad**2)
                        m_corr = m / (1 - beta1**(i+1))
                        v_corr = v / (1 - beta2**(i+1))
                        z -= lr * m_corr / (torch.sqrt(v_corr) + eps)
                    else:
                        z = z - lr * grad
                    if verbose:
                        print(f"{i+1}, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr}")
            else:
                z_answer = z_answer.float()
                x = x.float()
                # set initial point
                encoding_dist = vae_float.encode(x).latent_dist
                z0 = encoding_dist.sample(generator=None)*0.18215            
                
                z = z0.clone()
                if verbose:
                    print(f"start, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}")

                if adam:
                    beta1, beta2 = 0.9 , 0.999
                    eps = 1e-4
                    m, v = 0, 0

                for i in range(decoder_inv_steps):
                    lr = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr)
                    # Decode to get Dz
                    Dz = (vae_float.decode(1/0.18215*z, return_dict=False)[0]/2 + 0.5).clamp(0, 1)
                    Dz = 2*Dz-1
                    # Encode to get EDz
                    EDz = vae_float.encode(Dz).latent_dist.sample(generator=None)*0.18215
                    grad = EDz - z0
                    if adam:
                        m = beta1 * m + (1 - beta1) * grad
                        v = beta2 * v + (1 - beta2) * (grad**2)
                        m_corr = m / (1 - beta1**(i+1))
                        v_corr = v / (1 - beta2**(i+1))
                        z -= lr * m_corr / (torch.sqrt(v_corr) + eps)
                    else:
                        z = z - lr * grad
                    if verbose:
                        print(f"{i+1}, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr}") # return shape must be [1, 4, 64, 64]       
            return z.half()

        else:
            if not use_float:
                # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                z0 = self.get_image_latents(x, sample=False)
                z = z0.clone()
                cocoercivity_rate_array = torch.zeros(decoder_inv_steps)

                # for comparing limit value
                z_list = []
                z_dist = torch.zeros(decoder_inv_steps)
                cocoercivity_rate_array_another = torch.zeros(decoder_inv_steps)
                if verbose:
                    print(f"start, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}")

                if adam: 
                    beta1, beta2 = 0.9 , 0.999
                    eps = 1e-4
                    m, v = 0, 0

                # using momentum
                momentum = 0.9
                for i in range(decoder_inv_steps):
                    # lr = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr)
                    lr = decoder_lr
                    Dz = 2*self.decode_latents_tensor(z)-1
                    EDz = self.get_image_latents(Dz, sample=False)
                    grad = EDz - z0
                    if adam:
                        m = beta1 * m + (1 - beta1) * grad
                        v = beta2 * v + (1 - beta2) * (grad**2)
                        m_corr = m / (1 - beta1**(i+1))
                        v_corr = v / (1 - beta2**(i+1))
                        z_new = z - lr * m_corr / (torch.sqrt(v_corr) + eps)
                    else:
                        if momentum > 0:
                            if i > 0:
                                grad_momentum = momentum * grad_momentum + grad
                            else:
                                grad_momentum = grad
                            grad = grad_momentum
                        z_new = z - lr * grad
                    if verbose:
                        print(f"{i+1}, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr}")

                    EDz_new = self.get_image_latents(2*self.decode_latents_tensor(z_new)-1, sample=False)
                    inner_product = torch.inner(torch.flatten(EDz_new.float() - EDz.float()), torch.flatten(z_new.float()-z.float()))
                    cocoercivity_rate = inner_product / torch.norm(EDz_new.float()-EDz.float())**2
                    # print(inner_product, cocoercivity_rate)

                    cocoercivity_rate_array[i] = cocoercivity_rate
                    z_list.append(z)
                    z = z_new

                    # if i > 0:
                    #     z_dist[i-1] = ((z_list[i-1]-z_list[i]).norm().item())
                
                # calculating on limit area
                z_comp = z
                z_temp = 2*self.decode_latents_tensor(z_comp)-1
                EDz_comp = self.get_image_latents(z_temp, sample=False)
                for i in range(len(z_list)):
                    z_temp = 2*self.decode_latents_tensor(z_list[i])-1
                    EDz_i = self.get_image_latents(z_temp, sample=False)
                    inner_product = torch.inner(torch.flatten(EDz_i.float() - EDz_comp.float()), torch.flatten(z_list[i].float()-z_comp.float()))
                    cocoercivity_rate = inner_product / torch.norm(EDz_i.float()-EDz_comp.float())**2
                    cocoercivity_rate_array_another[i] = cocoercivity_rate
                    print(cocoercivity_rate, z_list[i].norm().item(), EDz_i.norm().item())
                    # print(compare1, compare2)

                for i in range(len(z_list)-1):
                    z_dist[i] = ((z_list[i]-z_comp).norm().item())

                extra_outputs = cocoercivity_rate_array
                extra_outputs_another = cocoercivity_rate_array_another
                another_output = torch.Tensor(z_dist)

            # TODO : Work on float32 later
            else:
                z_answer = z_answer.float()
                x = x.float()
                # set initial point
                encoding_dist = vae_float.encode(x).latent_dist
                z0 = encoding_dist.sample(generator=None)*0.18215            
                beta_array = torch.zeros(decoder_inv_steps)

                z = z0.clone()
                if verbose:
                    print(f"start, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}")

                if adam:
                    beta1, beta2 = 0.9 , 0.999
                    eps = 1e-4
                    m, v = 0, 0

                for i in range(decoder_inv_steps):
                    lr = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr)
                    # Decode to get Dz
                    Dz = (vae_float.decode(1/0.18215*z, return_dict=False)[0]/2 + 0.5).clamp(0, 1)
                    Dz = 2*Dz-1
                    # Encode to get EDz
                    EDz = vae_float.encode(Dz).latent_dist.sample(generator=None)*0.18215
                    grad = EDz - z0
                    if adam:
                        m = beta1 * m + (1 - beta1) * grad
                        v = beta2 * v + (1 - beta2) * (grad**2)
                        m_corr = m / (1 - beta1**(i+1))
                        v_corr = v / (1 - beta2**(i+1))
                        z_new = z - lr * m_corr / (torch.sqrt(v_corr) + eps)
                    else:
                        z_new = z - lr * grad

                    z = z_new

                    if verbose:
                        print(f"{i+1}, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr}") # return shape must be [1, 4, 64, 64]
            
            return z.half(), extra_outputs, extra_outputs_another, another_output

    def ete_inversion(self, x, z_answer, adam=False, decoder_inv_steps=100, decoder_lr=0.01, do_classifier_free_guidance=True, guidance_scale=1.0, prompt_embeds=None, verbose=False, use_float=False):
        # Implement on adam later
        t = 1000
        dt = 1
        if not use_float:
            z0 = self.get_image_latents(x, sample=False)
            zT = self.backward_ot2(z0, do_classifier_free_guidance, guidance_scale, prompt_embeds, t, dt)
            z = z0.clone()
            if verbose:
                print(f"start, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}")

            if adam:
                beta1, beta2 = 0.9 , 0.999
                eps = 1e-4
                m, v = 0, 0

            for i in range(decoder_inv_steps):
                lr = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr)
                lr2 = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr/10)
                # lr2 = 0.0001 # best

                # 1. calculate grad1, update z0
                Dz = 2*self.decode_latents_tensor(z)-1
                EDz = self.get_image_latents(Dz, sample=False)
                grad = EDz - z0

                if adam:
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad**2)
                    m_corr = m / (1 - beta1**(i+1))
                    v_corr = v / (1 - beta2**(i+1))
                    z -= lr * m_corr / (torch.sqrt(v_corr) + eps)
                else:
                    z = z - lr * grad
                if verbose:
                    print(f"{i+1}-1, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr}")

                # 2. calculate grad2, update z0 again
                zT_new = self.backward_ot2(z, do_classifier_free_guidance, guidance_scale, prompt_embeds, t, dt)
                diff = self.forward_t2o(zT_new - zT, do_classifier_free_guidance, guidance_scale, prompt_embeds, t, dt)
                zT = zT_new
                if adam:
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad**2)
                    m_corr = m / (1 - beta1**(i+1))
                    v_corr = v / (1 - beta2**(i+1))
                    z -= lr * m_corr / (torch.sqrt(v_corr) + eps)
                else:
                    z = z - lr2 * diff
                if verbose:
                    print(f"\t{i+1}-2, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr2}")

            return z
        else:
            pass

    def forward_t2o(self, noise, do_classifier_free_guidance, guidance_scale, prompt_embeds, t, dt):
        latent_model_input = torch.cat([noise] * 2) if do_classifier_free_guidance else noise
        vec_t = torch.ones((latent_model_input.shape[0],), device=noise.device) * t
        v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

        if do_classifier_free_guidance:
            v_pred_neg, v_pred_text = v_pred.chunk(2)
            v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)
        
        latents = noise + dt * v_pred
        
        return latents
    
    def backward_ot2(self, latents, do_classifier_free_guidance, guidance_scale, prompt_embeds, t, dt):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t
        v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

        if do_classifier_free_guidance:
            v_pred_neg, v_pred_text = v_pred.chunk(2)
            v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)
        
        noise = latents - dt * v_pred
        
        return noise

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    def decode_latents_for_gradient(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        return image

    def prompt_optimization(
            self,
            prompt: Union[str, List[str]] = None,
            latents: Optional[torch.FloatTensor] = None,
            num_inversion_steps: int = 50,
            num_inference_steps: int = 1,
            guidance_scale: float = 7.5,
            use_random_initial_noise: bool = False,
            verbose: bool = False,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            num_images_per_prompt: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
        ):
        """
        Prompt optimization of RectifiedFlowPipeline. Gets input of 1,4,64,64 latents (which is denoised), and returns original latents and estimated prompt embeddings by joint optimization.
        To make it simple, prompt optimization gets prompt to compare the result, but does not use in calculation
        I don't know how to perform decode_prompt, so I will simply deal with error between original prompt and estimated prompt

        Args:
            
        """
        # To calculate inversion time
        t_s = time.time()

        do_classifier_free_guidance = guidance_scale > 1.0
        device = self._execution_device
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        answer_prompt_embeds, answer_negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # initialize prompt embeddings by entering null text
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            "",
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps, then reverse for inversion
        timesteps = [(1. - i/num_inversion_steps) * 1000. for i in range(num_inversion_steps)]

        # 5. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        #extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        dt = 1.0 / num_inversion_steps

        current_latents = latents # Save latents, this is our target

        # Target latent : "current_latents"
        # Target prompt : "answer_prompt_embeds"

        # 6. Inversion loop of Euler discretization from t = 1 to t = 0
        # Original concept does not work properly, so we choose to generate a random distribution to make v_pred at the beginning
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if use_random_initial_noise:
                    # Instead of using latents, perform inital guess (to make scale reliable)
                    initial_latents = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                    
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([initial_latents] * 2) if do_classifier_free_guidance else initial_latents
                    vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t
                    v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample
                    # perform guidance 
                    if do_classifier_free_guidance:
                        v_pred_neg, v_pred_text = v_pred.chunk(2)
                        v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

                    # instead of + in generation, switch to - since this is inversion process (not that meaningful since this is only process of setting initial value)
                    latents = latents - dt * v_pred

                    # Our work : perform forward step method
                    latents = self.forward_step_method(latents, current_latents, t, dt, prompt_embeds=prompt_embeds, do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale, verbose=verbose)
                else:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 

                    v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample
                    #v_pred = model(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

                    # perform guidance 
                    if do_classifier_free_guidance:
                        v_pred_neg, v_pred_text = v_pred.chunk(2)
                        v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

                    current_latents = latents
                    latents = latents - dt * v_pred # instead of + in generation, switch to - since this is inversion process (not that meaningful since this is only process of setting initial value)
                    #latents = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)

                    # Our work : perform forward step method
                    latents, prompt_embeds = self.prompt_optimization_gradient_method(
                        latents, 
                        current_latents, t, dt, 
                        prompt_embeds=prompt_embeds,
                        answer_prompt_embeds=answer_prompt_embeds,
                        do_classifier_free_guidance=do_classifier_free_guidance, 
                        guidance_scale=guidance_scale, verbose=verbose)
            

                if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Offload all models
        self.maybe_free_model_hooks()

        # After all process,
        ## Comparison of latents : current_latents, latents
        ## Comparison of prompt embeddings : answer_prompt_embeds, prompt_embeds

        inv_time = time.time() - t_s

        print(f"inversion time : {inv_time}")

        # Creating image
        timesteps = [(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 

            v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance 
            if do_classifier_free_guidance:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

            recon_latents = latents + dt * v_pred

        image = self.vae.decode(recon_latents / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        return latents, image

    def prompt_optimization_method(
        self, 
        latents,
        current_latents, 
        t, dt, prompt_embeds,
        answer_prompt_embeds,
        do_classifier_free_guidance,
        guidance_scale,
        verbose=True,
        warmup = False,
        warmup_time = 0,
        original_step_size=0.1, step_size=0.1,
        factor=0.5, patience=20, th=1e-5
    ):
        """
        Our goal is to get latents, prompt_embeds correctly that can guess current_latents
        """

        latents_s = latents
        prompt_embeds_s = prompt_embeds
        step_scheduler = StepScheduler(current_lr=step_size, factor=factor, patience=patience)
        steps = 1000

        for i in range(steps):
            if warmup:
                if i < warmup_time:
                    step_size = original_step_size * (i+1)/(warmup_time)

            latent_model_input = torch.cat([latents_s] * 2) if do_classifier_free_guidance else latents_s
            vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t
            v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

            if do_classifier_free_guidance:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)
            
            latents_t = latents_s + dt * v_pred

            loss = torch.nn.functional.mse_loss(latents_t, current_latents, reduction='mean')
            #loss = torch.nn.functional.mse_loss(prompt_embeds_s, answer_prompt_embeds, reduction='mean')
            if loss.item() < th:
                break

            latents_s = latents_s - step_size * (latents_t - current_latents)
            prompt_embeds_s = prompt_embeds_s - step_size * (prompt_embeds_s - answer_prompt_embeds)
            step_size = step_scheduler.step(loss)

            if verbose:
                print(i, round((latents_t - current_latents).norm().item()/current_latents.norm().item(), 5), end = " ")
                print(round((prompt_embeds_s - answer_prompt_embeds).norm().item()/(answer_prompt_embeds).norm().item(), 5))

        return latents_s, prompt_embeds_s

    def prompt_optimization_gradient_method(
        self, 
        latents,
        current_latents, 
        t, dt, prompt_embeds,
        answer_prompt_embeds,
        do_classifier_free_guidance,
        guidance_scale,
        verbose=True,
        warmup = False,
        warmup_time = 0,
        original_step_size=0.1, step_size=10,
        factor=0.1, patience=30, th=1e-5
    ):
        """
        Our goal is to get latents, prompt_embeds correctly that can guess current_latents
        """

        latents_s = latents.clone().requires_grad_(True)
        prompt_embeds_s = prompt_embeds.clone().requires_grad_(True)
        step_scheduler = StepScheduler(current_lr=step_size, factor=factor, patience=patience)
        steps = 10000

        for i in range(steps):
            if warmup:
                if i < warmup_time:
                    step_size = original_step_size * (i + 1) / (warmup_time)

            latent_model_input = torch.cat([latents_s] * 2) if do_classifier_free_guidance else latents_s
            vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t
            v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds_s).sample

            if do_classifier_free_guidance:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)

            latents_t = latents_s + dt * v_pred

            loss_latents = torch.nn.functional.mse_loss(latents_t, current_latents, reduction='mean')
            loss_prompt_embeds = torch.nn.functional.mse_loss(prompt_embeds_s, answer_prompt_embeds, reduction='mean')
            loss = loss_latents + 100* loss_prompt_embeds

            if loss.item() < th:
                break

            loss.backward()

            with torch.no_grad():
                latents_s -= step_size * latents_s.grad
                prompt_embeds_s -= step_size * prompt_embeds_s.grad

                # Manually zero the gradients after updating the parameters
                latents_s.grad.zero_()
                prompt_embeds_s.grad.zero_()

            step_size = step_scheduler.step(loss_latents.item())

            if verbose:
                print(i, round((latents_t - current_latents).norm().item() / current_latents.norm().item(), 5), end=" ")
                print(round((prompt_embeds_s - answer_prompt_embeds).norm().item() / answer_prompt_embeds.norm().item(), 5), end=" ")
                print(step_size)

        return latents_s.detach(), prompt_embeds_s.detach()

def get_lr_cosine_with_warmup(i, num_steps=100, num_warmup_steps=10, lr_max=0.01):
    assert i>=0 and i<num_steps
    if i<num_warmup_steps:
        lr = (i+1)/num_warmup_steps * lr_max
    else:
        lr = lr_max * (1 + math.cos(math.pi * (i-num_warmup_steps)/ (num_steps - num_warmup_steps)))/2
    return lr

class StepScheduler(ReduceLROnPlateau):
    def __init__(self, mode='min', current_lr=0, factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        if current_lr == 0:
            raise ValueError('Step size cannot be 0')

        self.min_lr = min_lr
        self.current_lr = current_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            import warnings
            warnings.warn("EPOCH_DEPRECATION_WARNING", UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.current_lr

    def _reduce_lr(self, epoch):
        old_lr = self.current_lr
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.current_lr = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                            "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                        ' to {:.4e}.'.format(epoch_str,new_lr))