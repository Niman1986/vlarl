"""
openvla.py

This module defines the OpenVLA model for VLLM inference.
Ref: https://github.com/vllm-project/vllm/blob/v0.7.2/vllm/model_executor/models/llava.py
"""

from typing import (Iterable, List, Literal, Mapping, Optional, Protocol, Set,
                    Tuple, TypedDict, Union, Any, Callable, cast, Dict, ClassVar)
from abc import abstractmethod
from functools import cached_property, lru_cache
from packaging.version import Version

import torch
import torch.nn as nn
from PIL import Image
from transformers import (BatchFeature, ProcessorMixin, PretrainedConfig, AutoConfig, PreTrainedTokenizerBase)
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from transformers.models.auto import CONFIG_MAPPING

# from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, PrismaticConfig, VISION_BACKBONE_TO_TIMM_ID
# from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from vllm.model_executor.layers.sampler import get_sampler
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                   MultiModalInputs, MultiModalKwargs,
                                   NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                  ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                       BaseProcessingInfo, ProcessingCache,
                                       PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.siglip import (
    get_max_siglip_image_tokens,
)
from functools import partial
from timm.models.vision_transformer import LayerScale


# === Data Structures for Image Inputs ===
class OpenVLAImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape: (batch_size * num_images, num_channels, height, width)
    """

class OpenVLAImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: (batch_size * num_images, image_feature_size, hidden_size)
    """

OpenVLAImageInputs = Union[OpenVLAImagePixelInputs, OpenVLAImageEmbeddingInputs]

# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return wrapper

# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor

def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Utilities for Mapping Prismatic names to HF names ===
# fmt: off
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    "clip-vit-l": [224], "siglip-vit-so400m": [224], "dinov2-vit-l": [224], "in1k-vit-l": [224],

    "clip-vit-l-336px": [336],
    "siglip-vit-so400m-384px": [384],

    "dinoclip-vit-l-336px": [336, 336],
    "dinosiglip-vit-so-224px": [224, 224],
    "dinosiglip-vit-so-384px": [384, 384],
}
VISION_BACKBONE_TO_TIMM_ID: Dict[str, List[str]] = {
    "clip-vit-l": ["vit_large_patch14_clip_224.openai"],
    "clip-vit-l-336px": ["vit_large_patch14_clip_336.openai"],

    "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
    "in1k-vit-l": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],

    "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
    "siglip-vit-so400m-384px": ["vit_so400m_patch14_siglip_384"],

    "dinoclip-vit-l-336px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_clip_336.openai"],
    "dinosiglip-vit-so-224px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
    "dinosiglip-vit-so-384px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_384"],
}
TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "clip-vit-l": ["quick_gelu"], "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None], "in1k-vit-l": [None],
    "siglip-vit-so400m": [None], "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None], "dinosiglip-vit-so-384px": [None, None]
}

LLM_BACKBONE_TO_HF_PATH = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf", "llama2-13b-pure": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",

    "vicuna-v15-7b": "lmsys/vicuna-7b-v1.5", "vicuna-v15-13b": "lmsys/vicuna-13b-v1.5",

    "mistral-v0.1-7b-pure": "mistralai/Mistral-7B-v0.1",
    "mistral-v0.1-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",

    "phi-2-3b": "microsoft/phi-2",

    "qwen25-0_5b-extra": "Qwen/Qwen2.5-0.5B",
}
LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama", "llama2-13b-pure": "llama", "llama2-7b-chat": "llama", "llama2-13b-chat": "llama",
    "vicuna-v15-7b": "llama", "vicuna-v15-13b": "llama",

    "mistral-v0.1-7b-pure": "mistral", "mistral-v0.1-7b-instruct": "mistral",

    "phi-2-3b": "phi",

    # "qwen25-0_5b-extra": "qwen2.5",
}

VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)
# fmt: on

class PrismaticConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "letterbox",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(f"Vision backbone `{vision_backbone_id}` not in {VALID_VISION_BACKBONES = }")

        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"LLM backbone `{llm_backbone_id}` not in {VALID_LLM_BACKBONES = }")

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(self.vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
        )

        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[self.vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[self.vision_backbone_id]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[self.llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        try:
            self.text_config = (
                CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](**text_config)
                if text_config is not None
                else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
            )
        except:
            # hf_hub_path = QWEN25_MODELS[self.llm_backbone_id]["hf_hub_path"]
            hf_token = kwargs.get("token", None)
            self.text_config = AutoConfig.from_pretrained(self.hf_llm_id, token=hf_token)

        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAConfig(PrismaticConfig):
    model_type: str = "openvla"

    def __init__(
        self,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        super().__init__(**kwargs)


# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


# === MultiModal Projector for OpenVLA ===
class OpenVLAMultiModalProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        use_fused_vision_backbone: bool,
        projector_hidden_act: str = "gelu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        
        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = ColumnParallelLinear(
                vision_hidden_size,
                text_hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1" if prefix else "fc1",
            )
            self.fc2 = RowParallelLinear(
                text_hidden_size,
                text_hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2" if prefix else "fc2",
            )
            self.act_fn1 = get_act_fn(projector_hidden_act)
        else:
            initial_projection_dim = 4 * vision_hidden_size
            self.fc1 = ColumnParallelLinear(
                vision_hidden_size,
                initial_projection_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1" if prefix else "fc1",
            )
            self.fc2 = ColumnParallelLinear(
                initial_projection_dim,
                text_hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2" if prefix else "fc2",
            )
            self.fc3 = RowParallelLinear(
                text_hidden_size,
                text_hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc3" if prefix else "fc3",
            )
            self.act_fn1 = get_act_fn(projector_hidden_act)
            self.act_fn2 = get_act_fn(projector_hidden_act)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            hidden_states, _ = self.fc1(image_features)
            hidden_states = self.act_fn1(hidden_states)
            hidden_states, _ = self.fc2(hidden_states)
        else:
            hidden_states, _ = self.fc1(image_features)
            hidden_states = self.act_fn1(hidden_states)
            hidden_states, _ = self.fc2(hidden_states)
            hidden_states = self.act_fn2(hidden_states)
            hidden_states, _ = self.fc3(hidden_states)
        return hidden_states


# === Fused Vision Tower for Dinosiglip ===
class FusedVisionTower(nn.Module):
    def __init__(self, hf_config: OpenVLAConfig, num_hidden_layers_override: Optional[int] = None):
        super().__init__()
        import timm
        self.use_fused = True
        # Primary featurizer
        self.featurizer = timm.create_model(
            hf_config.timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=hf_config.image_sizes[0],
            act_layer=hf_config.timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks)-2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # Fused featurizer
        self.fused_featurizer = timm.create_model(
            hf_config.timm_model_ids[1],
            pretrained=False,
            num_classes=0,
            img_size=hf_config.image_sizes[1],
            act_layer=hf_config.timm_override_act_layers[1],
        )
        self.fused_featurizer.forward = unpack_tuple(
            partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks)-2})
        )
        self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale modules to be HF-compatible
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        for module in self.fused_featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Expecting pixel_values shape: (batch_size, 6, H, W) where channels are stacked (3 for primary, 3 for fused)
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches = self.featurizer(img)
        patches_fused = self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)


def _init_vision_tower(hf_config: OpenVLAConfig, quant_config: Optional[QuantizationConfig] = None):
    backbone_id = hf_config.vision_backbone_id
    vision_feature_layer = getattr(hf_config, "vision_feature_layer", None)
    if vision_feature_layer is not None:
        if vision_feature_layer < 0:
            num_hidden_layers = hf_config.num_hidden_layers + vision_feature_layer + 1
        else:
            num_hidden_layers = vision_feature_layer + 1
    else:
        num_hidden_layers = None

    if hf_config.use_fused_vision_backbone:
        if backbone_id.startswith("dinosiglip"):
            return FusedVisionTower(hf_config, num_hidden_layers_override=num_hidden_layers)
        else:
            raise NotImplementedError(f"Fused vision backbone not supported for {backbone_id}; only dinosiglip is supported.")
    else:
        raise NotImplementedError(f"Unsupported vision backbone: {backbone_id}; only dinosiglip is supported.")

class OpenVLAPrismaticProcessor(PrismaticProcessor):
    """
    HACK: support text-only input
    Ref: https://github.com/huggingface/transformers/blob/6b550462139655d488d4c663086a63e98713c6b9/src/transformers/models/llava/processing_llava.py#L144
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        # print(f"text: {text}")
        # print(f"images: {images}")
        if images is None:
            image_inputs = {}
        else:
            image_inputs = {"pixel_values": self.image_processor(images, return_tensors=return_tensors)["pixel_values"]}
        # left padding here
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length, padding_side="left"
        )
        # text_inputs.input_ids, text_inputs.attention_mask = self._add_special_token(text_inputs.input_ids, text_inputs.attention_mask)
        return BatchFeature(data={**text_inputs, **image_inputs})

    # def _add_special_token(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    #     ## If the special empty token ('') does not already appear after the colon (':') token in the prompt
    #     ## (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time

    #     print(f"pre input_ids: {input_ids}")

    #     if not torch.all(input_ids[:, -1] == 29871):    # True TODO: == add_special_tokens in base_tokenizer ?
    #         # Create a tensor of the special token with the same batch size
    #         special_token = torch.full((input_ids.size(0), 1), 29871, dtype=torch.long).to(input_ids.device)
    #         # Concatenate the special token to each sequence in the batch
    #         input_ids = torch.cat((input_ids, special_token), dim=1)
    #         attention_mask = torch.cat((attention_mask, torch.ones_like(special_token)), dim=1)

    #     print(f"post input_ids: {input_ids}")
        
    #     return input_ids, attention_mask

def get_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    processor_cls: type[ProcessorMixin] = ProcessorMixin,
    **kwargs: Any,
):
    """Load a processor for the given model name via HuggingFace."""
    processor = processor_cls.from_pretrained(
        processor_name,
        *args,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    return cast(ProcessorMixin, processor)

cached_get_processor = lru_cache(get_processor)

class OpenVLAProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        # return self.ctx.get_hf_config(OpenVLAConfig)
        return self.ctx.model_config.hf_config

    def get_hf_processor(self) -> ProcessorMixin:
        # processor = self.ctx.get_hf_processor(ProcessorMixin)
        processor = cached_get_processor(self.ctx.model_config.tokenizer, trust_remote_code=True, processor_cls=OpenVLAPrismaticProcessor)
        # HACK: the processor is not a subclass of PrismaticProcessor, so we need to avoid the type check
        # ref: https://github.com/vllm-project/vllm/blob/256a2d29dc2358d7c0a5d38c0faf152095335929/vllm/transformers_utils/processor.py#L9
        assert isinstance(processor, ProcessorMixin)
        return processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_max_image_tokens(self) -> int:
        hf_config = self.ctx.model_config.hf_config
        backbone_id = hf_config.vision_backbone_id
        if backbone_id.startswith("dinosiglip"):
            timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[backbone_id]    # e.g., ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"]
            hf_config.image_size = hf_config.image_sizes[0]
            hf_config.patch_size = int(timm_model_ids[0].split("patch")[1].split("_")[0])   # HACK: get patch_size from timm_model_ids
            num_image_tokens = get_max_siglip_image_tokens(hf_config)
        else:
            raise NotImplementedError(f"Unsupported vision backbone: {backbone_id}; only dinosiglip is supported.")
        return num_image_tokens


# === MultiModal Processor for OpenVLA ===
class OpenVLAMultiModalProcessor(BaseMultiModalProcessor):
    """
    A multi-modal processor for OpenVLA.
    This class handles the processing of image inputs for OpenVLA models.
    """
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        """Replacement for image tokens"""
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.pad_token_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))
            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                num_image_tokens = self.info.get_max_image_tokens()
            return [image_token_id] * num_image_tokens
        
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]


class OpenVLADummyInputsBuilder(BaseDummyInputsBuilder[OpenVLAProcessingInfo]):
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        """Dummy image and text inputs"""
        hf_config = self.info.get_hf_config()
        num_images = mm_counts.get("image", 0)
        image_token = "<PAD>"
        mm_data = {"image": self._get_dummy_images(width=hf_config.image_sizes[0], height=hf_config.image_sizes[0], num_images=num_images)}
        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )

# === Main Model Class ===
@MULTIMODAL_REGISTRY.register_processor(OpenVLAMultiModalProcessor,
                                        info=OpenVLAProcessingInfo,
                                        dummy_inputs=OpenVLADummyInputsBuilder,
                                        )
class OpenVLAForActionPrediction(nn.Module, SupportsMultiModal, SupportsPP):
    """
    OpenVLA model for VLLM inference.
    """
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # Extract configuration from vllm_config
        config = vllm_config.model_config.hf_config  # type: OpenVLAConfig
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Instantiate vision tower (only dinosiglip is supported)
        self.vision_backbone = _init_vision_tower(config, quant_config=quant_config)

        # Use the hidden size from the vision tower for projection.
        self.projector = OpenVLAMultiModalProjector(
            vision_hidden_size=self.vision_backbone.embed_dim,
            text_hidden_size=config.text_config.hidden_size,
            use_fused_vision_backbone=config.use_fused_vision_backbone,
            projector_hidden_act="gelu",
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "projector"),
        )

        # Initialize the language model with vllm_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["LlamaForCausalLM"], # HACK: openvla do not have architectures key in text_config
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler
        return get_sampler()

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.image_sizes[0]
        expected_dims = (3, h, w) if not self.config.use_fused_vision_backbone else (6, h, w)
        actual_dims = tuple(data.shape[1:])
        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. You supplied {tuple(data.shape)}."
            )
        return data

    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[dict]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of pixel values. Got type: {type(pixel_values)}")
            return OpenVLAImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(flatten_bn(pixel_values, concat=True))
            )
        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of image embeddings. Got type: {type(image_embeds)}")
            return OpenVLAImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True)
            )
        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(self, vision_tower: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(torch.bfloat16)  # HACK: openvla supports bfloat16 originally
        image_features = vision_tower(pixel_values)
        return image_features

    def _process_image_pixels(self, inputs: OpenVLAImagePixelInputs) -> torch.Tensor:
        assert self.vision_backbone is not None
        pixel_values = inputs["data"]
        return self._image_pixels_to_features(self.vision_backbone, pixel_values)

    def _process_image_input(self, image_input: OpenVLAImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]
        assert self.vision_backbone is not None
        image_features = self._process_image_pixels(image_input)
        return self.projector(image_features)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        return self._process_image_input(image_input)

    def get_input_embeddings(
        self, input_ids: torch.Tensor, multimodal_embeddings: Optional[NestedTensors] = None
    ) -> torch.Tensor:
        # print(f"input_ids: {input_ids}")
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.config.pad_token_id
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for OpenVLA."""

        # print(input_ids.shape)
        # print(input_ids)

        if intermediate_tensors is not None:
            inputs_embeds = None
        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids,
            positions, 
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
