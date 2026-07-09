from swift.utils import Processor
from swift.template import TemplateType
from swift.model import Model, ModelGroup, ModelMeta, ModelLoader, register_model
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel

from model.modeling_qwen3_vq import Qwen3LCQATForCausalLM

class MyModelLoader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        return AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor, model_kwargs) -> PreTrainedModel:
        return Qwen3LCQATForCausalLM.from_pretrained(
            model_dir, config=config, torch_dtype=self.torch_dtype, trust_remote_code=True, **model_kwargs
        )

register_model(
    ModelMeta(
        model_type='qwen3_lcqat',
        model_groups=[
            ModelGroup([Model('qwen3_lcqat', None)])
        ],
        template=TemplateType.qwen3,
        is_multimodal=False,
        architectures=["Qwen3ForCausalLM"]
    ))
