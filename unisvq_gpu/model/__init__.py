from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from .modeling_qwen3_vq import Qwen3LCQATConfig, Qwen3LCQATForCausalLM, Qwen3LCQATCompressionConfig, Qwen3LCQATForCompression

AutoConfig.register('qwen3_lcqat', Qwen3LCQATConfig)
AutoModel.register(Qwen3LCQATConfig, Qwen3LCQATForCausalLM)
AutoModelForCausalLM.register(Qwen3LCQATConfig, Qwen3LCQATForCausalLM)

AutoConfig.register('qwen3_lcqat_compression', Qwen3LCQATCompressionConfig)
AutoModel.register(Qwen3LCQATCompressionConfig, Qwen3LCQATForCompression)
AutoModelForCausalLM.register(Qwen3LCQATCompressionConfig, Qwen3LCQATForCompression)
