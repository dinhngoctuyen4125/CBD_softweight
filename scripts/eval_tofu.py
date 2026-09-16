import hydra
from hydra.core.hydra_config import HydraConfig
from transformers import AutoTokenizer
import os
from pathlib import Path

from uld.utils import init_script
from uld.data.conv_util import create_template
from uld.tofuutil import tofu_eval
from uld.model import EVAL_INIT_FUNCS
from codetiming import Timer

@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(configs):
    LOGGER = init_script(configs)
    LOGGER.info("Configs", configs=configs)
    OUTPUTDIR = HydraConfig.get().runtime.output_dir
    device = os.getenv("EVAL_DEVICE", "cuda:0")
    print("DEVICE", device)

    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    local_tofu = Path(os.getenv("TOFU_BASE_MODEL", ""))
    if local_files_only and local_tofu.is_dir():
        # Keep behavior unchanged unless the config points to the HF id.
        try:
            if str(getattr(configs.model, "model_path", "")) == "locuslab/tofu_ft_llama2-7b":
                configs.model.model_path = str(local_tofu)
            if str(getattr(configs.model, "tokenizer_path", "")) == "locuslab/tofu_ft_llama2-7b":
                configs.model.tokenizer_path = str(local_tofu)
        except Exception:
            pass

    conv_template = create_template(configs.data.conv_template)
    model_mode = configs.get('model_mode', None)
    init_func = EVAL_INIT_FUNCS.get(model_mode.get('mode', 'base'))
    model = init_func(
        base_model_config=configs.model,
        model_mode_config=configs.model_mode,
        ckpt_path=configs.ckpt_path,
        device=device,
    )

    tok_name = getattr(configs.model, "tokenizer_path", None) or getattr(configs.model, "model_path", None) or "locuslab/tofu_ft_llama2-7b"
    tokenizer = AutoTokenizer.from_pretrained(tok_name, local_files_only=local_files_only)
    tokenizer.padding_side = "left"
    tokenizer.padding_size = 'longest'
    tokenizer.pad_token = tokenizer.eos_token

    right_pad_tokenizer = AutoTokenizer.from_pretrained(tok_name, local_files_only=local_files_only)
    right_pad_tokenizer.padding_side = 'right'
    right_pad_tokenizer.padding_size = 'longest'
    right_pad_tokenizer.pad_token = tokenizer.eos_token

    with Timer("Evaluation", text="{name} spent: {:0.4f} seconds"):
        tofu_eval(OUTPUTDIR, LOGGER, configs.data, model, tokenizer, right_pad_tokenizer, conv_template, only_forget_quality=False)

if __name__ == "__main__":
    main()
