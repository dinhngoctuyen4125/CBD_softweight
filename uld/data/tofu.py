import copy
import json
import os
import torch
import datasets
from datasets import load_dataset

from .conv_util import create_template
from .datamodule import TrainDataModule, TorchDataset


class ToFU_DataModule(TrainDataModule):

    def __init__(
        self, 
        split, 
        tokenizer,
        conv_template_config, 
        max_len=256, 
        batch_size=8, 
        with_retain=False, 
        retain_num=400, 
        retain_match_forget=True,
        with_dpo=False, 
        expand_forget=False, 
        with_perturb=False, # Our method
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.dpo_mode = with_dpo
        self.conv_template = create_template(conv_template_config, tokenizer=tokenizer)

        def flatten_perturb(perturb_dataset):
            for sample in perturb_dataset:
                perturb_answer_list = sample.pop('perturbed_answer')
                newsample = copy.deepcopy(sample)
                for perturb_ans in perturb_answer_list[:1]:
                    newsample['answer'] = perturb_ans
                    yield newsample

        def load_tofu_split(dataset_name, split_name):
            local_candidates = []
            for candidate in (dataset_name, os.environ.get("TOFU_DATA_NAME"), "TOFU"):
                if candidate and candidate not in local_candidates:
                    local_candidates.append(candidate)

            for local_root in local_candidates:
                local_file = os.path.join(local_root, f"{split_name}.json")
                if os.path.exists(local_file):
                    print(f"[TOFU local] loading {local_file}")
                    with open(local_file, "r", encoding="utf-8") as f:
                        first_non_ws = ""
                        while True:
                            ch = f.read(1)
                            if not ch:
                                break
                            if not ch.isspace():
                                first_non_ws = ch
                                break
                        f.seek(0)
                        if first_non_ws == "[":
                            return datasets.Dataset.from_list(json.load(f))
                        rows = [json.loads(line) for line in f if line.strip()]
                        return datasets.Dataset.from_list(rows)
            return load_dataset("locuslab/TOFU", split_name)["train"]

        forget_eval = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), split)
        # Only remove columns that exist in the dataset
        cols_to_remove = [c for c in ['paraphrased_answer', 'paraphrased_question', 'perturbed_answer'] if c in forget_eval.column_names]
        if cols_to_remove:
            forget_eval = forget_eval.remove_columns(cols_to_remove)
        self.forget_eval = forget_eval

        retain_eval = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), 'retain_perturbed')
        cols_to_remove = [c for c in ['paraphrased_answer', 'paraphrased_question', 'perturbed_answer'] if c in retain_eval.column_names]
        if cols_to_remove:
            retain_eval = retain_eval.remove_columns(cols_to_remove)
        self.retain_eval = retain_eval

        perturb_eval = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), split)
        if 'perturbed_answer' not in getattr(perturb_eval, "column_names", []):
            fallback_split = f"{split}_perturbed"
            try:
                perturb_eval = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), fallback_split)
            except Exception:
                pass
        perturb_eval = datasets.Dataset.from_generator(flatten_perturb, gen_kwargs={"perturb_dataset": perturb_eval})
        self.perturb_eval = perturb_eval

        paraphrase_eval = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), split)
        if not (
            'paraphrased_answer' in getattr(paraphrase_eval, "column_names", [])
            and 'paraphrased_question' in getattr(paraphrase_eval, "column_names", [])
        ):
            fallback_split = f"{split}_perturbed"
            try:
                paraphrase_eval = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), fallback_split)
            except Exception:
                pass
        cols_to_remove = [c for c in ['answer', 'perturbed_answer', 'paraphrased_question'] if c in paraphrase_eval.column_names]
        if cols_to_remove:
            paraphrase_eval = paraphrase_eval.remove_columns(cols_to_remove)
        if 'paraphrased_answer' in paraphrase_eval.column_names:
            paraphrase_eval = paraphrase_eval.rename_column('paraphrased_answer', 'answer')
        self.paraphrase_eval = paraphrase_eval

        # Construct training 
        base_forget_data = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), split)
        base_retain_data = datasets.Dataset.from_dict({'question': [], 'answer': []})
        self.forget_length = len(base_forget_data)
        self.retain_length = 0
        if with_retain:
            print("Adding retain data")
            retain_split = "retain" + str(100 - int(split.split("_")[0].replace("forget", ""))).zfill(2)
            retain_train = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), retain_split)
            #! Optional: match retain count to forget count (default True for safety).
            if retain_match_forget:
                retain_num = min(retain_num, len(base_forget_data))
            retain_train = retain_train.select(
                range(len(retain_train) - retain_num, len(retain_train))
            )
            self.retain_length += len(retain_train)
            base_retain_data = datasets.concatenate_datasets([base_retain_data, retain_train])

        #! Augment forget data
        if expand_forget:
            print("Adding forget data")
            expand_qanum = kwargs.get('expand_qanum', 2)
            if expand_qanum > 0:
                expand_qa = collect_expand_data(
                    expand_qanum=expand_qanum, path=kwargs.get('paraphrase_path'),
                )
                tmpdata = datasets.Dataset.from_list([{'question': q, 'answer': a} for q, a in expand_qa])
            else:
                #! Otherwise we copy the original forget data
                tmpdata = load_tofu_split(kwargs.get("name", "locuslab/TOFU"), split)
            base_forget_data = datasets.concatenate_datasets([base_forget_data, tmpdata])
            self.forget_length += len(tmpdata)
            
        if with_perturb:
            print("Adding perturb data")
            perturb_qa = collect_perturb_data( 
                expand_qanum=kwargs.get('expand_qanum', 3),
                path=kwargs.get('perturb_path')
            )
            tmpdata = datasets.Dataset.from_list([{'question': q, 'answer': a} for q, a in perturb_qa])
            self.retain_length += len(tmpdata)
            base_retain_data = datasets.concatenate_datasets([base_retain_data, tmpdata])

        base_forget_data = datasets.concatenate_datasets([
            base_forget_data, base_retain_data
        ]) 
        self.forget_data = base_forget_data
        self.eval_sets = {
            'forget': self.forget_eval,
            'retain': self.retain_eval,
            'perturb': self.perturb_eval,
            'paraphrase': self.paraphrase_eval,
        }
        print("In all ToFU Train: ", self.forget_length, self.retain_length)


def collect_expand_data(
    expand_qanum=10, path="data/aug_data/tofu/forget10_perturbed/paraphrase_res.csv",
):
    res = []
    import pandas as pd
    df = pd.read_csv(path)
    for idx, line in df.iterrows():
        para_question = list(set(eval(line.iloc[2])))
        para_answer = list(set(eval(line.iloc[3])))
        tmpres = []
        for para_q in para_question:
            for para_a in para_answer:
                tmpres.append((para_q, para_a))
        tmpres = tmpres[:expand_qanum]
        res.extend(tmpres)
    print("Expand num: ", len(res))
    return res

def collect_perturb_data(
    expand_qanum=10, path="data/aug_data/tofu/forget10_perturbed/perturb_res.csv",
):
    res = []
    import pandas as pd
    df = pd.read_csv(path)
    for idx, line in df.iterrows():
        para_question = line.iloc[2]
        para_answer = list(set(eval(line.iloc[3])))
        tmpres = []
        for para_a in para_answer:
            tmpres.append((para_question, para_a))
        tmpres = tmpres[:expand_qanum]
        res.extend(tmpres)
    print("Perturb num: ", len(res))
    return res
