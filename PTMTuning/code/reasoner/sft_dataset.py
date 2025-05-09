import random
from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer

IGNORE_INDEX = -100


def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_path, add_eos_token=True)

    if tokenizer.eos_token == "":
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        tokenizer.eos_token = "</s>"

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = cfg.model.tokenizer.padding_side  # "left"
    return tokenizer


def find_token_instruction_masking(input_ids, token_pattern):
    """Find the last occurrence of a token_pattern in a list."""
    ret = []
    token_pattern_len = len(token_pattern)
    for ex_input_ids in input_ids:
        search_end = len(ex_input_ids)
        found = False
        for j in range(search_end - token_pattern_len, -1, -1):
            if ex_input_ids[j : j + token_pattern_len] == token_pattern:
                ret.append(j + token_pattern_len)
                found = True
                break
        if not found:
            ret.append(0)  # If not found, return 0 # assuming left truncation
    return ret


class MathDataset:
    """
    Dataset class for processing EEDI math MCQs into query/content inputs for retrieval
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)
        self.rng = random.Random(cfg.seed)

        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.token_pattern = self.tokenizer.encode("Answer:\n", add_special_tokens=False)

    def preprocess_function(self, df):
        formatted_texts = []
        system = "Analyze the incorrect answer to detect flaws in the student's reasoning."

        for _, row in df.iterrows():
            question = row["QuestionText"]
            correct_answer = row["CorrectAnswerText"]
            incorrect_answer = row["InCorrectAnswerText"]
            # related_misconceptions = row["related_misconceptions"]

            user_message = f"Question: {question}\nCorrect Answer: {correct_answer}\nIncorrect Answer: {incorrect_answer}"
            # user_message = f"{user_message}\nSuspected Misconceptions: {related_misconceptions}"

            assistant_message = row["Explanation"]

            # conversation = [
            #     {"role": "system", "content": system},
            #     {"role": "user", "content": user_message},
            #     {"role": "assistant", "content": assistant_message},
            # ]

            # text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False).strip()

            text = f"{system}\n\nQuery: {user_message}\nAnswer:\n{assistant_message}{self.eos_token}"

            formatted_texts.append(text)

        df["text"] = formatted_texts

        return df

    def tokenize(self, examples):
        tokenized = self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_length=True,
        )
        # pdb.set_trace()
        # get labels --
        labels = deepcopy(tokenized["input_ids"])
        assistant_start_idxs = find_token_instruction_masking(tokenized["input_ids"], self.token_pattern)

        for idx, src_len in enumerate(assistant_start_idxs):
            labels[idx][:src_len] = [IGNORE_INDEX] * src_len

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "length": tokenized["length"],
            "labels": labels,
        }

    def get_dataset(self, df):
        df = df.copy()
        df = self.preprocess_function(df).reset_index(drop=True)

        ds = Dataset.from_pandas(df)
        ds = ds.map(self.tokenize, batched=True)

        return ds