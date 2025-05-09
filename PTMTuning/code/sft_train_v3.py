import os
import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTTrainingArguments
import matplotlib.pyplot as plt


def plot_loss(losses, output_dir):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()


@hydra.main(config_path='conf', config_name='conf_abl_semi')
def main(cfg: DictConfig):
    # Initialize tokenizer and special tokens
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    special_tokens = cfg.model.get('special_tokens', []) + ['<COT_START>', '<COT_END>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    # Load model and resize embeddings
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    # Custom dataset with loss_mask to ignore prompt tokens in loss
    class CotDataset(torch.utils.data.Dataset):
        def __init__(self, samples, tokenizer, max_length, templates):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.templates = templates if isinstance(templates, list) else [templates]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            entry = self.samples[idx]
            template = self.templates[idx % len(self.templates)]
            # build prompt
            prompt = template.format(
                question=entry['question'],
                correct=entry['correct'],
                incorrect=entry['incorrect'],
                examples="\n".join([
                    f"Q: {ex['question']}\nIncorrect: {ex['incorrect']}\nReason: {ex['reason']}" 
                    for ex in entry.get('example_cots', [])
                ]),
                cot_start='<COT_START>', cot_end='<COT_END>'
            )
            cot = entry['cot']
            final = entry['misconception']
            # full sequence = prompt + cot + final
            full = prompt + cot + final

            enc_prompt = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length)
            enc_full = self.tokenizer(full, truncation=True, padding='max_length', max_length=self.max_length)

            input_ids = torch.tensor(enc_full['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(enc_full['attention_mask'], dtype=torch.long)
            labels = torch.tensor(enc_full['input_ids'], dtype=torch.long)

            # Mask out prompt tokens in labels
            prompt_len = sum(enc_prompt['attention_mask'])  # count non-padded prompt tokens
            labels[:prompt_len] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

    # load samples and templates from config
    samples = cfg.data.samples
    prompt_templates = cfg.data.get('prompt_templates', [cfg.data.prompt_template])
    dataset = CotDataset(samples, tokenizer, cfg.train.max_length, prompt_templates)

    # Setup SFT training arguments including DeepSpeed
    training_args = SFTTrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accumulation_steps,
        learning_rate=cfg.train.lr,
        logging_steps=cfg.train.log_interval,
        fp16=cfg.train.mixed_precision == 'fp16',
        deepspeed=cfg.train.deepspeed,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # Train and collect losses
    result = trainer.train()
    losses = result.history.get('loss', [])

    # Plot loss curve
    if losses:
        plot_loss(losses, os.getcwd())


if __name__=='__main__':
    main()
