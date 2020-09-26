from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
import random

class EasyGPT2:

    def __init__(self,
                 model_size='small',
                 epochs=10,
                 batch_size=1,
                 learning_rate=2e-4,
                 warmup_steps=2000,
                 eos='<EOS>',
                 block_size=512):

        if not torch.cuda.is_available():
            raise RuntimeError('No cuda GPU available.')

        if model_size == 'small':
            gpt_size = 'gpt2'
        elif model_size == 'medium':
            gpt_size = 'gpt2-medium'
        elif model_size == 'large':
            gpt_size = 'gpt2-large'
        else:
            raise ValueError('"{}" is not a valid `model_size`.'.format(model_size))

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_size)
        self.model = GPT2LMHeadModel.from_pretrained(gpt_size)
        self.model = self.model.to('cuda')

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.eos = eos
        self.block_size = block_size

    def finetune(self, texts):

        self.texts = texts
        train = GPT2Dataset(self.tokenizer, self.eos.join(texts), block_size=self.block_size)
        train_loader = DataLoader(train, shuffle=False, batch_size=self.batch_size,)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.warmup_steps,
                                                    num_training_steps=-1)

        model = self.model.cuda()

        model.train()
        for epoch in range(self.epochs):

            print(f"Epoch {epoch} started")

            sum_loss = 0

            model.train()
            for data in train_loader:

                output = model(data.cuda(), labels = data.cuda())

                loss, _ = output[:2]

                loss.backward()

                sum_loss += loss.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            print(f"Training loss: {sum_loss / len(train_loader)}")

    def generate(self, prompt=None, max_length=200, min_length=None):

        if not prompt:
            prompt = random.choice(self.texts) + self.eos

        prompt_tokens = self.tokenizer.encode(prompt, return_tensors='pt')

        gen_tokens = self.model.generate(prompt_tokens.cuda(), do_sample=True, min_length=min_length, max_length=max_length, top_k=50)[0]

        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return gen_text[len(prompt):]

class GPT2Dataset(Dataset):

    def __init__(self, tokenizer, text, block_size=512):

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        self.examples = []

        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(tokenized_text[i : i + block_size] + [tokenizer.eos_token_id])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
