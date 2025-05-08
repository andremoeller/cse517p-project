#!/usr/bin/env python
import os
import random
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        """only one that has correct predictions outputted."""
        # your code here
        device = "cuda"
        torch.cuda.reset_peak_memory_stats(device)    # ← (1) clear any previous peaks
        preds = []
        start = time.time()
        for prompt in data:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded.input_ids.to("cuda")
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, position_ids=position_ids)[0]
            next_logits = logits[0, -1, :]
            topk = torch.topk(next_logits, k=3)
            top_ids = topk.indices.tolist()
            top_tokens_decoded = self.tokenizer.batch_decode([[tid] for tid in top_ids], clean_up_tokenization_spaces=False)
            preds.append(''.join(top_tokens_decoded))
        end = time.time()
        elapsed = end - start
        samples_per_sec = float(len(data)) / elapsed
        print(f"total inference time: {elapsed:.03f} for {len(data)} samples. {samples_per_sec:.03f} samples/sec")

        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb    = peak_bytes / (1024 ** 3)
        print(f"peak GPU memory allocated: {peak_gb:.2f} GB")
        return preds

    # loop over this
    def run_pred(self, data, batch_size: int = 8, max_length: int = 16):
        """
        Batched next-character prediction with EvaByte.
        Each line of the output contains the 3 most-likely next bytes/characters.

        max_length >= 16 is necessary to get acceptable accuracy (on example set).

        batch size, max_length:
        - 1, 12: 13683MiB, 60-80%
        - total inference time: 151.830s for 2000 samples (13.173 samples/sec)
        - peak GPU memory allocated: 13.11 GB
        - 16, [8-64]: OOM
        - 
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # make sure we have a pad token ID (EvaByte defines one, but be safe)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        preds = []
        t0 = time.time()
        total = len(data)

        for start in range(0, total, batch_size):
            batch_prompts = data[start:start + batch_size]
            print(f"batch {start//batch_size+1}/{(total-1)//batch_size+1} "
                  f"({start}/{total})")

            # tokenize & pad to longest sequence in the batch
            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=False
            )
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side    = "right"
            input_ids = encoded.input_ids.to(device)

            # seq_len per example (number of non-pad tokens)
            pad_id = self.tokenizer.pad_token_id
            seq_lens = (input_ids != pad_id).sum(dim=1)     # (batch,)

            # 2. Build position_ids tensor
            max_len = input_ids.size(1)
            position_ids = torch.arange(max_len, dtype=torch.long,
                                        device=device).unsqueeze(0).expand_as(input_ids)

            # 3. Forward pass
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids
                )[0]                       # shape (batch, max_len, vocab)

            # 4. Collect top-3 next-byte predictions for each prompt
            for i, L in enumerate(seq_lens):
                next_logits = logits[i, L - 1]              # logits at last real token
                top_ids = torch.topk(next_logits, k=3).indices.tolist()
                top_chars = self.tokenizer.batch_decode(
                    [[tid] for tid in top_ids],
                    clean_up_tokenization_spaces=False
                )
                preds.append("".join(top_chars))

        #  5. Stats
        elapsed = time.time() - t0
        print(f"total inference time for batch_size {batch_size}. max length {max_length}: {elapsed:.3f}s for {total} samples "
              f"({total/elapsed:.3f} samples/sec)")

        if device == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"peak GPU memory allocated: {peak_gb:.2f} GB")

        return preds


    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data, batch_size=8, max_length=16)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
