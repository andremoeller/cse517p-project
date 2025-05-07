#!/usr/bin/env python
import os
import random
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        # small, 300M
        # base,  500M
        # large, 1.2B
        # xl,    3.7B
        # xxl,   13B
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-xl", use_fast=True)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(
            "google/byt5-xl",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

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

    def run_pred(self, data, batch_size: int = 32):
        """
        Minibatched inference:

        - byt5-xl: 
        """
        # TODO: cpu mode isn't quite ready -- need some changes to make apple mps (metal performance shaders) work.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        preds = []
        start = time.time()
        num_prompts = len(data)
        batch_count = 0
        for idx in range(0, num_prompts, batch_size):
            batch_count += 1
            start_batch = time.time()
            batch_prompts = data[idx : idx + batch_size]
            done = min(idx, num_prompts)
            print(f"batch {idx / batch_size}/{len(data) // batch_size}; {done}/{num_prompts} inferences")

            # Tokenize the whole batch, pad to longest in batch
            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            ).to(device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=1,
                    do_sample=False
                )

            # grab the generated token for each in the batch
            # output_ids shape: (batch, seq_len+1)
            # so we take output_ids[:, -1]
            next_ids = output_ids[:, -1]
            # decode each single-token tensor
            for tid in next_ids:
                preds.append(
                    self.tokenizer.decode(
                        [int(tid)],
                        clean_up_tokenization_spaces=False
                    )
                )
            end_batch = time.time()
            elapsed_batch = end_batch - start_batch
            samples_per_sec_batch = batch_size / elapsed_batch
            print(f"minibatch with {batch_size} samples: {elapsed_batch * 1000:.03f} ms ({samples_per_sec_batch:.03f} samples/sec)")

        end = time.time()
        elapsed = end - start
        samples_per_sec = len(data) / elapsed
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        print(f"total inference time: {elapsed:.03f}s for {num_prompts} samples "
            f"({samples_per_sec:.03f} samples/sec)")
        if device == "cuda":
            print(f"peak GPU memory allocated: {peak_gb:.2f} GB")
        return preds

    def run_pred_single(self, data):
        """
        Predicts without batching:
        - 

        - with byt5-small: ~39 samples/sec without batching, peak 0.57GB mem allocated.
          - nvidia-smi on 20K examples: 849MiB/23034MiB, ~15-20%
        - with byt5-xl: ~16 samples/sec, 6.97GB peak. (~2x model), on 20K.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.reset_peak_memory_stats(device)
        preds = []
        start = time.time()
        num_prompts = len(data)
        for idx, prompt in enumerate(data):
            if idx % 100 == 0:
                elapsed = time.time() - start
                samples_per_sec = idx / elapsed
                print(f"{idx}/{num_prompts} inferences done, {samples_per_sec:.03f} samples/sec")
            
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,  # arbitrary; default is 1e30.
            ).to(device)
            self.tokenizer.truncation_side = "left" # we're interested in rightmost bytes.

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=1,
                    do_sample=False
                )

            next_token = output_ids[0, -1:]
            pred_char = self.tokenizer.decode(next_token, clean_up_tokenization_spaces=False)
            preds.append(pred_char)

        end = time.time()
        elapsed = end - start
        samples_per_sec = len(data) / elapsed
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        print(f"total inference time: {elapsed:.03f}s for {len(data)} samples "
            f"({samples_per_sec:.03f} samples/sec)")
        print(f"peak GPU memory allocated: {peak_gb:.2f} GB")
        return preds

    def save(self, work_dir):
        pass
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            # f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        pass
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            # dummy_save = f.read()
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
        # model = MyModel()
        print('Loading training data')
        # train_data = MyModel.load_training_data()
        print('Training')
        # model.run_train(train_data, args.work_dir)
        print('Saving model')
        # model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
