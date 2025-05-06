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
        # self.model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")
        # self.model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", torch_dtype=torch.bfloat16, trust_remote_code=True, load_in_8bit=True).eval().to("cuda")
        #     assert q.dtype in [torch.bfloat16, torch.float], "Only support bf16 and fp32 for now"
        # self.model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", device_map="auto", trust_remote_code=True, load_in_8bit=True).eval()
        # self.model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", device_map="auto", trust_remote_code=True).eval()
        self.model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        # "W0506 07:43:07.249000 28992 site-packages/torch/_inductor/utils.py:1137] [0/0] Not enough SMs to use max_autotune_gemm mode"
        # very, very slow.
        self.model = torch.compile(self.model)
        # warmup attempt for better perf reading:
        _ = self.model(input_ids=torch.zeros(1,16,device="cuda",dtype=torch.long), use_cache=False)
        torch.cuda.synchronize()
        # self.model = torch.compile(self.model)

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

    
    def run_preds(self, data, batch_size: int = 2):
        """
        batch size, max_length for tokenizer:
        - 32, 32: OOM
        - 16, 32: OOM
        - 16, 16: OOM
        - 2, 16: OOM
        - 8, 32:  assert chunk_mask is not None: File "/root/.cache/huggingface/modules/transformers_modules/evabyte/EvaByte/1c6283b2c439b731897e4202af789da99ba9ace2/eva_agg_kernel.py", line 1420, in triton_eva_agg_fwd
        - 4, 32: assert chunk_mask is not None:  File "/root/.cache/huggingface/modules/transformers_modules/evabyte/EvaByte/1c6283b2c439b731897e4202af789da99ba9ace2/eva_agg_kernel.py", line 1420, in triton_eva_agg_fwd 
        - 2, 32: assert chunk_mask is not None:  File "/root/.cache/huggingface/modules/transformers_modules/evabyte/EvaByte/1c6283b2c439b731897e4202af789da99ba9ace2/eva_agg_kernel.py", line 1420, in triton_eva_agg_fwd 
        - 2, 2: works
        - 4, 2: works, bad accuracy, 3.88 samples/sec
        - 4, 4: assert chunk_mask is not None
        - 4, 2: again works, peak 20.11GB, 5.204 samples/sec
        - original run pred: 3.4 samples/sec, 13.11GB.
        - original run pred: 3.218 samples/sec, 13.11GB.
        - original run pred: 4.47 samples/sec, 13.11GB.
        - original run pred w/ torch compile: 4.47 samples/sec, 13.11GB.



        """
        device = "cuda"
        all_preds = []
        torch.cuda.reset_peak_memory_stats(device)    # ← (1) clear any previous peaks
        start = time.time()
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            # Tokenize & pad on CPU
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=2,
                return_attention_mask=True,
            )
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side    = "right"

            input_ids = encoded.input_ids.to(device, non_blocking=True)
            attention_mask = encoded.attention_mask.to(device, non_blocking=True)


            # If your model truly needs position_ids, build one per batch:
            # seq_len = input_ids.shape[1]
            # position_ids = torch.arange(
            #    seq_len, device=device
            # ).unsqueeze(0).expand(input_ids.size(0), -1)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Grab the last-token logits for each example
            next_logits = logits[:, -1, :]
            topk_vals, topk_idxs = next_logits.topk(k=3, dim=-1)

            # Decode in batch: this returns List[List[str]] of shape (batch, 3)
            # Note: batch_decode wants a list of lists of token-ids
            token_lists = [
                idxs.cpu().tolist() for idxs in topk_idxs
            ]
            str_lists = self.tokenizer.batch_decode(
                token_lists,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )

            # Collect predictions—join the 3 tokens, or pick str_lists[i][0] if only top-1
            all_preds.extend("".join(lst) for lst in str_lists)
        end = time.time()
        elapsed = end - start
        samples_per_sec = float(len(data)) / elapsed
        print(f"total inference time: {elapsed:.03f} for {len(data)} samples. {samples_per_sec:.03f} samples/sec")
        return all_preds

    def run_predss(self, data, batch_size: int = 1):
        """
        batch size, max_length for tokenizer:
        - 32, 32: OOM
        - 16, 32: OOM
        - 16, 16: OOM
        - 2, 16: OOM
        - 8, 32:  assert chunk_mask is not None: File "/root/.cache/huggingface/modules/transformers_modules/evabyte/EvaByte/1c6283b2c439b731897e4202af789da99ba9ace2/eva_agg_kernel.py", line 1420, in triton_eva_agg_fwd
        - 4, 32: assert chunk_mask is not None:  File "/root/.cache/huggingface/modules/transformers_modules/evabyte/EvaByte/1c6283b2c439b731897e4202af789da99ba9ace2/eva_agg_kernel.py", line 1420, in triton_eva_agg_fwd 
        - 2, 32: assert chunk_mask is not None:  File "/root/.cache/huggingface/modules/transformers_modules/evabyte/EvaByte/1c6283b2c439b731897e4202af789da99ba9ace2/eva_agg_kernel.py", line 1420, in triton_eva_agg_fwd 
        - 2, 2: works
        - 4, 2: works, bad accuracy, 3.88 samples/sec
        - 4, 4: assert chunk_mask is not None
        - 4, 2: again works, peak 20.11GB, 5.204 samples/sec
        - 2, 2: 3.353 samples/sec, 16.11GB.
        --->  each batch row size takes ~2GB?
        - 2, 4: assert chunk_mask is not None:
        - 2, 8: assert chunk_mask is not None
        - 1, 32: 
        """

        device = "cuda"
        torch.cuda.reset_peak_memory_stats(device)    # ← (1) clear any previous peaks

        all_preds = []
        start = time.time()

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=32,
                return_attention_mask=True,
            )
            input_ids      = encoded.input_ids.to(device, non_blocking=True)
            attention_mask = encoded.attention_mask.to(device, non_blocking=True)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            next_logits = logits[:, -1, :]
            topk_idxs   = next_logits.topk(k=3, dim=-1).indices
            token_lists = [idxs.cpu().tolist() for idxs in topk_idxs]
            str_lists   = self.tokenizer.batch_decode(
                token_lists,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            all_preds.extend("".join(lst) for lst in str_lists)

        end = time.time()
        elapsed = end - start
        s_per_s = len(data) / elapsed

        # (3) read back the peak
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb    = peak_bytes / (1024 ** 3)

        print(f"total inference time: {elapsed:.03f}s for {len(data)} samples "
            f"({s_per_s:.03f} samples/sec)")
        print(f"peak GPU memory allocated: {peak_gb:.2f} GB")

        return all_preds

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
