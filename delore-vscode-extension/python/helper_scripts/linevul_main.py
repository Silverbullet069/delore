# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function
import argparse
import glob
import json
import logging
import os
import pickle
import random
import re
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing
from linevul_model import Model
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, PrecisionRecallDisplay
from sklearn.metrics import auc
import matplotlib.pyplot as plt
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# word-level tokenizer
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


def levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    matrix = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]

    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(
                    matrix[i - 1][j - 1] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j] + 1
                )

    return matrix[-1][-1]


def are_strings_similar(a, b, similarity_threshold=0.7):  # 0.7 is good enough
    distance = levenshtein(a, b)
    longest_length = max(len(a), len(b))

    if longest_length == 0:
        return True

    similarity = (longest_length - distance) / longest_length
    return similarity >= similarity_threshold


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label  # label might be missing, due to 'use' mode


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        elif file_type == 'use':
            file_path = args.use_data_file

        self.examples = []
        df = pd.read_csv(file_path)

        funcs = df["processed_func"].tolist()
        logger.info(f'TextDataset {file_type} funcs: {funcs}')

        # labels are the "ground truth", if you planned to use the model, you can leave this blank.
        labels = df["target"].tolist()
        logger.info(f'TextDataset {file_type} labels: {labels}')

        for i in tqdm(range(len(funcs))):
            # Convert function to features using tokenizer
            self.examples.append(convert_examples_to_features(
                funcs[i], labels[i], tokenizer, args))

        logger.info(f'TextDataset {file_type} examples: {self.examples}')

        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(
                    ' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def convert_examples_to_features(func, label, tokenizer, args):
    if args.use_word_level_tokenizer:
        encoded = tokenizer.encode(func)
        encoded = encoded.ids
        if len(encoded) > 510:
            encoded = encoded[:510]
        encoded.insert(0, 0)
        encoded.append(2)
        if len(encoded) < 512:
            padding = 512 - len(encoded)
            for _ in range(padding):
                encoded.append(1)
        source_ids = encoded
        source_tokens = []
        return InputFeatures(source_tokens, source_ids, label)

    # source

    # Splitting input text into individual tokens, which can be words, subwords, characters, ... depends on tokenizer's config
    # Maximum length of tokens allowed, minus 2? ensure there is space for additional special tokens like [CLS] or [SEP], padding tokens, ... which are commonly added to the beginning and end of token sequences in NLP tasks.
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-2]

    logger.info(f'convert_example_to_features() - code_tokens: {code_tokens}')

    # That's why you need minus 2
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    logger.info(
        f'convert_example_to_features() - source_tokens: {source_tokens}')

    # Token IDs are numerical encoding of tokens, which are often used as input to ML models, Neural Network.
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    logger.info(f'convert_example_to_features() - source_ids: {source_ids}')

    # Ensure that input sequences have uniform lengths
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    logger.info(
        f'convert_example_to_features() - padding_length: {padding_length}')
    logger.info(f'convert_example_to_features() - source_ids: {source_ids}')

    return InputFeatures(source_tokens, source_ids, label)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",
                args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(input_ids=inputs_ids, labels=labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer,
                                       eval_dataset, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    PrecisionRecallDisplay.from_predictions(
        y_trues, logits[:, 1], name='LineVul')
    plt.savefig(f'eval_precision_recall_{args.model_name}.pdf')

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader

    # iterate over the elements of a dataset in a sequential order, without shuffling
    # when training machine learing models, it's common to shuffle the data during training to prevent the model from learning spurious patterns that maybe present due to the order of the data
    # but, it the cases of testing/evaluation, you must retain its order
    test_sampler = SequentialSampler(test_dataset)
    logger.info(f'test() - test_sampler: {test_sampler}')

    # Create batches of data for testing
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    logger.info(f'test() - test_dataloader: {test_dataloader}')

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("Num examples = %d", len(test_dataset))
    logger.info("Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    # Change mode to 'evaluation'
    model.eval()

    logits = []
    y_trues = []
    for batch in test_dataloader:
        # Move each tensor in the batch to the device specified by `args.device`
        (inputs_ids, labels) = [x.to(args.device) for x in batch]

        logger.info(f'Inputs Ids: {inputs_ids}')
        logger.info(f'Labels: {labels}')

        # Disable Gradient calculation
        # During the execution of the block, any code that wrapped insize this:
        # - won't tract operations
        # - won't build computational graph
        # - won't calculate gradients
        with torch.no_grad():
            # ! THIS IS IT
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

            logger.info('******* BEGIN torch.no_grad() *******')
            logger.info(f'lm_loss: {lm_loss}')
            logger.info(f'logit: {logit}')
            logger.info(f'eval_loss: {eval_loss}')
            logger.info(f'logits: {logits}')
            logger.info(f'y_trues: {y_trues}')
            logger.info('******* END torch.no_grad() *******')

        nb_eval_steps += 1

        logger.info('******* END BATCH ITERATION *******')

    logger.info('******* CALCULATE SCORES *******')

    # calculate scores
    logits = np.concatenate(logits, 0)  # turn List into NDArray
    logger.info(f'  logits: {logits}')

    y_trues = np.concatenate(y_trues, 0)  # turn List into NDArray
    logger.info(f'  y_trues: {y_trues}')

    y_preds = logits[:, 1] > best_threshold
    logger.info(f'  y_preds: {y_preds}')

    acc = accuracy_score(y_trues, y_preds)
    logger.info(f'  acc: {acc}')

    recall = recall_score(y_trues, y_preds)  # 90/90+10 = 0.9
    logger.info(f'  recall: {recall}')

    precision = precision_score(y_trues, y_preds)  # 90/90+0 = 1.0
    logger.info(f'  precision: {precision}')

    f1 = f1_score(y_trues, y_preds)
    logger.info(f'  f1: {f1}')

    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,
    }
    logger.info(f'test() - result: {result}')

    # Display pdf Precision + Recall
    PrecisionRecallDisplay.from_predictions(
        y_trues, logits[:, 1], name="LineVul")
    plt.savefig(f'test_precision_recall_{args.model_name}.pdf')

    logger.info("***** Test results *****")

    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    logits = [l[1] for l in logits]
    logger.info(f'test() - logits: {logits}')

    result_df = generate_result_df(logits, y_trues, y_preds, args)
    logger.info(f'test() - result_df:')
    logger.info(result_df)
    logger.info(result_df.columns)
    logger.info(len(result_df))

    sum_lines, sum_flaw_lines = get_line_statistics(result_df)
    logger.info(f'test() - sum_lines: {sum_lines}')
    logger.info(f'test() - sum_flaw_lines: {sum_flaw_lines}')

    # write raw predictions if needed
    if args.write_raw_preds:
        write_raw_preds_csv(args, y_preds)

    # define reasoning method
    if args.reasoning_method == "all":
        all_reasoning_method = [
            "attention", "lig", "saliency", "deeplift", "deeplift_shap", "gradient_shap"]
    else:
        all_reasoning_method = [args.reasoning_method]

    # TODO: examine this option
    if args.do_sorting_by_line_scores:
        # (RQ2) Effort@TopK%Recall & Recall@TopK%LOC for the whole test set
        # flatten the logits
        for reasoning_method in all_reasoning_method:
            dataloader = DataLoader(
                test_dataset, sampler=test_sampler, batch_size=1, num_workers=0)
            progress_bar = tqdm(dataloader, total=len(
                dataloader), desc=reasoning_method)

            all_pos_score_label = []
            all_neg_score_label = []
            index = 0
            total_pred_as_vul = 0

            for mini_batch in progress_bar:
                logger.info(f"test() - ")

                # if predicted as vulnerable
                if result_df["logits"][index] > 0.5:
                    total_pred_as_vul += 1
                    all_lines_score_with_label = \
                        line_level_localization(flaw_lines=result_df["flaw_line"][index],
                                                tokenizer=tokenizer,
                                                model=model,
                                                mini_batch=mini_batch,
                                                original_func=result_df["processed_func"][index],
                                                args=args,
                                                top_k_loc=None,
                                                top_k_constant=None,
                                                reasoning_method=reasoning_method,
                                                index=index)
                    all_pos_score_label.append(all_lines_score_with_label)
                    logger.info(all_lines_score_with_label)
                    logger.info(all_pos_score_label)

                # else predicted as non vulnerable
                else:
                    all_lines_score_with_label = \
                        line_level_localization(flaw_lines=result_df["flaw_line"][index],
                                                tokenizer=tokenizer,
                                                model=model,
                                                mini_batch=mini_batch,
                                                original_func=result_df["processed_func"][index],
                                                args=args,
                                                top_k_loc=None,
                                                top_k_constant=None,
                                                reasoning_method=reasoning_method,
                                                index=index)
                    all_neg_score_label.append(all_lines_score_with_label)

                    logger.info(all_lines_score_with_label)
                    logger.info(all_neg_score_label)

                # Next function
                index += 1

            is_attention = True if reasoning_method == "attention" else False
            total_pos_lines, pos_rank_df = rank_lines(
                all_pos_score_label, is_attention, ascending_ranking=False)

            logger.info(total_pos_lines)
            logger.info(pos_rank_df)

            if is_attention:
                total_neg_lines, neg_rank_df = rank_lines(
                    all_neg_score_label, is_attention, ascending_ranking=True)

                logger.info(total_neg_lines, neg_rank_df)
            else:
                total_neg_lines, neg_rank_df = rank_lines(
                    all_neg_score_label, is_attention, ascending_ranking=False)

            effort, inspected_line = top_k_effort(
                pos_rank_df, sum_lines, sum_flaw_lines, args.effort_at_top_k)
            logger.info(effort)
            logger.info(inspected_line)

            recall_value = top_k_recall(
                pos_rank_df, neg_rank_df, sum_lines, sum_flaw_lines, args.top_k_recall_by_lines)
            logger.info(recall_value)

            logger.info(
                f"total functions predicted as vulnerable: {total_pred_as_vul}")

            to_write = ""
            to_write += "\n" + f"Reasoning Method: {reasoning_method}" + "\n"
            to_write += f"total predicted vulnerable lines: {total_pos_lines}" + "\n"
            logger.info(f"total predicted vulnerable lines: {total_pos_lines}")

            to_write += f"total lines: {sum_lines}" + "\n"
            logger.info(f"total lines: {sum_lines}")

            to_write += f"total flaw lines: {sum_flaw_lines}" + "\n"
            logger.info(f"total flaw lines: {sum_flaw_lines}")

            vul_as_vul = sum(pos_rank_df["label"].tolist())
            to_write += f"total flaw lines in predicted as vulnerable: {vul_as_vul}" + "\n"
            logger.info(
                f"total flaw lines in predicted as vulnerable: {vul_as_vul}")

            to_write += f"top{args.effort_at_top_k}-Effort: {effort}" + "\n"
            logger.info(f"top{args.effort_at_top_k}-Effort: {effort}")

            to_write += f"total inspected line to find out {args.effort_at_top_k} of flaw lines: {inspected_line}" + "\n"
            logger.info(
                f"total inspected line to find out {args.effort_at_top_k} of flaw lines: {inspected_line}")

            to_write += f"top{args.top_k_recall_by_lines}-Recall: {recall_value}" + "\n"
            logger.info(
                f"top{args.top_k_recall_by_lines}-Recall: {recall_value}")

            with open("./results/rq2_result.txt", "a") as f:
                f.write(to_write)

    if args.do_sorting_by_pred_prob:
        rank_df = rank_dataframe(
            df=result_df, rank_by="logits", ascending=False)
        logger.info(f'test() - rank_df: {rank_df}')

        effort, inspected_line = top_k_effort_pred_prob(
            rank_df, sum_lines, sum_flaw_lines, args.effort_at_top_k, label_col_name="y_preds")
        logger.info(f'test() - effort: {effort}')
        logger.info(f'test() - inspected_line: {inspected_line}')

        top_k_recall_val = top_k_recall_pred_prob(
            rank_df, sum_lines, sum_flaw_lines, args.top_k_recall_by_pred_prob, label_col_name="y_preds")
        logger.info(f'test() - top_k_recall_val: {top_k_recall_val}')

        with open("./results/rq2_result_pred_prob.txt", "a") as f:
            f.write(
                f"\n Sorted By Prediction Probabilities \n top{args.effort_at_top_k}-Effort: {effort} \n top{args.top_k_recall_by_pred_prob}-Recall: {top_k_recall_val}")
            logger.info(
                f"\n Sorted By Prediction Probabilities \n top{args.effort_at_top_k}-Effort: {effort} \n top{args.top_k_recall_by_pred_prob}-Recall: {top_k_recall_val}")

    # (RQ3) Line level evaluation for True Positive cases
    if args.do_local_explanation:
        for reasoning_method in all_reasoning_method:
            logger.info(
                f"***** Running Explanation - {reasoning_method} *****")

            # Return indices where condition is true
            correct_indices = np.where((y_trues == y_preds))
            correct_indices = list(correct_indices[0])
            correct_prediction_count = len(correct_indices)

            logger.info(
                f"test() - correct_indices (modified): {correct_indices}")
            logger.info(
                f"test() - correct_prediction_count: {correct_prediction_count}")

            tp_indices = np.where((y_trues == y_preds) & (y_trues == 1))
            tp_indices = list(tp_indices[0])
            correct_vulnerable_count = len(tp_indices)

            logger.info(f"test() - tp_indices (modified): {tp_indices}")
            logger.info(
                f"test() - correct vulnerable count: {correct_vulnerable_count}")

            # ! HERE IT IS, DIG DEEPER
            # localization part
            dataloader = DataLoader(
                test_dataset, sampler=test_sampler, batch_size=1, num_workers=0)  # 1 sample per batch, load data in main process, do not generate subprocess

            # prepare data for line-level reasoning
            df = pd.read_csv(args.test_data_file)

            # stats for line-level evaluation
            top_k_locs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            # why turn this into array with 1 element?
            top_k_constant = [args.top_k_constant]

            sum_total_lines = 0
            total_flaw_lines = 0
            total_function = 0
            all_top_10_correct_idx = []
            all_top_10_not_correct_idx = []

            # for CodeBERT reasoning
            total_correctly_predicted_flaw_lines = [
                0 for _ in range(len(top_k_locs))]
            total_correctly_localized_function = [
                0 for _ in range(len(top_k_constant))]
            total_min_clean_lines_inspected = 0
            ifa_records = []
            top_10_acc_records = []
            total_max_clean_lines_inspected = 0

            # vulnerability exist but not applicable (flaw tokens are out of seq length)
            na_explanation_total = 0
            na_eval_results_512 = 0
            na_defective_data_point = 0

            # track progress
            # use DataLoader
            progress_bar = tqdm(dataloader, total=len(dataloader))

            # used to locate the row in test data
            index = 0
            for mini_batch in progress_bar:
                logger.info(f"test() - mini_batch: {mini_batch}")

                # if true positive (vulnerable predicted as vulnerable), do explanation
                if index in tp_indices:
                    # if flaw line exists
                    # if not exist, the data is as type of float (nan)
                    if isinstance(df["flaw_line"][index], str) and isinstance(df["flaw_line_index"][index], str):
                        line_eval_results = \
                            line_level_localization_tp(flaw_lines=df["flaw_line"][index],
                                                       tokenizer=tokenizer,
                                                       model=model,
                                                       mini_batch=mini_batch,
                                                       original_func=df["processed_func"][index],
                                                       args=args,
                                                       top_k_loc=top_k_locs,
                                                       top_k_constant=top_k_constant,
                                                       reasoning_method=reasoning_method,
                                                       index=index,
                                                       write_invalid_data=False)

                        logger.info(
                            f'test() - line_eval_results: {line_eval_results}')

                        if line_eval_results == "NA":
                            na_explanation_total += 1
                            na_eval_results_512 += 1
                        else:
                            total_function += 1
                            sum_total_lines += line_eval_results["total_lines"]
                            logger.info(
                                f'test() - sum_total_lines: {sum_total_lines}')

                            total_flaw_lines += line_eval_results["num_of_flaw_lines"]
                            logger.info(
                                f'test() - total_flaw_lines: {total_flaw_lines}')

                            # IFA metric
                            total_min_clean_lines_inspected += line_eval_results["min_clean_lines_inspected"]
                            logger.info(
                                f'test() - total_min_clean_lines_inspected: {total_min_clean_lines_inspected}')

                            # For IFA Boxplot
                            ifa_records.append(
                                line_eval_results["min_clean_lines_inspected"])
                            logger.info(f'test() - ifa_records: {ifa_records}')

                            # For Top-10 Acc Boxplot
                            # todo
                            # top_10_acc_records.append(line_eval_results[])

                            # All effort metric
                            total_max_clean_lines_inspected += line_eval_results["max_clean_lines_inspected"]
                            for j in range(len(top_k_locs)):
                                total_correctly_predicted_flaw_lines[
                                    j] += line_eval_results["all_correctly_predicted_flaw_lines"][j]
                            # top 10 accuracy
                            for k in range(len(top_k_constant)):
                                total_correctly_localized_function[
                                    k] += line_eval_results["all_correctly_localized_function"][k]
                            # top 10 correct idx and not correct idx
                            if line_eval_results["top_10_correct_idx"] != []:
                                all_top_10_correct_idx.append(
                                    line_eval_results["top_10_correct_idx"][0])
                            if line_eval_results["top_10_not_correct_idx"] != []:
                                all_top_10_not_correct_idx.append(
                                    line_eval_results["top_10_not_correct_idx"][0])

                            logger.info(
                                f'test() - total_max_clean_lines_inspected: {total_max_clean_lines_inspected}')
                            logger.info(
                                f'test() - total_correctly_predicted_flaw_lines: {total_correctly_predicted_flaw_lines}')
                            logger.info(
                                f'test() - total_correctly_localized_function: {total_correctly_localized_function}')
                            logger.info(
                                f'test() - all_top_10_correct_idx: {all_top_10_correct_idx}')
                            logger.info(
                                f'test() - all_top_10_not_correct_idx: {all_top_10_not_correct_idx}')
                    else:
                        na_explanation_total += 1
                        na_defective_data_point += 1
                index += 1

            # write IFA records for IFA Boxplot
            with open(f"./ifa_records/ifa_{reasoning_method}.txt", "w+") as f:
                f.write(str(ifa_records))
            # write Top-10 Acc records for Top-10 Accuracy Boxplot
            # todo
            # with open(f"./top_10_acc_records/top_10_acc_{reasoning_method}.txt", "w+") as f:
            #    f.write(str())

            logger.info(f"Total number of functions: {total_function}")
            logger.info(f"Total number of lines: {sum_total_lines}")
            logger.info(f"Total number of flaw lines: {total_flaw_lines}")
            logger.info(
                f"Total Explanation Not Applicable: {na_explanation_total}")
            logger.info(
                f"NA Eval Results (Out of 512 Tokens): {na_eval_results_512}")
            logger.info(f"NA Defective Data Point: {na_defective_data_point}")

            line_level_results = [{f"codebert_{reasoning_method}_top20%_recall":
                                   [round(total_correctly_predicted_flaw_lines[i] /
                                          total_flaw_lines, 2) * 100 for i in range(len(top_k_locs))],
                                   f"codebert_{reasoning_method}_top10_accuracy":
                                   [round(total_correctly_localized_function[i] / total_function, 2)
                                       * 100 for i in range(len(top_k_constant))],
                                   f"codebert_{reasoning_method}_ifa":
                                   round(total_min_clean_lines_inspected /
                                         total_function, 2),
                                   f"codebert_{reasoning_method}_recall@topk%loc_auc":
                                   auc(x=top_k_locs, y=[round(
                                       total_correctly_predicted_flaw_lines[i] / total_flaw_lines, 2) for i in range(len(top_k_locs))]),
                                   f"codebert_{reasoning_method}_total_effort":
                                   round(total_max_clean_lines_inspected /
                                         sum_total_lines, 2),
                                   "avg_line_in_one_func":
                                   int(sum_total_lines / total_function),
                                   "total_func":
                                   total_function,
                                   "all_top_10_correct_idx": all_top_10_correct_idx,
                                   "all_top_10_not_correct_idx": all_top_10_not_correct_idx}]

            with open('./results/line_level_correct_idx.pkl', 'wb') as f:
                pickle.dump(all_top_10_correct_idx, f)
            with open('./results/line_level_not_correct_idx.pkl', 'wb') as f:
                pickle.dump(all_top_10_not_correct_idx, f)

            logger.info("***** Line Level Result *****")
            logger.info(f'test() - line_level_results: {line_level_results}')

            # output results
            # with open("./results/local_explanation.pkl", "wb") as f:
            #    pickle.dump(line_level_results, f)


def generate_result_df(logits, y_trues, y_preds, args):
    df = pd.read_csv(args.test_data_file)
    all_num_lines = []
    # extract all functions in test file
    all_processed_func = df["processed_func"].tolist()

    for func in all_processed_func:
        # extract num lines of each functions in test file
        all_num_lines.append(get_num_lines(func))
        logger.info(f'generate_result_df() - func: {func}')
        logger.info(
            f'generate_result_df() - func_num_lines: {get_num_lines(func)}')
    logger.info(f'generate_result_df() - all_num_lines: {all_num_lines}')

    flaw_line_indices = df["flaw_line_index"].tolist()
    logger.info(
        f'generate_result_df() - flaw_line_indices: {flaw_line_indices}')

    all_num_flaw_lines = []
    total_flaw_lines = 0
    for indices in flaw_line_indices:
        if isinstance(indices, str):
            indices = indices.split(",")
            num_flaw_lines = len(indices)
            total_flaw_lines += num_flaw_lines
        else:
            num_flaw_lines = 0
        all_num_flaw_lines.append(num_flaw_lines)
    assert len(logits) == len(y_trues) == len(
        y_preds) == len(all_num_flaw_lines)

    # prettier-ignore
    return pd.DataFrame({
        "logits": logits,
        "y_trues": y_trues,
        "y_preds": y_preds,
        "index": list(range(len(logits))),
        "num_flaw_lines": all_num_flaw_lines,
        "num_lines": all_num_lines,
        "flaw_line": df["flaw_line"],
        "processed_func": df["processed_func"]
    })


def write_raw_preds_csv(args, y_preds):
    df = pd.read_csv(args.test_data_file)
    df["raw_preds"] = y_preds
    df.to_csv("./results/raw_preds.csv", index=False)


def get_num_lines(func):
    func = func.split("\n")
    func = [line for line in func if len(line) > 0]
    return len(func)


def get_line_statistics(result_df):
    total_lines = sum(result_df["num_lines"].tolist())
    total_flaw_lines = sum(result_df["num_flaw_lines"].tolist())
    return total_lines, total_flaw_lines


def rank_lines(all_lines_score_with_label, is_attention, ascending_ranking):
    # flatten the list
    all_lines_score_with_label = [
        line for lines in all_lines_score_with_label for line in lines]
    if is_attention:
        all_scores = [line[0].item() for line in all_lines_score_with_label]
    else:
        all_scores = [line[0] for line in all_lines_score_with_label]
    all_labels = [line[1] for line in all_lines_score_with_label]
    rank_df = pd.DataFrame({"score": all_scores, "label": all_labels})
    rank_df = rank_dataframe(rank_df, "score", ascending_ranking)
    return len(rank_df), rank_df


def rank_dataframe(df, rank_by: str, ascending: bool):
    df = df.sort_values(by=[rank_by], ascending=ascending)
    df = df.reset_index(drop=True)
    return df


def top_k_effort(rank_df, sum_lines, sum_flaw_lines, top_k_loc, label_col_name="label"):
    target_flaw_line = int(sum_flaw_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += 1
        if rank_df[label_col_name][i] == 1:
            caught_flaw_line += 1
        if target_flaw_line == caught_flaw_line:
            break
    effort = round(inspected_line / sum_lines, 4)
    return effort, inspected_line


def top_k_effort_pred_prob(rank_df, sum_lines, sum_flaw_lines, top_k_loc, label_col_name="y_preds"):
    target_flaw_line = int(sum_flaw_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += rank_df["num_lines"][i]
        if rank_df[label_col_name][i] == 1 or rank_df[label_col_name][i] is True:
            caught_flaw_line += rank_df["num_flaw_lines"][i]
        if caught_flaw_line >= target_flaw_line:
            break
    effort = round(inspected_line / sum_lines, 4)
    return effort, inspected_line


def top_k_recall(pos_rank_df, neg_rank_df, sum_lines, sum_flaw_lines, top_k_loc):
    target_inspected_line = int(sum_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    inspect_neg_lines = True
    for i in range(len(pos_rank_df)):
        inspected_line += 1
        if inspected_line > target_inspected_line:
            inspect_neg_lines = False
            break
        if pos_rank_df["label"][i] == 1 or pos_rank_df["label"][i] is True:
            caught_flaw_line += 1
    if inspect_neg_lines:
        for i in range(len(neg_rank_df)):
            inspected_line += 1
            if inspected_line > target_inspected_line:
                break
            if neg_rank_df["label"][i] == 1 or neg_rank_df["label"][i] is True:
                caught_flaw_line += 1
    return round(caught_flaw_line / sum_flaw_lines, 4)


def top_k_recall_pred_prob(rank_df, sum_lines: int, sum_flaw_lines: int, top_k_loc: float, label_col_name="y_preds"):
    target_inspected_line = int(sum_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += rank_df["num_lines"][i]
        if inspected_line > target_inspected_line:
            break
        if rank_df[label_col_name][i] == 1 or rank_df[label_col_name][i] is True:
            caught_flaw_line += rank_df["num_flaw_lines"][i]
    return round(caught_flaw_line / sum_flaw_lines, 4)


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def create_ref_input_ids(input_ids, ref_token_id, sep_token_id, cls_token_id):
    seq_length = input_ids.size(1)
    ref_input_ids = [cls_token_id] + [ref_token_id] * \
        (seq_length-2) + [sep_token_id]
    return torch.tensor([ref_input_ids])


def line_level_localization_tp(flaw_lines: str, tokenizer, model, mini_batch, original_func: str, args, top_k_loc: list, top_k_constant: list, reasoning_method: str, index: int, write_invalid_data: bool):
    # function for captum LIG.
    def predict(input_ids):
        return model(input_ids=input_ids)[0]

    def lig_forward(input_ids):
        logits = model(input_ids=input_ids)[0]
        logger.info(
            f"line_level_localization_tp() - lig_forward(): logits: {logits}")

        y_pred = 1  # for positive attribution, y_pred = 0 for negative attribution
        pred_prob = logits[y_pred].unsqueeze(-1)
        logger.info(
            f"line_level_localization_tp() - lig_forward(): pred_prob: {pred_prob}")

        return pred_prob

    flaw_line_seperator = "/~/"
    (input_ids, labels) = mini_batch

    # separate tensor from computational graph,
    ids = input_ids[0].detach().tolist()

    logger.info(f"line_level_localization_tp() - input_ids: {input_ids}")

    logger.info(
        f"line_level_localization_tp() - len(input_ids): {len(input_ids)}")
    logger.info(f"line_level_localization_tp() - labels: {labels}")

    logger.info(f"line_level_localization_tp() - ids: {ids}")
    logger.info(f"line_level_localization_tp() - len(ids): {len(ids)}")

    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    original_lines = ''.join(all_tokens).split("Ċ")

    logger.info(f"line_level_localization_tp() - all_tokens: {all_tokens}")
    logger.info(
        f"line_level_localization_tp() - len(all_tokens): {len(all_tokens)}")
    logger.info(
        f"line_level_localization_tp() - original_lines: {original_lines}")

    # flaw line verification
    # get flaw tokens ground truth
    flaw_lines = get_all_flaw_lines(
        flaw_lines=flaw_lines, flaw_line_seperator=flaw_line_seperator)
    logger.info(
        f"line_level_localization_tp() - flaw_lines (ground truth): {flaw_lines}")

    flaw_tokens_encoded = encode_all_lines(
        all_lines=flaw_lines, tokenizer=tokenizer)
    logger.info(
        f"line_level_localization_tp() - flaw_tokens_encoded (ground_truth): {flaw_tokens_encoded}")

    verified_flaw_lines = []
    do_explanation = False
    encoded_all = ''.join(all_tokens)
    logger.info(f"line_level_localization_tp() - encoded_all: {encoded_all}")
    for i in range(len(flaw_tokens_encoded)):
        encoded_flaw = ''.join(flaw_tokens_encoded[i])
        logger.info(
            f"line_level_localization_tp() - encoded_flaw: {encoded_flaw}")

        # Pretty weird behavior, this is unchanged after loop, why recalculate each loop?
        # encoded_all = ''.join(all_tokens)
        # logger.info(f"line_level_localization_tp() - encoded_all: {encoded_all}")

        if encoded_flaw in encoded_all:
            verified_flaw_lines.append(flaw_tokens_encoded[i])
            logger.info(
                f"line_level_localization_tp() - verified_flaw_lines: {verified_flaw_lines}")
            do_explanation = True

    logger.info(
        f"line_level_localization_tp() - verified_flaw_lines (loop finished): {verified_flaw_lines}")

    # do explanation if at least one flaw line exist in the encoded input
    if do_explanation:

        # ! ATTENTION MECHANISM
        if reasoning_method == "attention":

            # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
            input_ids = input_ids.to(args.device)
            logger.info(
                f'line_level_localization_tp() - input_ids: {input_ids}')

            # ! VERY IMPORTANT
            prob, attentions = model(
                input_ids=input_ids, output_attentions=True)
            logger.info(f'line_level_localization_tp() - prob: {prob}')
            logger.info(
                f'line_level_localization_tp() - attentions: {attentions}')

            # take from tuple then take out mini-batch attention values
            attentions = attentions[0][0]  # 5D Tensor
            logger.info(
                f'line_level_localization_tp() - attentions (modified): {attentions}')  # 3D tensor
            logger.info(
                f'line_level_localization_tp() - len(attentions): {len(attentions)}')  # len of 3D tensor

            attention = None
            # go into the layer
            logger.info(
                f'line_level_localization_tp() ********* GO INTO THE LAYER *********')
            for i in range(len(attentions)):
                layer_attention = attentions[i]
                logger.info(
                    f'line_level_localization_tp() - attentions[i]: {attentions[i]}')

                # summerize the values of each token dot other tokens
                layer_attention = sum(layer_attention)  # sum(attensions[i])
                logger.info(
                    f'line_level_localization_tp() - layer_attention: {layer_attention}')

                if attention is None:
                    attention = layer_attention
                else:
                    attention += layer_attention
                logger.info(
                    f'line_level_localization_tp() - attention: {attention}')

            logger.info(
                f'line_level_localization_tp() ********* END OF FOR LOOP *********')

            # clean att score for <s> and </s>
            # If you notice, they have exceptionally high score
            attention = clean_special_token_values(attention, padding=True)
            logger.info(
                f'line_level_localization_tp() - attention (for loop completed): {attention}')

            # attention should be 1D tensor with seq length representing each token's attention value
            word_att_scores = get_word_att_scores(
                all_tokens=all_tokens, att_scores=attention)
            logger.info(
                f'line_level_localization_tp() - word_att_scores: {word_att_scores}')

            all_lines_score, flaw_line_indices = get_all_lines_score(
                word_att_scores, verified_flaw_lines)
            logger.info(
                f'line_level_localization_tp() - all_lines_score: {all_lines_score}')
            logger.info(
                f'line_level_localization_tp() - flaw_line_indices: {flaw_line_indices}')

            # return if no flaw lines exist
            if len(flaw_line_indices) == 0:
                return "NA"

            total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, top_10_correct_idx, top_10_not_correct_idx \
                = \
                line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                      top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=True, index=index)

        # ! Layer Integrated Gradient
        elif reasoning_method == "lig":
            ref_token_id, sep_token_id, cls_token_id = tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id
            logger.info(
                f'line_level_localization_tp() - lig - ref_token_id: {ref_token_id}')
            logger.info(
                f'line_level_localization_tp() - lig - sep_token_id: {sep_token_id}')
            logger.info(
                f'line_level_localization_tp() - lig - cls_token_id: {cls_token_id}')

            ref_input_ids = create_ref_input_ids(
                input_ids, ref_token_id, sep_token_id, cls_token_id)
            logger.info(
                f'line_level_localization_tp() - lig - ref_token_id (modified): {ref_token_id}')

            # send data to device
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            ref_input_ids = ref_input_ids.to(args.device)

            logger.info(
                f'line_level_localization_tp() - lig - input_ids: {input_ids}')
            logger.info(
                f'line_level_localization_tp() - lig - labels: {labels}')
            logger.info(
                f'line_level_localization_tp() - lig - ref_input_ids: {ref_input_ids}')

            lig = LayerIntegratedGradients(
                lig_forward, model.encoder.roberta.embeddings)
            logger.info(f'line_level_localization_tp() - lig: {lig}')

            attributions, delta = lig.attribute(inputs=input_ids,
                                                baselines=ref_input_ids,
                                                internal_batch_size=32,
                                                return_convergence_delta=True)
            logger.info(
                f'line_level_localization_tp() - lig - attributions: {attributions}')
            logger.info(f'line_level_localization_tp() - lig - delta: {delta}')

            score = predict(input_ids)  # captum LIG
            logger.info(f'line_level_localization_tp() - lig - score: {score}')

            pred_idx = torch.argmax(score).cpu().numpy()
            logger.info(
                f'line_level_localization_tp() - lig - pred_idx: {pred_idx}')

            pred_prob = score[pred_idx]
            logger.info(
                f'line_level_localization_tp() - lig - pred_prob: {pred_prob}')

            attributions_sum = summarize_attributions(attributions)
            logger.info(
                f'line_level_localization_tp() - lig - attributions_sum: {attributions_sum}')

            attr_scores = attributions_sum.tolist()
            logger.info(
                f'line_level_localization_tp() - lig - attr_score: {attr_scores}')

            # each token should have one score
            assert len(all_tokens) == len(attr_scores)

            # store tokens and attr scores together in a list of tuple [(token, attr_score)]
            word_attr_scores = get_word_att_scores(
                all_tokens=all_tokens, att_scores=attr_scores)
            logger.info(
                f'line_level_localization_tp() - lig - word_attr_scores: {word_attr_scores}')

            # remove <s>, </s>, <unk>, <pad>
            word_attr_scores = clean_word_attr_scores(
                word_attr_scores=word_attr_scores)
            logger.info(
                f'line_level_localization_tp() - lig - word_attr_scores (modified): {word_attr_scores}')

            all_lines_score, flaw_line_indices = get_all_lines_score(
                word_attr_scores, verified_flaw_lines)
            logger.info(
                f'line_level_localization_tp() - lig - all_lines_score: {all_lines_score}')
            logger.info(
                f'line_level_localization_tp() - lig - flaw_line_indices: {flaw_line_indices}')

            # return if no flaw lines exist
            if len(flaw_line_indices) == 0:
                return "NA"

            total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, top_10_correct_idx, top_10_not_correct_idx \
                = \
                line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                      top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=True, index=index)  # interesting, there is index here!

        # ! Other mechanisms
        elif reasoning_method == "deeplift" or \
                reasoning_method == "deeplift_shap" or \
                reasoning_method == "gradient_shap" or \
                reasoning_method == "saliency":

            # send data to device
            input_ids = input_ids.to(args.device)
            input_embed = model.encoder.roberta.embeddings(
                input_ids).to(args.device)

            logger.info(
                f'line_level_localization_tp() - other - input_ids: {input_ids}')
            logger.info(
                f'line_level_localization_tp() - other - input_embed: {input_embed}')

            if reasoning_method == "deeplift":
                # baselines = torch.randn(1, 512, 768, requires_grad=True).to(args.device)
                baselines = torch.zeros(
                    1, 512, 768, requires_grad=True).to(args.device)
                reasoning_model = DeepLift(model)
            elif reasoning_method == "deeplift_shap":
                # baselines = torch.randn(16, 512, 768, requires_grad=True).to(args.device)
                baselines = torch.zeros(
                    16, 512, 768, requires_grad=True).to(args.device)
                reasoning_model = DeepLiftShap(model)
            elif reasoning_method == "gradient_shap":
                # baselines = torch.randn(16, 512, 768, requires_grad=True).to(args.device)
                baselines = torch.zeros(
                    16, 512, 768, requires_grad=True).to(args.device)
                reasoning_model = GradientShap(model)
            elif reasoning_method == "saliency":
                reasoning_model = Saliency(model)

            # attributions -> [1, 512, 768]
            if reasoning_method == "saliency":
                attributions = reasoning_model.attribute(input_embed, target=1)
            else:
                attributions = reasoning_model.attribute(
                    input_embed, baselines=baselines, target=1)

            logger.info(
                f'line_level_localization_tp() - other - baselines: {baselines}')
            logger.info(
                f'line_level_localization_tp() - other - reasoning_model: {reasoning_model}')

            attributions_sum = summarize_attributions(attributions)
            logger.info(
                f'line_level_localization_tp() - other - attributions: {attributions}')
            logger.info(
                f'line_level_localization_tp() - other - attributions_sum: {attributions_sum}')

            attr_scores = attributions_sum.tolist()
            logger.info(
                f'line_level_localization_tp() - other - attr_scores: {attr_scores}')

            # each token should have one score
            assert len(all_tokens) == len(attr_scores)

            # store tokens and attr scores together in a list of tuple [(token, attr_score)]
            word_attr_scores = get_word_att_scores(
                all_tokens=all_tokens, att_scores=attr_scores)
            logger.info(
                f'line_level_localization_tp() - other - word_attr_scores: {word_attr_scores}')

            # remove <s>, </s>, <unk>, <pad>
            word_attr_scores = clean_word_attr_scores(
                word_attr_scores=word_attr_scores)
            logger.info(
                f'line_level_localization_tp() - other - word_attr_scores (modified): {word_attr_scores}')

            all_lines_score, flaw_line_indices = get_all_lines_score(
                word_attr_scores, verified_flaw_lines)
            logger.info(
                f'line_level_localization_tp() - other - all_lines_scores: {all_lines_score}')
            logger.info(
                f'line_level_localization_tp() - other - flaw_line_indices: {flaw_line_indices}')

            # return if no flaw lines exist
            if len(flaw_line_indices) == 0:
                return "NA"

            total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, top_10_correct_idx, top_10_not_correct_idx \
                = \
                line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                      top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=True, index=index)

        # NOTE: I've also logged it in test()
        results = {"total_lines": total_lines,
                   "num_of_flaw_lines": num_of_flaw_lines,
                   "all_correctly_predicted_flaw_lines": all_correctly_predicted_flaw_lines,
                   "all_correctly_localized_function": all_correctly_localized_func,
                   "min_clean_lines_inspected": min_clean_lines_inspected,
                   "max_clean_lines_inspected": max_clean_lines_inspected,
                   "top_10_correct_idx": top_10_correct_idx,
                   "top_10_not_correct_idx": top_10_not_correct_idx}

        logger.info(f'line_level_localization_tp() - results: {results}')
        return results

    else:  # do_explaination == False
        if write_invalid_data:
            with open("../invalid_data/invalid_line_lev_data.txt", "a") as f:
                f.writelines("--- ALL TOKENS ---")
                f.writelines("\n")
                alltok = ''.join(all_tokens)
                alltok = alltok.split("Ċ")
                for tok in alltok:
                    f.writelines(tok)
                    f.writelines("\n")
                f.writelines("--- FLAW ---")
                f.writelines("\n")
                for i in range(len(flaw_tokens_encoded)):
                    f.writelines(''.join(flaw_tokens_encoded[i]))
                    f.writelines("\n")
                f.writelines("\n")
                f.writelines("\n")

    # if no flaw line exist in the encoded input
    return "NA"


def line_level_localization(flaw_lines: str, tokenizer, model, mini_batch, original_func: str, args,
                            top_k_loc: list, top_k_constant: list, reasoning_method: str, index: int):
    # Only used --do_sorting_by_line_scores is true

    # 2 function for captum LIG.
    def predict(input_ids):
        return model(input_ids=input_ids)[0]

    def lig_forward(input_ids):
        logits = model(input_ids=input_ids)[0]
        y_pred = 1  # for positive attribution, y_pred = 0 for negative attribution
        pred_prob = logits[y_pred].unsqueeze(-1)
        return pred_prob

    flaw_line_seperator = "/~/"
    (input_ids, labels) = mini_batch
    ids = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    original_lines = ''.join(all_tokens).split("Ċ")

    # flaw line verification
    # get flaw tokens ground truth
    flaw_lines = get_all_flaw_lines(
        flaw_lines=flaw_lines, flaw_line_seperator=flaw_line_seperator)
    flaw_tokens_encoded = encode_all_lines(
        all_lines=flaw_lines, tokenizer=tokenizer)
    verified_flaw_lines = []
    for i in range(len(flaw_tokens_encoded)):
        encoded_flaw = ''.join(flaw_tokens_encoded[i])
        encoded_all = ''.join(all_tokens)
        if encoded_flaw in encoded_all:
            verified_flaw_lines.append(flaw_tokens_encoded[i])

    if reasoning_method == "attention":
        # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
        input_ids = input_ids.to(args.device)
        model.eval()
        model.to(args.device)
        with torch.no_grad():
            prob, attentions = model(
                input_ids=input_ids, output_attentions=True)
        # take from tuple then take out mini-batch attention values
        attentions = attentions[0][0]
        attention = None
        # go into the layer
        for i in range(len(attentions)):
            layer_attention = attentions[i]
            # summerize the values of each token dot other tokens
            layer_attention = sum(layer_attention)
            if attention is None:
                attention = layer_attention
            else:
                attention += layer_attention
        # clean att score for <s> and </s>
        attention = clean_special_token_values(attention, padding=True)
        # attention should be 1D tensor with seq length representing each token's attention value
        word_att_scores = get_word_att_scores(
            all_tokens=all_tokens, att_scores=attention)
        all_lines_score, flaw_line_indices = get_all_lines_score(
            word_att_scores, verified_flaw_lines)
        all_lines_score_with_label = \
            line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                  top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=False)

    elif reasoning_method == "lig":

        ref_token_id, sep_token_id, cls_token_id = tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id
        ref_input_ids = create_ref_input_ids(
            input_ids, ref_token_id, sep_token_id, cls_token_id)

        # send data to device
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        ref_input_ids = ref_input_ids.to(args.device)

        lig = LayerIntegratedGradients(
            lig_forward, model.encoder.roberta.embeddings)

        attributions, delta = lig.attribute(inputs=input_ids,
                                            baselines=ref_input_ids,
                                            internal_batch_size=32,
                                            return_convergence_delta=True)
        score = predict(input_ids)
        pred_idx = torch.argmax(score).cpu().numpy()
        pred_prob = score[pred_idx]
        attributions_sum = summarize_attributions(attributions)
        attr_scores = attributions_sum.tolist()
        # each token should have one score
        assert len(all_tokens) == len(attr_scores)
        # store tokens and attr scores together in a list of tuple [(token, attr_score)]
        word_attr_scores = get_word_att_scores(
            all_tokens=all_tokens, att_scores=attr_scores)
        # remove <s>, </s>, <unk>, <pad>
        word_attr_scores = clean_word_attr_scores(
            word_attr_scores=word_attr_scores)
        all_lines_score, flaw_line_indices = get_all_lines_score(
            word_attr_scores, verified_flaw_lines)
        all_lines_score_with_label = \
            line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                  top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=False)
    elif reasoning_method == "deeplift" or \
            reasoning_method == "deeplift_shap" or \
            reasoning_method == "gradient_shap" or \
            reasoning_method == "saliency":
        # send data to device
        input_ids = input_ids.to(args.device)
        input_embed = model.encoder.roberta.embeddings(
            input_ids).to(args.device)
        if reasoning_method == "deeplift":
            # baselines = torch.randn(1, 512, 768, requires_grad=True).to(args.device)
            baselines = torch.zeros(
                1, 512, 768, requires_grad=True).to(args.device)
            reasoning_model = DeepLift(model)
        elif reasoning_method == "deeplift_shap":
            # baselines = torch.randn(16, 512, 768, requires_grad=True).to(args.device)
            baselines = torch.zeros(
                16, 512, 768, requires_grad=True).to(args.device)
            reasoning_model = DeepLiftShap(model)
        elif reasoning_method == "gradient_shap":
            # baselines = torch.randn(16, 512, 768, requires_grad=True).to(args.device)
            baselines = torch.zeros(
                16, 512, 768, requires_grad=True).to(args.device)
            reasoning_model = GradientShap(model)
        elif reasoning_method == "saliency":
            reasoning_model = Saliency(model)
        # attributions -> [1, 512, 768]
        if reasoning_method == "saliency":
            attributions = reasoning_model.attribute(input_embed, target=1)
        else:
            attributions = reasoning_model.attribute(
                input_embed, baselines=baselines, target=1)
        attributions_sum = summarize_attributions(attributions)
        attr_scores = attributions_sum.tolist()
        # each token should have one score
        assert len(all_tokens) == len(attr_scores)
        # store tokens and attr scores together in a list of tuple [(token, attr_score)]
        word_attr_scores = get_word_att_scores(
            all_tokens=all_tokens, att_scores=attr_scores)
        # remove <s>, </s>, <unk>, <pad>
        word_attr_scores = clean_word_attr_scores(
            word_attr_scores=word_attr_scores)
        all_lines_score, flaw_line_indices = get_all_lines_score(
            word_attr_scores, verified_flaw_lines)

        all_lines_score_with_label = \
            line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                  top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=False)
    return all_lines_score_with_label


def line_level_evaluation(all_lines_score: list, flaw_line_indices: list, top_k_loc: list, top_k_constant: list, true_positive_only: bool, index=None):

    if true_positive_only:
        # Generate line indices ranking based on attr values
        ranking = sorted(range(len(all_lines_score)),
                         key=lambda i: all_lines_score[i], reverse=True)
        logger.info(f'line_level_evaluation() - ranking: {ranking}')

        # output
        num_of_flaw_lines = len(flaw_line_indices)  # total flaw lines
        total_lines = len(all_lines_score)  # clean lines + flaw lines

        ### TopK% Recall ###
        all_correctly_predicted_flaw_lines = []

        ### IFA ###
        ifa = True
        all_clean_lines_inspected = []

        logger.info(
            f'line_level_evaluation() - *********** BEGIN TOP_K LOOP **********')
        for top_k in top_k_loc:  # top_k = 0
            correctly_predicted_flaw_lines = 0

            for indice in flaw_line_indices:  # indice = 1
                # if within top-k
                # k = int(17 * 0.0) = int(0) = 0
                k = int(len(all_lines_score) * top_k)

                # if detecting any flaw lines
                # ranking[0:0] = [], this will always be False
                if indice in ranking[: k]:
                    correctly_predicted_flaw_lines += 1
                if ifa:
                    # calculate Initial False Alarm
                    # IFA counts how many clean lines are inspected until the first vulnerable line is found when inspecting the lines ranked by the approaches.
                    # ! REMEMBER: "when ranked by the approaches, not when it's in its orginal order"
                    flaw_line_idx_in_ranking = ranking.index(indice)
                    logger.info(
                        f'line_level_evaluation() - flaw_line_idx_in_ranking: {flaw_line_idx_in_ranking}')

                    # e.g. flaw_line_idx_in_ranking = 3 will include 1 vulnerable line and 3 clean lines
                    all_clean_lines_inspected.append(flaw_line_idx_in_ranking)

                    logger.info(
                        f'line_level_evaluation() - all_clean_lines_inspected: {all_clean_lines_inspected}')

            # for IFA
            min_clean_lines_inspected = min(all_clean_lines_inspected)
            # for All Effort
            max_clean_lines_inspected = max(all_clean_lines_inspected)
            # only do IFA and All Effort once
            ifa = False
            # append result for one top-k value
            all_correctly_predicted_flaw_lines.append(
                correctly_predicted_flaw_lines)

        ### Top10 Accuracy ###
        all_correctly_localized_func = []
        top_10_correct_idx = []
        top_10_not_correct_idx = []
        correctly_located = False
        for k in top_k_constant:
            for indice in flaw_line_indices:
                # if detecting any flaw lines
                if indice in ranking[:k]:  # the first 10 lines with the highest score
                    """
                    # extract example for the paper
                    if index == 2797:
                        logger.info("2797")
                        logger.info("ground truth flaw line index: ", indice)
                        logger.info("ranked line")
                        logger.info(ranking)
                        logger.info("original score")
                        logger.info(all_lines_score)
                    """
                    # append result for one top-k value
                    all_correctly_localized_func.append(1)
                    correctly_located = True
                else:
                    all_correctly_localized_func.append(0)
            if correctly_located:
                top_10_correct_idx.append(index)
            else:
                top_10_not_correct_idx.append(index)
        return total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, \
            top_10_correct_idx, top_10_not_correct_idx

    else:  # true_positive_only == False

        # all_lines_score_with_label: [[line score, line level label], [line score, line level label], ...]
        all_lines_score_with_label = []
        for i in range(len(all_lines_score)):
            if i in flaw_line_indices:
                all_lines_score_with_label.append([all_lines_score[i], 1])
            else:
                all_lines_score_with_label.append([all_lines_score[i], 0])
        return all_lines_score_with_label


def clean_special_token_values(all_values, padding=False):
    # special token in the beginning of the seq
    all_values[0] = 0
    if padding:
        # get the last non-zero value which represents the att score for </s> token
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        # special token in the end of the seq
        all_values[-1] = 0
    return all_values


def clean_shap_tokens(all_tokens):
    for i in range(len(all_tokens)):
        all_tokens[i] = all_tokens[i].replace('Ġ', '')
    return all_tokens


def get_all_lines_score(word_att_scores: list, verified_flaw_lines: list):
    verified_flaw_lines = [''.join(l) for l in verified_flaw_lines]

    logger.info(
        f'get_all_lines_score(): verified_flaw_lines: {verified_flaw_lines}')

    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]

    # to return
    all_lines_score = []
    flaw_line_indices = []

    score_sum = 0
    line_idx = 0
    line = ""
    for i in range(len(word_att_scores)):
        # summerize if meet line separator or the last token

        token = word_att_scores[i][0]  # or 'token', whatever you named it
        token_score = word_att_scores[i][1]  # whatever you named it

        if ((token in separator) or (i == (len(word_att_scores) - 1))) and score_sum != 0:
            score_sum += token_score
            logger.info(f'get_all_lines_score(): score_sum: {score_sum}')

            all_lines_score.append(score_sum)
            logger.info(
                f'get_all_lines_score(): all_lines_score: {all_lines_score}')

            # Check in dataset if current line is flaw line
            is_flaw_line = False
            for l in verified_flaw_lines:
                if l == line:
                    is_flaw_line = True
            if is_flaw_line:
                flaw_line_indices.append(line_idx)

            # Reset
            line = ""
            score_sum = 0
            line_idx += 1

        # else accumulate score
        elif token not in separator:
            line += token
            score_sum += token_score

    return all_lines_score, flaw_line_indices


def get_all_flaw_lines(flaw_lines: str, flaw_line_seperator: str) -> list:
    if isinstance(flaw_lines, str):
        flaw_lines = flaw_lines.strip(flaw_line_seperator)
        flaw_lines = flaw_lines.split(flaw_line_seperator)
        flaw_lines = [line.strip() for line in flaw_lines]
    else:
        flaw_lines = []
    return flaw_lines


def encode_all_lines(all_lines: list, tokenizer) -> list:
    encoded = []
    for line in all_lines:
        encoded.append(encode_one_line(line=line, tokenizer=tokenizer))
    return encoded


def get_word_att_scores(all_tokens: list, att_scores: list) -> list:
    word_att_scores = []
    for i in range(len(all_tokens)):
        token, att_score = all_tokens[i], att_scores[i]
        word_att_scores.append([token, att_score])
    return word_att_scores


def clean_word_attr_scores(word_attr_scores: list) -> list:
    to_be_cleaned = ['<s>', '</s>', '<unk>', '<pad>']
    cleaned = []
    for word_attr_score in word_attr_scores:
        if word_attr_score[0] not in to_be_cleaned:
            cleaned.append(word_attr_score)
    return cleaned


def encode_one_line(line, tokenizer):
    # add "@ " at the beginning to ensure the encoding consistency, i.e., previous -> previous, not previous > pre + vious
    code_tokens = tokenizer.tokenize("@ " + line)
    logger.info(f'encode_one_line() - code_tokens: {code_tokens}')

    return [token.replace("Ġ", "") for token in code_tokens if token != "@"]

# new function that I wrote
# basically, it will resembles test(), but much more simplify
# there will be code duplicates, but this is the best method to study the workflow of LineVul as well


def use(input_lines, input_model_name, args, model, tokenizer, use_dataset, best_threshold=0.5):

    # Build DataLoader, iterate over the elements of a Dataset in see order, without shuffling
    use_sampler = SequentialSampler(use_dataset)
    logger.info(f'use() - use_sampler: {use_sampler}')

    # Create batches of data for use
    use_dataloader = DataLoader(
        use_dataset, sampler=use_sampler, batch_size=args.use_batch_size, num_workers=0)
    logger.info(f'use() - use_dataloader: {use_dataloader}')

    # Multi-GPU
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  # you need model here

    # ! DETECTION
    logger.info(f'use() - ************** DETECT **************')
    logger.info(f'use() - No. of functions = {len(use_dataset)}')
    logger.info(f'use() - Batch size = {args.use_batch_size}')

    # Change mode to 'evaluation'
    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0

    logits = []

    for batch in use_dataloader:
        logger.info('use() - ******* START BATCH ITERATION *******')

        # Move each tensor in the batch to the device specified by `args.device`
        (input_ids, labels) = [x.to(args.device) for x in batch]
        logger.info(f'use() - inputs_ids: {input_ids}')
        logger.info(f'use() - labels: {labels}')

        with torch.no_grad():
            lm_loss, logit = model(input_ids=input_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())

            logger.info(f'lm_loss: {lm_loss}')
            logger.info(f'logit: {logit}')
            logger.info(f'eval_loss: {eval_loss}')
            logger.info(f'logits: {logits}')

        nb_eval_steps += 1

        logger.info('use() - ******* END BATCH ITERATION *******')

    # List -> ndarray
    logits = np.concatenate(logits, 0)
    print(f'use() - logits: {logits}')
    tensor = torch.tensor(logits[0])
    probs = F.softmax(tensor, dim=0)
    print(f'use() - probs: {probs}')

    y_preds = logits[:, 1] > best_threshold
    logger.info(f'use() - y_preds: {y_preds}')

    if args.function_level:

        is_vulnerable = y_preds[0]
        standard_output = json.dumps({
            "modelName": input_model_name,
            "isVulnerable": True if is_vulnerable else False
        })

        print(standard_output)
        return

    # ! END OF DETECTION

    if args.line_level:

        # ndarray -> List
        logits = [l[1] for l in logits]
        logger.info(f'use() - logits: {logits}')

        ##################################
        # generate_result_df() for use() method
        df = pd.read_csv(args.use_data_file)
        all_num_lines = []
        all_processed_func = df["processed_func"].tolist()

        for func in all_processed_func:
            func_num_lines = get_num_lines(func)
            all_num_lines.append(func_num_lines)

            logger.info(f'use() - func: {func}')
            logger.info(f'use() - func_num_lines: {func_num_lines}')

        logger.info(f'test() - all_num_lines: {all_num_lines}')

        # Check if y_preds and logits has the same length
        assert len(logits) == len(y_preds)

        indices = list(range(len(logits)))
        #################################

        # ! LOCALIZATION using 1 reasoning method
        # Calculate score
        data_loader = DataLoader(
            use_dataset, sampler=use_sampler, batch_size=1, num_workers=0)
        progress_bar = tqdm(data_loader, total=len(
            data_loader), desc=args.reasoning_method)

        rel_line_num_on_editor = 0
        all_positive_score_label = []
        all_negative_score_label = []

        for mini_batch in progress_bar:
            logger.info('use() - ******* START MINI BATCH ITERATION *******')
            all_lines_score, ranking, all_lines_content = line_level_localization_custom(
                tokenizer=tokenizer,
                model=model,
                mini_batch=mini_batch,
                index=rel_line_num_on_editor,
                original_func=all_processed_func[rel_line_num_on_editor],
                args=args,
                reasoning_method='attention'
            )

            logger.info(
                f'use() - all_lines_score: {all_lines_score}')

            logger.info(
                f'use() - ranking: {ranking}')

            logger.info(
                f'use() - all_lines_content: {all_lines_content}'
            )

            # if predicted as vulnerable
            if logits[rel_line_num_on_editor] > best_threshold:
                all_positive_score_label.append(all_lines_score)

            # else predicted as non-vulnerable
            else:
                all_negative_score_label.append(all_lines_score)

            logger.info('use() - ******* END MINI BATCH ITERATION *******')

        logger.info(
            f'use() - all_positive_score_label: {all_positive_score_label}')
        logger.info(
            f'use() - all_negative_score_label: {all_negative_score_label}')

    output_lines = []
    if (len(all_negative_score_label) > 0 and len(all_positive_score_label) == 0):
        for rel_line_num_on_editor, content in enumerate(input_lines):
            output_lines.append({
                "content": content,
                "num": rel_line_num_on_editor,
                "score": 0,
                "isVulnerable": False
            })

        output_dict = {
            "modelName": 'linevul',
            "lines": output_lines
        }
        output_json = json.dumps(output_dict)
        print(output_json)
        return

    if (len(all_positive_score_label) > 0 and len(all_negative_score_label) == 0):

        lines = []
        # only the first 'processed_func', if our input is more than 1 func, change this
        for num, score in enumerate(all_positive_score_label[0]):
            lines.append({
                "content": all_lines_content[ranking[num]],
                "num": ranking[num],
                "score": score.item(),
                "isVulnerable": True if (num < 10) else False
            })  # Top 10 lines with the highest score are considered vul

        # The lines sorted to extract top 10, now resort the line in ascending order
        lines.sort(key=lambda line: line['num'])

        # Compare with unproccessed lines to retain line's order at VSCode editor
        for rel_line_num_on_editor, content in enumerate(input_lines):

            if content == '':  # consecutive new lines
                output_lines.append({
                    "num": rel_line_num_on_editor,
                    "content": '',
                    "score": 0,
                    "isVulnerable": False
                })
                continue

            is_founded_similar = False
            for i, line in enumerate(lines):

                # NOTE: remember to remove all space character to achieve maximum similarity
                if (are_strings_similar(content.replace(' ', ''), line['content'].replace(' ', ''))):
                    output_lines.append({
                        "num": rel_line_num_on_editor,
                        "content": content,
                        "score": line['score'],
                        "isVulnerable": line['isVulnerable']
                    })
                    lines.pop(i)
                    is_founded_similar = True
                    break

            # not all lines are predicted
            if (not is_founded_similar):
                output_lines.append({
                    "num": rel_line_num_on_editor,
                    "content": content,
                    "score": 0,
                    "isVulnerable": False
                })

        standard_output = json.dumps({
            "modelName": input_model_name,
            "lines": output_lines
        })
        print(standard_output)
        return

    # both arrs are empty
    print('Something is definately wrong.')
    return


def line_level_localization_custom(tokenizer, model, mini_batch, index: int, original_func: str, args, reasoning_method: str):

    top_k_locs = args.top_k_locs
    top_k_constant = args.top_k_constant  # it might be an array in the future
    device = args.device

    (input_ids, labels) = mini_batch
    logger.info(f'line_level_localization_custom() - input_ids: {input_ids}')
    logger.info(f'line_level_localization_custom() - labels: {labels}')

    ids = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    original_lines = ''.join(all_tokens).split("Ċ")

    # debug
    # the way IDE and model seeing lines is very different
    print('Original lines: ', original_lines, file=sys.stderr)
    print('Length: ', len(original_lines), file=sys.stderr)

    logger.info(f'line_level_localization_custom() - ids: {ids}')
    logger.info(f'line_level_localization_custom() - all_tokens: {all_tokens}')
    logger.info(
        f'line_level_localization_custom() - original_lines: {original_lines}')

    if reasoning_method == 'attention':
        input_ids = input_ids.to(device)
        model.eval()  # TODO: already use() in main(), idk why this is still here
        # TODO: already done in main(), idk why this is still here
        model.to(device)

        with torch.no_grad():
            # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
            prob, attentions = model(
                input_ids=input_ids, output_attentions=True)

        # take from tuple, then take out mini-batch attention values
        attentions = attentions[0][0]
        attention = None

        # go into the layer
        for i in range(len(attentions)):
            # summerize the values of each token dot other tokens
            layer_attention = sum(attentions[i])

            if attention is None:  # init
                attention = layer_attention
            else:
                attention += layer_attention

        # clean attribute score for <s> and </s>, they are biased
        attention = clean_special_token_values(attention, padding=True)

        # attention should be 1D Tensor with seq length representing each token's attention value
        word_attribute_scores = get_word_attribute_scores_custom(
            all_tokens=all_tokens, att_scores=attention)

        all_lines_score, ranking, all_lines_content = get_all_lines_sorted_score_and_ranking_custom(
            word_attribute_scores)

        # already log in delegate function, don't have to log now
        return all_lines_score, ranking, all_lines_content


def get_word_attribute_scores_custom(all_tokens: list, att_scores: list) -> list:
    word_attribute_scores = []
    for i in range(len(all_tokens)):
        token, attribute_score = all_tokens[i], att_scores[i]
        word_attribute_scores.append([token, attribute_score])
    return word_attribute_scores


def get_all_lines_sorted_score_and_ranking_custom(word_attribute_scores: list):

    # word_attribute_scores -> [[token, att_value], [token, att_value], ...]
    separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]

    # init output
    all_lines_score = []
    all_lines_content = []

    score_sum = 0
    line_content = ""
    line_index = 0

    for i in range(len(word_attribute_scores)):

        token = word_attribute_scores[i][0]
        token_score = word_attribute_scores[i][1]

        # summerize if encounter line separator or last token
        if ((token in separator) or (i == (len(word_attribute_scores) - 1))) and score_sum != 0:
            logger.info(
                f'get_all_lines_sorted_score_and_ranking_custom() - line_content: {line_content}')

            score_sum += token_score
            logger.info(
                f'get_all_lines_sorted_score_and_ranking_custom() - score_sum: {score_sum}')

            all_lines_score.append(score_sum)
            logger.info(
                f'get_all_lines_sorted_score_and_ranking_custom() - all_lines_score: {all_lines_score}')

            # save line content
            all_lines_content.append(line_content)

            # reset
            score_sum = 0
            line_content = ""
            line_index += 1

        # else accumulate score
        elif token not in separator:
            line_content += token
            score_sum += token_score

    # Generate line indices ranking based on attribute values
    ranking = sorted(range(len(all_lines_score)),
                     key=lambda i: all_lines_score[i], reverse=True)

    # Sort all_lines_score by score, descending order
    all_lines_score.sort(reverse=True)

    logger.info(
        f'get_all_lines_sorted_score_and_ranking_custom() - ranking: {ranking}')
    logger.info(
        f'get_all_lines_sorted_score_and_ranking_custom() - all_lines_score: {all_lines_score}')
    logger.info(
        f'get_all_lines_sorted_score_and_ranking_custom() - all_lines_content: {all_lines_content}')

    return all_lines_score, ranking, all_lines_content


def main():

    parser = argparse.ArgumentParser()

    # new parameters
    parser.add_argument("--do_use", action='store_true',
                        help="Whether to use the model.")
    parser.add_argument("--function_level", action='store_true',
                        help="Whether to run model to only detect vulnerability at function-level")
    parser.add_argument("--line_level", action='store_true',
                        help="Whether to run model to only detect vulnerability at line-level")
    parser.add_argument("--input_json", default=None, type=str, required=True,
                        help='The JSON string that contains the function content.')

    # old parameters
    parser.add_argument("--use_batch_size", default=512, type=int,
                        help="Batch size per GPU/CPU for use.")
    parser.add_argument("--top_k_locs", type=float, nargs='*',
                        default=[0, 0.1, 0.2, 0.3, 0.4,
                                 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Multiple Top-K Accuracies, range from least IFA to highest IFA')
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=512, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    # RQ2
    parser.add_argument("--effort_at_top_k", default=0.2, type=float,
                        help="Effort@TopK%Recall: effort at catching top k percent of vulnerable lines")
    parser.add_argument("--top_k_recall_by_lines", default=0.01, type=float,
                        help="Recall@TopK percent, sorted by line scores")
    parser.add_argument("--top_k_recall_by_pred_prob", default=0.2, type=float,
                        help="Recall@TopK percent, sorted by prediction probabilities")

    parser.add_argument("--do_sorting_by_line_scores", default=False, action='store_true',
                        help="Whether to do sorting by line scores.")
    parser.add_argument("--do_sorting_by_pred_prob", default=False, action='store_true',
                        help="Whether to do sorting by prediction probabilities.")
    # RQ3 - line-level evaluation
    # it might be an array in the future
    parser.add_argument('--top_k_constant', type=int, default=10,
                        help="Top-K Accuracy constant")
    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    # raw predictions
    parser.add_argument("--write_raw_preds", default=False, action='store_true',
                        help="Whether to write raw predictions on test data.")
    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    args = parser.parse_args()

    ######################################################################################

    # CUDA or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
        datefmt='%Y/%m/%d %H:%M:%S', level=logging.WARNING)  # NOTE: change this back to display logger.info()

    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu,)

    # Initialize seed
    set_seed(args)

    # Download the configuration from Microsoft
    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)

    config.num_labels = 1  # only 1 label

    # A hyperparameter for AI model
    # Also the number of multi-head attention layers of the model
    # How many parallel attention mechanisms operate with each attention layer
    # Increase?
    # Pros: Focus on more diverse aspects of the input => capture more nuanced (pattern) relationships => improve performance.
    # Cons: Computational complexity, high memory requirements
    config.num_attention_heads = args.num_attention_heads  # 12

    if args.use_word_level_tokenizer:
        logger.info('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file(
            './word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        # Use Roberta Tokenizer from Microsoft
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)
    else:
        # Use Roberta Sequence Classification from Microsoft
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config, ignore_mismatched_sizes=True)

    # LineVul Model:
    # - encoder: RobertaForSequenceClassification,
    # - tokenizer: Roberta Tokenizer
    # - classifier: RobertaClassificationHead
    model = Model(model, config, tokenizer, args)

    logger.info("Training/evaluation/use parameters %s",
                args)  # display all parameters

    # ==================================================== #
    # Training                                             #
    # ==================================================== #

    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)

    # ==================================================== #
    # Evaluation                                           #
    # ==================================================== #

    path_to_model = os.path.join(
        args.output_dir, 'checkpoint-best-f1', args.model_name)
    logger.info(f'main() - path_to_model: {path_to_model}')

    # Load pre-trained model into device
    # keys = param name, values = tensors
    state_dict = torch.load(
        f=path_to_model, map_location=args.device)
    logger.info(f'main() - state_dict: {state_dict}')

    # Update LineVul model's param with the param stored in state dictionary
    model.load_state_dict(state_dict=state_dict, strict=False)
    model.to(args.device)  # add device to model

    if args.do_test:
        # Load test dataset
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        logger.info(f'main() - test_dataset: {test_dataset}')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)

    # ==================================================== #
    # Real Use                                             #
    # ==================================================== #
    if args.do_use:
        standard_input = json.loads(args.input_json)

        # debug
        print(f'Current working directory: {os.getcwd()}')
        print(f'Input: {standard_input}')
        print(f'Name: {standard_input["modelName"]}')
        print(f'Lines: {standard_input["lines"]}')

        input_model_name = standard_input["modelName"]
        # test if remove space and tab is better
        input_lines = [line.lstrip() for line in standard_input['lines']]
        input_func = "\n".join(input_lines)

        use_dataset_dict = {
            'processed_func': [input_func],
            'target': [0]  # 0 or 1 doesn't matter here
        }
        df = pd.DataFrame(use_dataset_dict)
        df.to_csv('temp.csv', index=False)
        args.use_data_file = 'temp.csv'

        use_dataset = TextDataset(tokenizer, args, file_type='use')
        use(input_lines, input_model_name, args, model,
            tokenizer, use_dataset, best_threshold=0.5)

    return ""


if __name__ == "__main__":
    main()
