import shutil

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn

from ace2005_module.data_load import ACE2005Dataset, all_triggers, all_entities, all_postags, all_arguments, \
    idx2trigger, idx2argument
from ace2005_module.model import Net
from ace2005_module.utils import find_triggers, calc_metric
from mytransformers import AdamW, WEIGHTS_NAME, get_constant_schedule_with_warmup
import csv
import random
import numpy as np
import os
import copy
import logging
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict
from utils_mybitfit import *
from settings_mybitfit import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS
from settings_mybitfit import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, \
    CONFIG_NAME
from scheduler import AnnealingLR
import tqdm
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def load_old_adapter(model, newname, oldname):
    state_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for i in state_dict:
        if oldname in i:
            new_i = i.replace(oldname, newname)
            new_state_dict[new_i] = state_dict[i].clone().detach()
    m, n = model.load_state_dict(new_state_dict, strict=False)
    logger.info("Load old adapter weight to new adapter weight, Unexpected: {}".format(n))


# Load pretrained adapters, not used in this paper
def load_pre_adapter(model, newname):
    pre_state_dict = torch.load(os.path.join('./PModel/model-pretrain'))
    new_state_dict = OrderedDict()
    for i in pre_state_dict:
        if "pretrain" in i:
            new_i = i.replace("pretrain", newname)
            new_state_dict[new_i] = pre_state_dict[i].clone().detach()
            logger.info("Load from {} to {}".format(i, new_i))
    m, n = model.load_state_dict(new_state_dict, strict=False)
    logger.info("Load old adapter weight to new adapter weight, Unexpected: {}".format(n))


# Calculate the entropy for weight coefficient
def cal_entropy_loss(ita):
    #ita = torch.stack(ita)
    ita = ita / args.select_temp
    dis = torch.nn.functional.softmax(ita, dim=1)
    #entropy_loss = args.last_dim_coe * (dis[-1][0] - 0.0) ** 2
    log_dis = torch.log(dis)
    entropy_loss = -torch.sum(dis * log_dis,dim=1)
    return entropy_loss.sum()



def freeze_for_mix(model):
    for name, param in model.named_parameters():
        if "ita" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def learnable_para_calculate(model, note, printname=False):
    learn_sum = 0
    else_sum = 0
    logger.info("Para requries gradient...")
    param_opt = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_opt.append((name, param))
            if printname:
                logger.info(name)
            learn_sum += param.nelement()
        else:
            else_sum += param.nelement()
            # """
            if "ita" in name:
                param_opt.append((name, param))
            # """
    logger.info(note + " Number of learned parameter: %.2fM" % (learn_sum / 1e6))
    logger.info(note + " Number of else parameter: %.2fM" % (else_sum / 1e6))
    logger.info(note + " Ratio: {}".format(1.0 * (learn_sum + else_sum) / else_sum))
    return param_opt


def print_para(model):
    logger.info("Print para")
    printted = [False, False, False, False, False]
    for name, param in model.named_parameters():
        for i in range(5):
            if "adapters." + str(i) in name and not printted[i]:
                logger.info(name)
                logger.info(param)
                printted[i] = True


def swap_name(org_name, seq_distil, ref1):
    # swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1)
    if not seq_distil and not ref1:
        return org_name
    if seq_distil:
        return org_name.replace("train", "distil")
    if ref1:
        return org_name.replace("train", "ref1")


def old_validation(model, valid_dataloader, train_loss_fct):
    cum_loss = 0.0
    cum_qa_loss = 0.0
    cum_lm_loss = 0.0
    cur_n_inputs = 0
    with torch.no_grad():
        model.eval()
        for (_, _, cqa, _, Y, gen_X, gen_Y, task_id, idx) in valid_dataloader:
            torch.cuda.empty_cache()
            n_inputs = cqa[0].shape[0]
            model.config.batch_task_id = task_id[0][0].item()
            qa_loss, lm_loss = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(),
                                          train_loss_fct)
            cum_loss += (qa_loss + lm_loss) * n_inputs
            cum_qa_loss += qa_loss * n_inputs
            cum_lm_loss += lm_loss * n_inputs
            cur_n_inputs += n_inputs
    return cum_loss / cur_n_inputs, cum_qa_loss / cur_n_inputs, cum_lm_loss / cur_n_inputs


def validation(model, iterator):
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    cum_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm, task_id = batch

            trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.predict_triggers(
                tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, adjm=adjm)

            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d = model.predict_arguments(
                    argument_hidden, argument_keys, arguments_2d, adjm)
                arguments_hat_all.extend(argument_hat_2d)
                # if i == 0:

                #     print("=====sanity check for triggers======")
                #     print('triggers_y_2d[0]:', triggers_y_2d[0])
                #     print("trigger_hat_2d[0]:", trigger_hat_2d[0])
                #     print("=======================")

                #     print("=====sanity check for arguments======")
                #     print('arguments_y_2d[0]:', arguments_y_2d[0])
                #     print('argument_hat_1d[0]:', argument_hat_1d[0])
                #     print("arguments_2d[0]:", arguments_2d)
                #     print("argument_hat_2d[0]:", argument_hat_2d)
                #     print("=======================")
                cum_loss += trigger_loss.item() + args.LOSS_alpha * argument_loss.item()
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)
                cum_loss += trigger_loss.item()

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    '''with open('temp', 'w', encoding="utf-8") as fout:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_pred.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            for w, t, t_h in zip(words[1:-1], triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))

            arg_write = arguments['events']
            for arg_key in arg_write:
                arg = arg_write[arg_key]# list,eg: [(0, 5, 25), (8, 19, 17), (20, 21, 29)]
                for ii,tup in enumerate(arg):
                    arg[ii] = (tup[0],tup[1],idx2argument[tup[2]])# 将id 转为 str
                arg_write[arg_key] = arg

            arghat_write =arguments_hat['events']
            for arg_key in arghat_write:
                arg = arghat_write[arg_key]# list,eg: [(0, 5, 25), (8, 19, 17), (20, 21, 29)]
                for ii,tup in enumerate(arg):
                    arg[ii] = (tup[0],tup[1],idx2argument[tup[2]])# 将id 转为 str
                arghat_write[arg_key] = arg

            fout.write('#真实值#\t{}\n'.format(arg_write))
            fout.write('#预测值#\t{}\n'.format(arghat_write))
            fout.write("\n")'''

    # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))

    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))

    print('[argument classification]')
    argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))

    print('[trigger identification]')
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    print('[argument identification]')
    arguments_true = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_true]
    arguments_pred = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_pred]
    argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

    metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)
    metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r, argument_f1)
    metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_, trigger_f1_)
    metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_,
                                                                                  argument_f1_)
    # final = fname + ".trigger-F%.2f argument-F%.2f" % (trigger_f1, argument_f1)
    '''with open(final, 'w', encoding="utf-8") as fout:
        result = open("temp", "r", encoding="utf-8").read()
        fout.write("{}\n".format(result))
        fout.write(metric)'''
    # os.remove("temp")
    # return metric, trigger_f1, argument_f1
    return cum_loss / len(iterator)


# Clear model gradient
def clear(model):
    old = model
    model = copy.deepcopy(old)
    del old
    torch.cuda.empty_cache()
    return model


def load_model(model_dir, return_adapter_list=False):
    from mytransformers import BertModel, BertConfig

    #model_config = BertConfig.from_json_file(os.path.join(model_dir, "config.json"))
    model_config = BertConfig.from_json_file("./bert-large-uncased/config.json")
    model = BertModel(model_config)
    model.resize_token_embeddings(50260 + len(args.tasks))

    adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"), allow_pickle=True)
    model.add_adapter_by_list(adapter_list, config=args.adapt_type)
    state_dict = torch.load(os.path.join(model_dir, "model-finish"), map_location='cuda:0')
    net = Net(trigger_size=len(all_triggers), PreModel=model, entity_size=len(all_entities),
              all_postags=len(all_postags),
              argument_size=len(all_arguments), device=args.device_ids[0], idx2trigger=idx2trigger)
    m, n = net.load_state_dict(state_dict, strict=False)
    logger.info("Missing : {}, Unexpected: {}".format(m, n))
    net.cuda()

    if return_adapter_list:
        return net, adapter_list
    else:
        return net


# Decision stage
def Mix(task_ids, model):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to Mix { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)

    prev_tasks = [args.tasks[task_ids[0] - 1]]
    prev_model_dir = get_model_dir(prev_tasks)
    model = load_model(prev_model_dir)


    model.PreModel.config.forward_mode = 'Mix'
    model.PreModel.config.testing = False
    model.PreModel.config.mix_ini = args.mix_ini
    names_to_train = model.PreModel.add_adapter(str(task_ids[0]), config=args.adapt_type)
    if args.pretrain_adapter:
        load_pre_adapter(model, str(task_ids[0]))
    model.PreModel.train_adapter(names_to_train)
    model.cuda()

    if args.clear_model:
        model = clear(model)

    param_opt = learnable_para_calculate(model, "whole", True)



    if not args.fp32:  # again because resize_token_embeddings makes embedding layer fp32
        model = FP16_Module(model)

    logger.warning("Adapter Mix test, not using extra data now...")
    train_qadata = ACE2005Dataset('./ace2005/train.json', task_ids[0])
    valid_qadata = ACE2005Dataset('./ace2005/dev.json', task_ids[0])

    # max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
    max_train_batch_size = args.z_max_batch_size

    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    valid_dataloader = create_dataloader(valid_qadata, "test")

    n_train_epochs = args.whole_mix_step
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    if args.whole_optim:
        param_optimizer = list(model.named_parameters())
    else:
        param_optimizer = param_opt

    # logger.info(param_optimizer)
    no_decay = ['bias', 'ln_1', 'ln_2', 'ln_f']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    logger.info("USE ARGS.ADAM_EPSILON NOW.....")
    # logger.info("USE ARGS.ADAM_EPSILON NOW.....")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.z_train_lrs[task_ids[0]], weight_decay=args.l2)
    logger.info("Start to use constant scheduler!")
    scheduler = get_constant_schedule_with_warmup(optimizer, args.z_warmup_step)

    ita = None
    tot_n_steps = 0

    mix_flag = 0
    from early_stop import EarlyStopping
    tbx_title = 'mix_' + str(task_ids[0]) + '/'
    early_stopping = EarlyStopping(patience=10, verbose=True, trace_func=lambda x: logger.info(x))
    for ep in range(n_train_epochs):
        logger.info("Epoch {}".format(ep))
        model.train()
        print_para(model)
        words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
        triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
        cum_loss = 0
        # learnable_para_calculate(model, "whole")
        for n_steps, batch in tqdm.tqdm(enumerate(train_dataloader)):
            tot_n_steps += 1
            model.zero_grad()
            optimizer.zero_grad()
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm, task_id = batch

            model.PreModel.config.batch_task_id = task_id[0]
            trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.predict_triggers(
                tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, adjm=adjm)
            if len(argument_keys) > 0:
                argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d = model.predict_arguments(
                    argument_hidden, argument_keys, arguments_2d, adjm)
                # argument_loss = criterion(argument_logits, arguments_y_1d)
                loss = trigger_loss + args.LOSS_alpha * argument_loss
                # if i == 0:

                #     print("=====sanity check for triggers======")
                #     print('triggers_y_2d[0]:', triggers_y_2d[0])
                #     print("trigger_hat_2d[0]:", trigger_hat_2d[0])

                #     print("=======================")

                #     print("=====sanity check for arguments======")
                #     print('arguments_y_2d[0]:', arguments_y_2d[0])
                #     print('argument_hat_1d[0]:', argument_hat_1d[0])
                #     print("arguments_2d[0]:", arguments_2d)
                #     print("argument_hat_2d[0]:", argument_hat_2d)
                #     print("=======================")

            else:
                loss = trigger_loss
            # normalized mixed score? not used
            if args.mix_loss_norm and model.PreModel.config.forward_mode == 'Mix':
                loss /= loss.item()
                loss *= args.mix_loss_coe

            ita = model.get_ita()
            en_loss = torch.tensor(0.)
            if task_ids[0] > 0 and model.PreModel.config.forward_mode == 'Mix':
                en_loss = cal_entropy_loss(ita)
                loss += en_loss * args.entropy_coe

            cum_loss += loss.item()
            if args.tbx:
                writer.add_scalar(tbx_title+'train',loss.item(),tot_n_steps)
            loss.backward()

            optimizer.step(None)
            scheduler.step()
            detached_ita = ita.detach().cpu()
            #### 训练精度评估 ####
            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

            for ii, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(
                    zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
                triggers_hat = triggers_hat[:len(words)]
                triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

                # [(ith sentence, t_start, t_end, t_type_str)]
                triggers_true.extend([(ii, *item) for item in find_triggers(triggers)])
                triggers_pred.extend([(ii, *item) for item in find_triggers(triggers_hat)])

                # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
                for trigger in arguments['events']:
                    t_start, t_end, t_type_str = trigger
                    for argument in arguments['events'][trigger]:
                        a_start, a_end, a_type_idx = argument
                        arguments_true.append((ii, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

                for trigger in arguments_hat['events']:
                    t_start, t_end, t_type_str = trigger
                    for argument in arguments_hat['events'][trigger]:
                        a_start, a_end, a_type_idx = argument
                        arguments_pred.append((ii, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            if args.constant_sch or task_ids[0] > 0:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = scheduler.get_last_lr()

            if (n_steps) % args.logging_steps == 0:  # monitoring
                trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
                argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
                ## 100step 清零
                words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
                triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
                #########################
                if len(argument_keys) > 0:
                    logger.info(
                        "[Events Detected]step: {}, loss: {:.3f}, trigger_loss:{:.3f}, argument_loss:{:.3f}".format(
                            n_steps,
                            loss,
                            trigger_loss.item(),
                            argument_loss.item()) +
                        '[trigger] P={:.3f}  R={:.3f}  F1={:.3f}'.format(trigger_p, trigger_r, trigger_f1) +
                        '[argument] P={:.3f}  R={:.3f}  F1={:.3f}'.format(argument_p, argument_r, argument_f1)
                    )
                    cum_loss = 0
                else:
                    logger.info("[No Events Detected]step: {}, loss: {:.3f} ".format(n_steps, loss) +
                                '[trigger] P={:.3f}  R={:.3f}  F1={:.3f}'.format(trigger_p, trigger_r, trigger_f1)
                                )
                    cum_loss = 0
                pass
        if not args.gradient_debug:
            new_val_loss = validation(model, valid_dataloader)
            if args.tbx:
                writer.add_scalar(tbx_title + 'val', new_val_loss, tot_n_steps)
            logger.info("valid loss: {}".format(new_val_loss))
            early_stopping(new_val_loss)
            if early_stopping.improving:
                if os.path.exists(model_dir) and os.path.isdir(model_dir):
                    shutil.rmtree(model_dir)
                    os.mkdir(model_dir)
                torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME + "finish"))
                adapter_list = model.PreModel.get_adapter_list()

                np.save(os.path.join(model_dir, "adapter_list.npy"), adapter_list)
                np.save(os.path.join(model_dir, "ita_list.npy"), detached_ita)
                np.save(os.path.join(model_dir, "uni_adapter.npy"), model.get_uni_adapter())
                logger.info("BEST MODEL SAVED!")
            if early_stopping.early_stop:
                break

        logger.info("ITA:")
        logger.info(ita)

        print_para(model)
        if args.gradient_debug:
            exit(0)

        if ep == args.warm_mix_step - 1:
            model.PreModel.config.forward_mode = 'Mix'
            for name, param in model.named_parameters():
                if "ita" in name:
                    param.requires_grad = True
    model = load_model(model_dir)
    ita_list = np.load(os.path.join(model_dir, "ita_list.npy"), allow_pickle=True)
    uni_adapter = np.load(os.path.join(model_dir, "uni_adapter.npy"), allow_pickle=True)
    model.set_ita(ita_list)
    model.set_uni_adapter(uni_adapter)
    if args.layer_debug:
        for i, layer_ita in enumerate(ita):
            if i == args.layer_debug_cnt:
                layer_ita[1] = 1.0

    # Make decision on which adapter to use for the new task (in each layer)
    cnt_true = model.PreModel.setup_task_adapter(task_ids[0])
    torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME + "finish"))
    adapter_list = model.PreModel.get_adapter_list()
    np.save(os.path.join(model_dir, "adapter_list.npy"), adapter_list)

    if cnt_true > 0:
        #fit_or_not = True
        fit_or_not = False
    else:
        fit_or_not = False

    # Not using Fit stage now
    current_fit_epoch = None
    trans = True



    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    return model, fit_or_not, trans, current_fit_epoch


# An extra phase to train newly added modules and reused modules on the new task only, not used
def Fit(task_ids, model, current_fit_epoch=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to Fit { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)

    train_dataset = [swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1) for t in tasks]
    valid_dataset = [TASK_DICT[t]["test"] for t in tasks]

    if args.load_model_for_stage:
        model = load_model(model_dir)
    else:
        load_model(model_dir)

    # Fit preparation
    model.config.forward_mode = 'Fit'
    model.config.testing = False
    model.train_adapter(str(task_ids[0]))
    # model.train_adapter_subname([str(task_ids[0])])
    model.cuda()
    if args.clear_model:
        model = clear(model)
    param_opt = learnable_para_calculate(model, "whole", True)

    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir, CONFIG_NAME))
    global TOKENS_WEIGHT
    while 50260 + len(args.tasks) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
        logger.info("Add one dim weight!")

    if not args.fp32:  # again because resize_token_embeddings makes embedding layer fp32
        model = FP16_Module(model)

    logger.warning("In Fit, not using extra data now...")
    # train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    train_qadata = ACE2005Dataset(train_dataset, task_ids[0])
    valid_qadata = ACE2005Dataset(valid_dataset, task_ids[0])

    max_train_batch_size = args.z_max_batch_size
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    valid_dataloader = create_dataloader(valid_qadata, "test")

    n_train_epochs = args.fit_epoch
    if current_fit_epoch is not None:
        n_train_epochs = current_fit_epoch

    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    if args.whole_optim:
        param_optimizer = list(model.named_parameters())
    else:
        param_optimizer = param_opt

    # logger.info(param_optimizer)
    no_decay = ['bias', 'ln_1', 'ln_2', 'ln_f']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    logger.info("USE ARGS.ADAM_EPSILON NOW.....")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.z_warmup_step)
    train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL,
                                      weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half))

    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)

    # model.config.batch_task_id = task_ids[0]
    for ep in range(n_train_epochs):
        model.train()
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        # learnable_para_calculate(model, "whole")
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y, task_id, idx) in enumerate(train_dataloader):
            # logger.info("One step!!!")
            n_inputs = cqa[0].shape[0]
            lens = cqa[0].shape[1]
            if lens > 500:
                logger.info(cqa[0].shape)
                continue

            model.config.batch_task_id = task_id[0][0].item()
            losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)

            if losses[1].item() == 0:
                loss = losses[0]
            else:
                loss = losses[0] + losses[1]

            train_once(loss, n_inputs)

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if args.constant_sch or task_ids[0] > 0:
                lr = scheduler.get_lr()[0]
            else:
                lr = scheduler.get_lr()

            if (n_steps + 1) % args.logging_steps == 0:
                logger.info(
                    'progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f}, avg batch size {:.1f}'
                    .format(ep + cur_n_inputs / len(train_qadata),
                            lr, cum_loss / cur_n_inputs,
                            cum_qa_loss / cur_n_inputs, cum_lm_loss / cur_n_inputs,
                            cur_n_inputs / (n_steps + 1)))

        if not args.gradient_debug:
            tot_n_steps += (n_steps + 1)
            val_loss, val_qa_loss, val_lm_loss = validation(model, valid_dataloader, train_loss_fct)
            logger.info(
                'epoch {}/{} done , tot steps {} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f}, val loss {:.2f}, vqa loss {:.2f}, vlm loss {:.2f}, avg batch size {:.1f}'.format(
                    ep + 1, n_train_epochs, tot_n_steps,
                    cum_loss / cur_n_inputs, cum_qa_loss / cur_n_inputs,
                    cum_lm_loss / cur_n_inputs, val_loss,
                    val_qa_loss, val_lm_loss, cur_n_inputs / (n_steps + 1)
                ))

        print_para(model)

    torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME + "finish"))
    adapter_list = model.get_adapter_list()
    np.save(os.path.join(model_dir, "adapter_list.npy"), adapter_list)
    logger.info("MODEL SAVED!")

    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    return model


# Training stage
def Transfer(task_ids, model, fit_bonus=0):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to transfer { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)

    train_extra_data = []
    if not args.generate_after:
        if ("lll" in args.seq_train_type or "llewc" in args.seq_train_type) and task_ids[
            0] > 0 and not args.pseudo_ablation:
            adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"), allow_pickle=True)
            replay = []
            for layer_list in adapter_list:
                c_module = layer_list["adapter_function"][task_ids[0]]
                for i in range(task_ids[0]):
                    if layer_list["adapter_function"][i] == c_module:
                        if i not in replay:
                            replay.append(i)
            # only replay those tasks which share modules with the current task
            logger.info("replay tasks: {}".format(replay))

            if len(replay) > 0:
                # TODO Add experience replay
                '''prev_task = args.tasks[task_ids[0] - 1]
                model.PreModel.config.forward_mode = 'Transfer'
                model.PreModel.config.testing = False
                with torch.no_grad():
                    create_extra_data(tasks[0], prev_task, model, train_extra_data, None, None, replay)'''
            pass

        logger.info('extra training data size: {}'.format(len(train_extra_data)))
    # prepare for transfer
    if not model:
        # this is for the first task!
        # which_model_to_load = model_dir if os.path.isfile(os.path.join(model_dir, FINAL_SAVE_NAME)) else args.model_name

        # You can use pre-downloaded pretrained model, or download it from the internet
        model = MODEL_CLASS.from_pretrained('./bert-large-uncased')

        # Initialize special generation tokens (for every task) IN ONE TIME!
        # DON'T add a special token everytime we have a new task (as LAMOL original implementation did)
        # This design will make the training process more stable!

        torch.manual_seed(42)
        model.resize_token_embeddings(50260 + len(args.tasks))
        # logger.info(model.transformer.wte.weight)
        torch.manual_seed(args.seed)

        model.config.forward_mode = 'Transfer'
        model.config.testing = False
        names_to_train = model.add_adapter(str(task_ids[0]), config=args.adapt_type)
        if args.pretrain_adapter:
            load_pre_adapter(model, str(task_ids[0]))

        if not args.adapterdrop:
            model.train_adapter(names_to_train)
        else:
            model.train_adapter(names_to_train, [0, 1, 2])
        model.cuda()
        if args.clear_model:
            model = clear(model)

        if not args.fp32:
            logger.info("Not support fp32 on mytransformers/adapters now...")
            exit(0)
        net = Net(trigger_size=len(all_triggers), PreModel=model, entity_size=len(all_entities),
                  all_postags=len(all_postags),
                  argument_size=len(all_arguments), device=args.device_ids[0], idx2trigger=idx2trigger)

        net.cuda()
        # model = FP16_Module(model)
    else:

        current_tasks = [args.tasks[task_ids[0]]]
        current_model_dir = get_model_dir(current_tasks)
        model = load_model(current_model_dir)


        model.PreModel.config.forward_mode = 'Transfer'
        model.PreModel.config.testing = False

        if args.partial_learn:
            model.PreModel.train_adapter(str(task_ids[0]))
        elif args.partial_transfer:
            model.PreModel.adapter_transfer()
        else:
            adapter_list = set(map(lambda x:x['adapter_function'][len(x['adapter_function']) - 1],np.load(os.path.join(model_dir, "adapter_list.npy"),allow_pickle=True)))

            model.PreModel.train_adapter([str(i) for i in adapter_list])

        net = model
        model = model.PreModel
        model.cuda()

        if args.clear_model:
            model = clear(model)
    param_opt = learnable_para_calculate(net, "whole", True)

    """
    if args.generate_after:
        if ("lll" in args.seq_train_type or "llewc" in args.seq_train_type) and task_ids[0] > 0 and not args.pseudo_ablation:
            prev_task = args.tasks[task_ids[0]-1]
            model.config.forward_mode = 'Transfer'
            model.config.testing = False
            with torch.no_grad():
                create_extra_data(tasks[0], prev_task, model, train_extra_data)

        logger.info('extra training data size: {}'.format(len(train_extra_data)))
"""
    # gen_token = get_gen_token(tasks[0])
    # TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    # SPECIAL_TOKENS[tasks[0]] = gen_token
    # SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    # logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir, CONFIG_NAME))
    # global TOKENS_WEIGHT
    # while 50260 + len(args.tasks) != TOKENS_WEIGHT.shape[0]:
    #     TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
    #     logger.info("Add one dim weight!")

    if not args.fp32:  # again because resize_token_embeddings makes embedding layer fp32
        model = FP16_Module(model)

    logger.warning("Transfer, using extra data now...")
    train_qadata = ACE2005Dataset('./ace2005/train.json', task_ids[0])
    valid_qadata = ACE2005Dataset('./ace2005/dev.json', task_ids[0])

    max_train_batch_size = args.z_max_batch_size
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    valid_dataloader = create_dataloader(valid_qadata, "test")

    if args.gradient_debug:
        n_train_epochs = 1
    elif task_ids[0] == 0:
        n_train_epochs = args.z_train_epochs[task_ids[0]]
    else:
        n_train_epochs = args.z_train_epochs[task_ids[0]] - fit_bonus

    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    if args.whole_optim:
        param_optimizer = list(model.named_parameters())
    else:
        param_optimizer = param_opt

    no_decay = ['bias', 'ln_1', 'ln_2', 'ln_f']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_grouped_names = [
        [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        [n for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    ]
    logger.info("name group")
    logger.info(optimizer_grouped_names)

    # logger.info("USE ARGS.ADAM_EPSILON NOW.....")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.z_train_lrs[task_ids[0]], weight_decay=args.l2)

    if args.constant_sch:
        logger.info("Start to use constant scheduler!")
        scheduler = get_constant_schedule_with_warmup(optimizer, args.z_warmup_step)
    elif not args.constant_sch and (not args.lamaml or (args.lamaml and task_ids[0] == 0)):
        logger.info("Start to use Annealling scheduler!")
        scheduler = AnnealingLR(optimizer, start_lr=args.z_train_lrs[task_ids[0]],
                                warmup_iter=int(args.n_warmup_ratio * len(train_qadata)),
                                num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)
    elif not args.constant_sch:
        logger.info("Start to use Annealling scheduler!")
        scheduler = AnnealingLR(optimizer, start_lr=args.z_train_lrs[task_ids[0]],
                                warmup_iter=int(args.n_warmup_ratio * len(train_qadata)),
                                num_iters=int(train_qadata.get_c_len() * 2 * n_train_epochs) + 100,
                                decay_style=args.decay_style)

    tot_n_steps = 0
    # train_once = TrainStep(model, optimizer, scheduler)

    # The reason why we use "path" variable: (path is passed to AdamW, modified in mytransformers/optimization.py)
    # The calculation path in this stage is different for different tasks in this stage
    # since we are using AdamW, 
    # (!!!) a zero gradient cannot gaurantee that the parameter is not changed, it might be changed by the state of optimizers (!!!)
    # thus we need to keep the track of calculation path for each task manually and pass it to optimizer to avoid such strange behaviors
    # Attention!

    path = [None for i in range(task_ids[0] + 1)]
    if task_ids[0] > 0:
        path = []
        o_path = model.get_path()
        for i in range(task_ids[0] + 1):
            c_name = []
            for layer, j in enumerate(o_path):
                c_layer_name = j[i]
                c_name.append('.' + str(layer) + '.attention.output.adapters.' + c_layer_name + '.')
                c_name.append('.' + str(layer) + '.output.adapters.' + c_layer_name + '.')
            path_one = []
            path_two = []
            for n, p in param_optimizer:
                if not any(nd in n for nd in no_decay):  # no no-decay in n
                    flag = 0
                    for name in c_name:
                        if name in n:
                            flag = 1
                            path_one.append(True)
                            break
                    if flag == 0:
                        path_one.append(False)
                else:  # contains no-decay in n
                    flag = 0
                    for name in c_name:
                        if name in n:
                            flag = 1
                            path_two.append(True)
                            break
                    if flag == 0:
                        path_two.append(False)
            path.append([path_one, path_two])
        logger.info(path)

        shared = []
        for i, c_path in enumerate(path):
            if True in c_path[0] or True in c_path[1]:
                shared.append(True)
            else:
                shared.append(False)
        logger.info("shared: {}".format(shared))
    from early_stop import EarlyStopping
    early_stopping = EarlyStopping(patience=10, verbose=True, trace_func=lambda x: logger.info(x))
    tbx_title = 'transfer_'+str(task_ids[0])+'/'
    for ep in range(n_train_epochs):
        logger.info("Epoch {}".format(ep))
        # model.train()
        net.train()
        print_para(model)
        words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
        triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
        cum_loss = 0

        for n_steps, batch in tqdm.tqdm(enumerate(train_dataloader)):
            tot_n_steps+=1
            net.zero_grad()
            optimizer.zero_grad()
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm, task_id = batch
            '''if cqa is None:
                continue

            n_inputs = cqa[0].shape[0]
            lens = cqa[0].shape[1]'''
            # Consider to add this when you have memory error, this should be rarely happened on datasets used by our paper!
            """
            if lens > 500:
                logger.info(cqa[0].shape)
                continue
            """

            if task_ids[0] > 0:
                if not shared[task_id[0]]:
                    logger.info("no shared, skipping")
                    continue
            model.config.batch_task_id = task_id[0]
            trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = net.predict_triggers(
                tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, adjm=adjm)
            if len(argument_keys) > 0:
                argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d = net.predict_arguments(
                    argument_hidden, argument_keys, arguments_2d, adjm)
                # argument_loss = criterion(argument_logits, arguments_y_1d)
                loss = trigger_loss + args.LOSS_alpha * argument_loss
                # if i == 0:

                #     print("=====sanity check for triggers======")
                #     print('triggers_y_2d[0]:', triggers_y_2d[0])
                #     print("trigger_hat_2d[0]:", trigger_hat_2d[0])

                #     print("=======================")

                #     print("=====sanity check for arguments======")
                #     print('arguments_y_2d[0]:', arguments_y_2d[0])
                #     print('argument_hat_1d[0]:', argument_hat_1d[0])
                #     print("arguments_2d[0]:", arguments_2d)
                #     print("argument_hat_2d[0]:", argument_hat_2d)
                #     print("=======================")

            else:
                loss = trigger_loss
            cum_loss += loss.item()
            if args.tbx:
                writer.add_scalar(tbx_title+'train',loss.item(),tot_n_steps)
            nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            loss.backward()

            optimizer.step( path[model.config.batch_task_id])
            scheduler.step()

            #### 训练精度评估 ####
            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

            for ii, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(
                    zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
                triggers_hat = triggers_hat[:len(words)]
                triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

                # [(ith sentence, t_start, t_end, t_type_str)]
                triggers_true.extend([(ii, *item) for item in find_triggers(triggers)])
                triggers_pred.extend([(ii, *item) for item in find_triggers(triggers_hat)])

                # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
                for trigger in arguments['events']:
                    t_start, t_end, t_type_str = trigger
                    for argument in arguments['events'][trigger]:
                        a_start, a_end, a_type_idx = argument
                        arguments_true.append((ii, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

                for trigger in arguments_hat['events']:
                    t_start, t_end, t_type_str = trigger
                    for argument in arguments_hat['events'][trigger]:
                        a_start, a_end, a_type_idx = argument
                        arguments_pred.append((ii, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            '''# For forward calculation, we make sure all examples from one batch is from the same task
            # and set the config to this task id every time (also need to do this for inference and generation)
            model.config.batch_task_id = task_id[0][0].item()
            losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)
            if losses[1].item() == 0:
                loss = losses[0]
            else:
                loss = losses[0] + losses[1]
            train_once(loss, n_inputs, path[model.config.batch_task_id])

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs'''

            if args.gradient_debug and task_ids[0] == 0:
                break

            if args.constant_sch:
                lr = scheduler.get_lr()[0]
            else:
                lr = scheduler.get_lr()


            if (n_steps) % args.logging_steps == 0:  # monitoring
                trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
                argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
                ## 100step 清零
                words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
                triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
                #########################
                if len(argument_keys) > 0:
                    logger.info(
                        "[Events Detected]step: {}, loss: {:.3f}, trigger_loss:{:.3f}, argument_loss:{:.3f}".format(
                            n_steps,
                            loss,
                            trigger_loss.item(),
                            argument_loss.item()) +
                        '[trigger] P={:.3f}  R={:.3f}  F1={:.3f}'.format(trigger_p, trigger_r, trigger_f1) +
                        '[argument] P={:.3f}  R={:.3f}  F1={:.3f}'.format(argument_p, argument_r, argument_f1)
                    )
                    cum_loss = 0
                else:
                    logger.info("[No Events Detected]step: {}, loss: {:.3f} ".format(n_steps, loss) +
                                '[trigger] P={:.3f}  R={:.3f}  F1={:.3f}'.format(trigger_p, trigger_r, trigger_f1)
                                )
                    cum_loss = 0
                pass

        if not args.gradient_debug:
            new_val_loss = validation(net, valid_dataloader)
            if args.tbx:
                writer.add_scalar(tbx_title+'val', new_val_loss,tot_n_steps)
            logger.info("valid loss: {}".format(new_val_loss))
            early_stopping(new_val_loss)
            if early_stopping.improving:
                if os.path.exists(model_dir) and os.path.isdir(model_dir):
                    shutil.rmtree(model_dir)
                    os.mkdir(model_dir)
                torch.save(net.state_dict(), os.path.join(model_dir, SAVE_NAME + "finish"))
                adapter_list = model.get_adapter_list()
                np.save(os.path.join(model_dir, "adapter_list.npy"), adapter_list)
                logger.info("BEST MODEL SAVED!")
            if early_stopping.early_stop:
                break

        print_para(model)
        if args.gradient_debug and task_ids[0] > 0:
            exit(0)
    model = load_model(model_dir)


    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    if args.layer_debug and task_ids[0] == len(args.tasks) - 1:
        model.config.forward_mode = 'Transfer'
        model.config.testing = False
        gen_path = os.path.join(model_dir, "lm-origin-{}-{}.csv".format(args.layer_debug_cnt, args.partial_learn))
        holder = []
        with torch.no_grad():
            create_extra_data(tasks[0], tasks[0], model, holder, None, gen_path, [1])

        logger.info("Modifying list")
        model.modify_list(args.layer_debug_cnt, 0, 1)

        gen_path = os.path.join(model_dir, "lm-modified-{}-{}.csv".format(args.layer_debug_cnt, args.partial_learn))
        holder = []
        with torch.no_grad():
            create_extra_data(tasks[0], tasks[0], model, holder, None, gen_path, [1])

        exit(0)

    return net


if __name__ == '__main__':
    global writer
    if args.tbx:
        logger.info("Using TensorBoardX")
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()
    '''import builtins
    from inspect import getframeinfo, stack

    original_print = print


    def print_wrap(*args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        original_print("FN:", caller.filename, "Line:", caller.lineno, "Func:", caller.function, ":::", *args, **kwargs)


    builtins.print = print_wrap'''
    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    if not args.z_debug:
        make_dir(args.model_dir_root)

        init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
        logger.info('args = {}'.format(str(args)))

        model = None
        if args.seq_train_type in ["multitask", "multilm"]:
            model = train(list(range(len(args.tasks))), model)
        else:
            if args.unbound:
                TASK_DICT = lll_unbound_setting(split_size=args.unbound)
            for task_id in range(len(args.tasks)):
                if task_id == 0:
                    model = Transfer([task_id], model)
                else:
                    if not args.fake_mix_debug:
                        model, Fit_or_Not, trans, current_fit_epoch = Mix([task_id], model)

                        fit_bonus = 0
                        if Fit_or_Not:
                            model = Fit([task_id], model, current_fit_epoch)
                            fit_bonus = 0
                        if trans:
                            model = Transfer([task_id], model, fit_bonus)
                    else:
                        logger.info("In fake mix debug!")
                        tmp_model = copy.deepcopy(model)
                        tmp_model, Fit_or_Not, trans, current_fit_epoch = Mix([task_id], tmp_model)
                        del tmp_model

                        tasks = [args.tasks[task_id]]
                        model_dir = get_model_dir(tasks)
                        adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
                        model.update_adapter_list(adapter_list)
                        model = Transfer([task_id], model, 0)
    else:
        init_logging(os.path.join(args.model_dir_root, 'log_train_debug.txt'))
        logger.info('args = {}'.format(str(args)))

        model = None
        if args.z_debug_tsk_num >= 1:
            from mytransformers import BertModel, BertConfig

            tasks = [args.tasks[args.z_debug_tsk_num - 1]]
            if args.z_debug_start_from_trans:
                tasks = [args.tasks[args.z_debug_tsk_num]]

            model, adapter_list = load_model(get_model_dir(tasks), return_adapter_list=True)

            global TOKENS_WEIGHT
            tsk_cnt = 0
            while tsk_cnt < args.z_debug_tsk_num:
                '''TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
                gen_token = get_gen_token(args.tasks[tsk_cnt])
                TOKENIZER.add_tokens([gen_token])
                SPECIAL_TOKENS[args.tasks[tsk_cnt]] = gen_token
                SPECIAL_TOKEN_IDS[args.tasks[tsk_cnt]] = TOKENIZER.convert_tokens_to_ids(gen_token)'''
                tsk_cnt += 1

            # model.resize_token_embeddings(len(TOKENIZER))
            if not args.fp32:
                model = FP16_Module(model)

            '''while 50260 + len(args.tasks) != TOKENS_WEIGHT.shape[0]:
                TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
                logger.info("Add one dim weight!")'''

        for task_id in range(args.z_debug_tsk_num, len(args.tasks)):
            # First task
            if task_id == 0:
                model = Transfer([task_id], model)
            # Recover training from one task
            elif task_id == args.z_debug_tsk_num and args.z_debug_start_from_trans:
                fit_bonus = 0
                model.PreModel.config.forward_mode = 'Transfer'
                model.PreModel.config.testing = False
                adapter_list = list(map(lambda x: x['adapter_function'][len(x['adapter_function']) - 1],
                                        adapter_list))

                model.PreModel.train_adapter([str(i) for i in adapter_list])

                model = Transfer([task_id], model, fit_bonus)
            # not the first task
            else:
                model, Fit_or_Not, trans, current_fit_epoch = Mix([task_id], model)
                fit_bonus = 0
                # Fit stage is not used in this paper, args.fit_epoch is set to 0 by default
                if Fit_or_Not:
                    model = Fit([task_id], model, current_fit_epoch)
                    fit_bonus = 0
                if trans:
                    model = Transfer([task_id], model, fit_bonus)
