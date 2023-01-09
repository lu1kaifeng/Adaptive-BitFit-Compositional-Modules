"""
2020/2/5修改
按事件类型，分多个CRF预测argu
"""
import os
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

# from pytorch_pretrained_bert import BertModel
# from transformers import BertModel

from .consts import NONE, TRIGGERS, event_cls
from .data_load import idx2trigger, argument2idx, idx2argument
from .utils import find_triggers, find_argument
from migration_model.models.CRF import CRF

import numpy
import torch
from torch import nn
from torch.nn import functional as F
### 迁移 JMEE ###

from migration_model.enet.models.EmbeddingLayer import EmbeddingLayer, MultiLabelEmbeddingLayer


# from migration_model.enet.models.GCN import GraphConvolution
# from migration_model.enet.models.HighWay import HighWay
# from migration_model.enet.models.SelfAttention import AttentionLayer
#
# from migration_model.enet.util import BottledXavierLinear
#
# from migration_model.text_cls_models.TextCNN import NgramCNN


class Net(nn.Module):
    def __init__(self, trigger_size=None, entity_size=None, all_postags=None, PreModel=None, hyper_para=None,
                 idx2trigger=None, postag_embedding_dim=50,
                 argument_size=None, entity_embedding_dim=50, device=torch.device("cuda")):
        super().__init__()
        ## PreTrainModel

        self.PreModel = PreModel
        self.idx2trigger = idx2trigger

        self.argument_size = argument_size

        self.hidden_size = self.PreModel.config.hidden_size

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
        )
        self.fc_trigger = nn.Sequential(
            nn.Linear(self.hidden_size, trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(self.hidden_size * 2, argument_size),
        )

        self.device = device

    # Ngramcnn
    # self.NgramCNN = NgramCNN(hidden_size=self.hidden_size)
    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d,
                         arguments_2d):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        # postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        # postags_x_2d = self.postag_embed(postags_x_2d)
        # entity_x_2d = self.entity_embed(entities_x_3d)

        if self.training:
            self.PreModel.train()
            enc = self.PreModel(tokens_x_2d).last_hidden_state
        else:
            self.PreModel.eval()
            with torch.no_grad():
                enc = self.PreModel(tokens_x_2d).last_hidden_state


        # x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]
        # logits = self.fc2(x + enc)

        batch_size = tokens_x_2d.shape[0]

        x = torch.stack([torch.index_select(x[i], 0, head_indexes_2d[i]) for i in range(batch_size)])

        trigger_logits = self.fc_trigger(x)
        trigger_hat_2d = trigger_logits.argmax(-1)
        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = nn.functional.cross_entropy(trigger_logits, triggers_y_2d.view(-1))

        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = x[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = x[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):
        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.fc_argument(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        if type(a_type_idx) is str:
                            a_type_idx = argument2idx[a_type_idx]
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)


        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys,
                                                                                 argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, idx2argument[a_label]))
        argument_loss = nn.functional.cross_entropy(argument_logits, arguments_y_1d)
        return argument_loss, arguments_y_1d, argument_hat_1d, argument_hat_2d

    def get_ita(self):
        ita_list = []
        for block in self.PreModel.encoder.layer:
            ita_list.append(block.ita)
        return ita_list

    def get_uni_adapter(self):
        uni_adapter = []
        for block in self.PreModel.encoder.layer:
            uni_adapter.append(block.uni_adapter)
        return uni_adapter

    def set_uni_adapter(self, ita_list):
        for ita, block in zip(ita_list, self.PreModel.encoder.layer):
            block.uni_adapter = ita

    def set_ita(self, ita_list):
        for ita, block in zip(ita_list, self.PreModel.encoder.layer):
            block.ita = ita


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
