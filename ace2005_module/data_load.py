import json
import re
import numpy as np
from migration_model.enet.corpus.Sentence import Sentence
from torch.utils import data
from mytransformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer, OpenAIGPTModel, OpenAIGPTTokenizer, \
    CTRLModel, CTRLTokenizer, TransfoXLModel, TransfoXLTokenizer, XLNetModel, XLNetTokenizer, XLMModel, XLMTokenizer, \
    DistilBertModel, DistilBertTokenizer, RobertaModel, RobertaTokenizer, XLMRobertaModel, XLMRobertaTokenizer, \
    AlbertModel, AlbertTokenizer

from ace2005_module.consts import NONE, PAD, CLS, SEP, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS


def build_vocab(labels, BIO_tagging=True):
    all_labels = [PAD, NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}
    #idx2label.update({tag: idx for idx, tag in enumerate(all_labels)})

    return all_labels, label2idx, idx2label


all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)


class ACE2005Dataset(data.Dataset):
    tokenizer = None

    def __init__(self, fpath, current_task):
        self.current_task = current_task
        self.task_name_dict = {'bc': 0, 'bn': 1, 'cts': 2, 'nw': 3, 'un': 4, 'wl': 5}
        self.task_id = []
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li, self.adjm_li = [], [], [], [], [], []

        with open(fpath, 'r') as f:
            data = json.load(f)

            def filter(x):
                x['path'] = self.task_name_dict[re.match('(\w+)/.*', x['path']).group(1)]
                return x

            data = list(map(filter, data))
            data = list(sorted(data, key=lambda x: x['path']))
            for item in data:
                if not (item['path'] == self.current_task):
                    continue
                words = item['words']
                entities = [[NONE] for _ in range(len(words))]
                triggers = [NONE] * len(words)
                postags = item['pos-tags']
                sentence = Sentence(json_content=item)
                adjm = (sentence.adjpos, sentence.adjv)
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }

                for entity_mention in item['golden-entity-mentions']:
                    arguments['candidates'].append(
                        (entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))

                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)
                        else:
                            entity_type = 'I-{}'.format(entity_type)

                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            entities[i].append(entity_type)

                for event_mention in item['golden-event-mentions']:
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        trigger_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)

                    event_key = (
                        event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        if role.startswith('Time'):
                            role = role.split('-')[0]
                        #arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))
                        arguments['events'][event_key].append((argument['start'], argument['end'], role))
                self.sent_li.append([CLS] + words + [SEP])
                self.task_id.append(item['path'])
                self.entities_li.append([[PAD]] + entities + [[PAD]])
                self.postags_li.append([PAD] + postags + [PAD])
                self.triggers_li.append(triggers)
                self.arguments_li.append(arguments)
                self.adjm_li.append(adjm)

    def __len__(self):
        #return 32
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, arguments, adjm, task_id = self.sent_li[idx], self.entities_li[idx], \
                                                                       self.postags_li[
                                                                           idx], self.triggers_li[idx], \
                                                                       self.arguments_li[idx], self.adjm_li[idx], \
                                                                       self.task_id[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        for w, e, p in zip(words, entities, postags):
            tokens = self.tokenizer.tokenize(w) if w not in [CLS, SEP] else [
                w]  ## w=offenses,而tokens= ['offense', '##s'],此时只保留offense,否则会导致触发词的漂移量错位
            tokens_xx = self.tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            p = [p] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            p = [postag2idx[postag] for postag in p]
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers, adjm, task_id

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def pad(batch):
    tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm, task_id = list(
        map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        postags_x_2d[i] = postags_x_2d[i] + [0] * (maxlen - len(postags_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]

    return tokens_x_2d, entities_x_3d, postags_x_2d, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d, adjm, task_id


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('../bert-large-uncased')
    ACE2005Dataset.tokenizer = tokenizer
    ds = ACE2005Dataset('../ace2005/test.json', 0)
    ds21 = ds[2]
    ds = ACE2005Dataset('../ace2005/test.json', 0)
    ds22 = ds[2]
    pass
