import torch
from tqdm import tqdm
'''在39对eval函数做了修改11.18'''

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, pos, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.pos = pos
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, gather_ids, gather_masks, partial_masks):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.gather_ids = gather_ids
        self.gather_masks = gather_masks
        self.partial_masks = partial_masks


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, logger):
        self.logger = logger

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines


class Processor(DataProcessor):
    """Processor NQG data set."""

    def __init__(self, logger, dataset, latent_size):
        self.logger = logger
        if dataset == "ACE04" or dataset == "ACE05":
            self.labels = ['PER', 'LOC', 'ORG', 'GPE', 'FAC', 'VEH', 'WEA']
        elif dataset == "GENIA":
            self.labels = ['None', 'G#RNA', 'G#protein', 'G#DNA', 'G#cell_type', 'G#cell_line']
        elif dataset == "CONLL":
            self.labels = ['LOC', 'PER', 'ORG', 'MISC']
        elif dataset=='WEIBO_++':
            self.labels=['LOC.NAM','PER.NAM','PER.NOM','LOC.NOM','GPE.NAM','ORG.NOM','ORG.NAM','GPE.NOM']
        elif dataset=='WEIBO_++MERGE':
            self.labels=['LOC','PER','GPE','ORG']
        elif dataset=='YOUKU':
            self.labels=['TV','PER','NUM','ORG']
        elif dataset=='ECOMMERCE':
            self.labels=['HCCX','MISC','XH','HPPX']
        elif dataset=='ACE':
            self.labels=['PHYS', 'WEA', 'PART-WHOLE', 'Transaction:Transfer-Ownership', 'rTransaction:Transfer-Money', 'rLife:Marry', 'rTransaction:Transfer-Ownership', 'rJustice:Sue', 'Justice:Charge-Indict', 'rBusiness:Declare-Bankruptcy', 'Business:End-Org', 'rJustice:Arrest-Jail', 'rORG-AFF', 'rContact:Phone-Write', 'FAC', 'rJustice:Extradite', 'rConflict:Demonstrate', 'PER-SOC', 'LOC', 'rPER-SOC', 'Justice:Convict', 'rJustice:Appeal', 'rContact:Meet', 'rJustice:Release-Parole', 'GPE', 'Conflict:Attack', 'Justice:Extradite', 'rPersonnel:Start-Position', 'rBusiness:End-Org', 'Contact:Meet', 'Life:Injure', 'rJustice:Sentence', 'rConflict:Attack', 'rBusiness:Merge-Org', 'Life:Die', 'Justice:Acquit', 'ORG-AFF', 'rGEN-AFF', 'Contact:Phone-Write', 'rJustice:Trial-Hearing', 'rJustice:Pardon', 'rLife:Injure', 'Transaction:Transfer-Money', 'Justice:Trial-Hearing', 'Justice:Arrest-Jail', 'Justice:Fine', 'Justice:Sue', 'Justice:Sentence', 'rLife:Divorce', 'Justice:Appeal', 'rPHYS', 'Business:Declare-Bankruptcy', 'Life:Marry', 'VEH', 'rLife:Die', 'GEN-AFF', 'ORG', 'rJustice:Charge-Indict', 'PER', 'Justice:Release-Parole', 'rMovement:Transport', 'Business:Merge-Org', 'Life:Divorce', 'Movement:Transport', 'Justice:Pardon', 'rJustice:Acquit', 'trigger', 'Personnel:Start-Position', 'Business:Start-Org', 'rLife:Be-Born', 'rPersonnel:End-Position', 'rART', 'rJustice:Fine', 'rPART-WHOLE', 'rPersonnel:Nominate', 'ART', 'Justice:Execute', 'Life:Be-Born', 'Conflict:Demonstrate', 'rJustice:Execute', 'Personnel:Nominate', 'rPersonnel:Elect', 'rBusiness:Start-Org', 'Personnel:End-Position', 'rJustice:Convict', 'Personnel:Elect']
        elif dataset=='JSON_ACE':
            self.labels=['Victim', 'FAC', 'rAttacker', 'rProsecutor', 'rORG-AFF.Founder', 'rInstrument', 'ORG-AFF.Student-Alum', 'rAgent', 'ORG-AFF.Ownership', 'PER-SOC.Lasting-Personal', 'rOrigin', 'trigger', 'GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'rORG-AFF.Membership', 'ART.User-Owner-Inventor-Manufacturer', 'Giver', 'rGEN-AFF.Org-Location', 'Artifact', 'rPART-WHOLE.Subsidiary', 'rPlace', 'rPHYS.Near', 'Org', 'PER-SOC.Business', 'rSeller', 'rART.User-Owner-Inventor-Manufacturer', 'VEH', 'PER-SOC.Family', 'GPE', 'Place', 'Entity', 'rOrg', 'rGEN-AFF.Citizen-Resident-Religion-Ethnicity', 'LOC', 'Agent', 'rEntity', 'rDestination', 'ORG-AFF.Founder', 'rTarget', 'rVehicle', 'rPlaintiff', 'rORG-AFF.Sports-Affiliation', 'Defendant', 'Attacker', 'rPerson', 'Vehicle', 'PER', 'rGiver', 'rAdjudicator', 'rORG-AFF.Employment', 'Instrument', 'ORG-AFF.Sports-Affiliation', 'rBuyer', 'PART-WHOLE.Artifact', 'Person', 'Beneficiary', 'Adjudicator', 'rPER-SOC.Lasting-Personal', 'Plaintiff', 'rORG-AFF.Ownership', 'ORG-AFF.Investor-Shareholder', 'rPER-SOC.Family', 'ORG-AFF.Membership', 'GEN-AFF.Org-Location', 'rPART-WHOLE.Artifact', 'PART-WHOLE.Geographical', 'Target', 'rPART-WHOLE.Geographical', 'rDefendant', 'WEA', 'rORG-AFF.Investor-Shareholder', 'PART-WHOLE.Subsidiary', 'ORG-AFF.Employment', 'Seller', 'Origin', 'PHYS.Located', 'rVictim', 'Prosecutor', 'rRecipient', 'Buyer', 'Destination', 'rArtifact', 'rBeneficiary', 'ORG', 'rPER-SOC.Business', 'rPHYS.Located', 'Recipient', 'PHYS.Near', 'rORG-AFF.Student-Alum']
        else:
            raise NotImplementedError()

        if dataset == "ACE05" or dataset == "GENIA" or dataset == "ACE04" or dataset == "CONLL" \
                or dataset=='WEIBO_++' or dataset=='WEIBO_++MERGE' or dataset=='YOUKU' or dataset=='ECOMMERCE' \
                or dataset=='ACE' or dataset=='JSON_ACE':
            self.interval = 4
        else:
            raise NotImplementedError()

        self.latent_size = latent_size

    def get_train_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "dev")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []

        for i in range(0, len(lines), self.interval):
            text_a = lines[i]
            label = lines[i + 2]

            examples.append(
                InputExample(guid=len(examples), text_a=text_a, pos=None, label=label))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []

        for (ex_index, example) in enumerate(tqdm(examples)):

            tokens = tokenizer.tokenize(example.text_a)

            gather_ids = list()
            for (idx, token) in enumerate(tokens):
                if (not token.startswith("##") and idx < max_seq_length - 2):
                    gather_ids.append(idx + 1)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            
            #is_head
            gather_padding = [0] * (max_seq_length - len(gather_ids))
            gather_masks = [1] * len(gather_ids) + gather_padding
            #is_head在句子中的索引
            gather_ids += gather_padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(gather_ids) == max_seq_length
            assert len(gather_masks) == max_seq_length

            partial_masks = self.generate_partial_masks(example.text_a.split(' '), max_seq_length, example.label,
                                                        self.labels)

            if ex_index < 2:
                self.logger.info("*** Example ***")
                self.logger.info("guid: %s" % (example.guid))
                self.logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                self.logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                self.logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                self.logger.info(
                    "gather_ids: %s" % " ".join([str(x) for x in gather_ids]))
                self.logger.info(
                    "gather_masks: %s" % " ".join([str(x) for x in gather_masks]))
                # self.logger.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks))

        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def generate_partial_masks_ori(self, tokens, max_seq_length, labels, tags):
        #所有标签
        total_tags_num = len(tags) + self.latent_size
        #L0，已知标签序列
        labels = labels.split('|')
        label_list = list()

        for label in labels:
            if not label:
                continue
            sp = label.strip().split(' ')
            start, end = sp[0].split(',')[:2]
            start = int(start)
            end = int(end) - 1
            label_list.append((start, end, sp[1]))
        #初始化所有节点为隐藏节点
        mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection

        for start, end, tag in label_list:
            #初始化观测到的节点
            if start < max_seq_length and end < max_seq_length:
                tag_idx = tags.index(tag)
                mask[start][end][tag_idx] = 1
                for k in range(total_tags_num):
                    if k != tag_idx:
                        mask[start][end][k] = 0
            #交叉边界
            for i in range(l):
                if i > end:
                    continue
                for j in range(i, l):
                    if j < start:
                        continue
                    if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
                        for k in range(total_tags_num):
                            mask[i][j][k] = 0

        #终点大于起点，（2，0）
        for i in range(l):
            for j in range(0, i):
                for k in range(total_tags_num):
                    mask[i][j][k] = 0

        #L0=0,L1=1
        for i in range(l):
            for j in range(i, l):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        if k < len(tags):
                            #不在观察序列中必然不在观测序列标签集中
                            mask[i][j][k] = 0
                        else:
                            #可能在其他标签集中
                            mask[i][j][k] = 1
        #将无用的拒绝
        for i in range(max_seq_length):
            for j in range(max_seq_length):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        mask[i][j][k] = 0
        

        return mask

    def generate_partial_masks(self, tokens, max_seq_length, labels, tags):
        #所有标签
        total_tags_num = len(tags) + self.latent_size
        #L0，已知标签序列
        labels = labels.split('|')
        label_list = list()

        for label in labels:
            if not label:
                continue
            sp = label.strip().split(' ')
            start, end = sp[0].split(',')[:2]
            start = int(start)
            end = int(end) - 1
            label_list.append((start, end, sp[1]))
        #初始化所有节点为隐藏节点
        mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection
        import random
        random.shuffle(label_list)

        flag=0
        for start, end, tag in label_list:
            if flag!=0:
            #初始化观测到的节点
                if start < max_seq_length and end < max_seq_length:
                    tag_idx = tags.index(tag)
                    mask[start][end][tag_idx] = 1
                    for k in range(total_tags_num):
                        if k != tag_idx:
                            mask[start][end][k] = 0
            flag+=1
            #交叉边界
            for i in range(l):
                if i > end:
                    continue
                for j in range(i, l):
                    if j < start:
                        continue
                    if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
                        for k in range(total_tags_num):
                            mask[i][j][k] = 0

        #终点大于起点，（2，0）
        for i in range(l):
            for j in range(0, i):
                for k in range(total_tags_num):
                    mask[i][j][k] = 0

        #L0=0,L1=1
        for i in range(l):
            for j in range(i, l):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        if k < len(tags):
                            #不在观察序列中必然不在观测序列标签集中
                            mask[i][j][k] = 0
                        else:
                            #可能在其他标签集中
                            mask[i][j][k] = 1
        #将无用的拒绝
        for i in range(max_seq_length):
            for j in range(max_seq_length):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        mask[i][j][k] = 0
        

        return mask


class MultitasksResultItem():

    def __init__(self, id, start_prob, end_prob, span_prob, label_id, position_id, start_id, end_id):
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.span_prob = span_prob
        self.id = id
        self.label_id = label_id
        self.position_id = position_id
        self.start_id = start_id
        self.end_id = end_id

def eval(args, outputs, partial_masks, label_size, gather_masks):

    correct, pred_count, gold_count = 0, 0, 0
    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()
    for output, partial_mask, l in zip(outputs, partial_masks, gather_masks):

        golds = list()
        preds = list()

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0:
                    if output[i][j] < label_size:
                        preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

    return correct, pred_count, gold_count

def eval_(args, outputs, partial_masks, labels, gather_masks):
    '''
    relation和event
    (只统计relation：在span里面，使用gold entity判断预测的关系是否正确，
    测试的时候使用新的标注方法，标注所有可能的关系，只要预测对其中的一个就算正确，要去重

    relation和event(包含entity，
    预测的entity判断预测的关系是否正确

    '''
    trigger_types= ['trigger']
    entity_types= ['LOC', 'GPE', 'ORG', 'FAC', 'PER', 'WEA', 'VEH']
    relation_types= ['PER-SOC.Business', 'rPART-WHOLE.Geographical', 'rORG-AFF.Founder', 'rART.User-Owner-Inventor-Manufacturer', 'rORG-AFF.Employment', 'ORG-AFF.Sports-Affiliation', 'rORG-AFF.Investor-Shareholder', 'ORG-AFF.Student-Alum', 'PER-SOC.Family', 'PART-WHOLE.Subsidiary', 'ORG-AFF.Employment', 'ORG-AFF.Ownership', 'PER-SOC.Lasting-Personal', 'PHYS.Located', 'PART-WHOLE.Artifact', 'GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'rORG-AFF.Membership', 'rGEN-AFF.Citizen-Resident-Religion-Ethnicity', 'rPER-SOC.Lasting-Personal', 'ART.User-Owner-Inventor-Manufacturer', 'ORG-AFF.Founder', 'rGEN-AFF.Org-Location', 'rORG-AFF.Ownership', 'ORG-AFF.Investor-Shareholder', 'rPER-SOC.Family', 'ORG-AFF.Membership', 'rORG-AFF.Sports-Affiliation', 'GEN-AFF.Org-Location', 'rPART-WHOLE.Subsidiary', 'rPER-SOC.Business', 'rPHYS.Located', 'PHYS.Near', 'rPHYS.Near', 'rPART-WHOLE.Artifact', 'PART-WHOLE.Geographical', 'rORG-AFF.Student-Alum']
    event_types= ['Victim', 'rSeller', 'rAttacker', 'rProsecutor', 'rGiver', 'rDefendant', 'rAdjudicator', 'Instrument', 'rInstrument', 'rAgent', 'Place', 'Seller', 'Entity', 'rBuyer', 'rOrigin', 'Origin', 'rVictim', 'Person', 'rOrg', 'Prosecutor', 'Beneficiary', 'Adjudicator', 'Plaintiff', 'Agent', 'rEntity', 'rRecipient', 'rDestination', 'Buyer', 'rTarget', 'Giver', 'rVehicle', 'Destination', 'rPlaintiff', 'rArtifact', 'rBeneficiary', 'Defendant', 'Artifact', 'rPlace', 'Attacker', 'rPerson', 'Vehicle', 'Recipient', 'Org', 'Target']
    label_size=len(labels)

    trigger_ids=[labels.index(item) for item in trigger_types]
    entity_ids=[labels.index(item) for item in entity_types]
    relation_ids=[labels.index(item) for item in relation_types]
    event_ids=[labels.index(item) for item in event_types]

    correct, pred_count, gold_count = 0, 0, 0
    trigger_pred_count,trigger_gold_count,trigger_correct=0,0,0
    relation_preds_count,relation_gold_count,relation_correct=0,0,0
    event_preds_count,event_gold_count,event_correct=0,0,0
    entity_preds_count,entity_gold_count,entity_correct=0,0,0
    two_relation_preds_count,two_relation_gold_count,two_relation_correct=0,0,0
    two_event_preds_count,two_event_gold_count,two_event_correct=0,0,0

    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()
    for output, partial_mask, l in zip(outputs, partial_masks, gather_masks):

        golds = list()
        preds = list()
        trigger_golds=list()
        trigger_preds=list()
        entity_golds=list()
        entity_preds=list()
        relation_golds=list()
        relation_preds=list()
        event_golds=list()
        event_preds=list()
        two_relation_preds=[]
        two_relation_golds=[]
        two_event_golds=[]
        two_event_preds=[]
        relation_golds_=[]
        event_golds_=[]
        entity_tuples=[]
        trigger_tuples=[]
        preds_entity_tuples=[]
        preds_trigger_tuples=[]

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0:
                    if output[i][j] < label_size:
                        preds.append("{}_{}_{}".format(i, j, int(output[i][j])))

                    if output[i][j] in trigger_ids:
                        trigger_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        preds_trigger_tuples.append([i,j])

                    if output[i][j] in entity_ids:
                        entity_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        preds_entity_tuples.append([i,j])
                    
                    if output[i][j] in relation_ids:
                        relation_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                    
                    if output[i][j] in event_ids:
                        event_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))

                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
                        if k in trigger_ids:
                            trigger_golds.append("{}_{}_{}".format(i, j, k))
                            trigger_tuples.append([i,j])
                        
                        if k in entity_ids:
                            entity_golds.append("{}_{}_{}".format(i, j, k))
                            entity_tuples.append([i,j])

        for i in range(l):
            for j in range(l):
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        if k in relation_ids:
                            relation_golds_.append("{}_{}_{}".format(i, j, k))
                            entity_1=[]
                            entity_2=[]
                            for t in range(i,j):
                                if [i,t] in entity_tuples:
                                    entity_1.append([i,t])
                            for t in range(j,l):
                                if [j,t] in entity_tuples:
                                    entity_2.append([j,t])
                            if len(entity_1)>1 or len(entity_2)>1:
                                print(len(entity_1),len(entity_2))
                                print('关系对应的实体存在起始位置相同的')
                            if len(entity_1)==0 or len(entity_2)==0:
                                print(len(entity_1),len(entity_2))
                                print('关系对应的实体不存在')
                            start_1,end_1=entity_1[0]
                            start_2,end_2=entity_2[0]
                            relation_golds_tmp=[]
                            for s_i in range(start_1,end_1+1):
                                for e_i in range(start_2,end_2+1):
                                    relation_golds_tmp.append("{}_{}_{}".format(s_i, e_i, k))
                            relation_golds.append(relation_golds_tmp)

                            preds_entity_1=[]
                            preds_entity_2=[]
                            for t in range(i,j):
                                if [i,t] in preds_entity_tuples:
                                    preds_entity_1.append([i,t])
                            for t in range(j,l):
                                if [j,t] in preds_entity_tuples:
                                    preds_entity_2.append([j,t])
                            if len(preds_entity_1)>1 or len(preds_entity_2)>1:
                                print(len(preds_entity_1),len(preds_entity_2))
                                print('关系对应的预测实体存在起始位置相同的')
                            if len(preds_entity_1)==0 or len(preds_entity_2)==0:
                                print(len(preds_entity_1),len(preds_entity_2))
                                print('关系对应的预测实体不存在')
                            else:
                                start_1,end_1=preds_entity_1[0]
                                start_2,end_2=preds_entity_2[0]
                                two_relation_golds_tmp=[]
                                for s_i in range(start_1,end_1+1):
                                    for e_i in range(start_2,end_2+1):
                                        two_relation_golds_tmp.append("{}_{}_{}".format(s_i, e_i, k))
                                two_relation_golds.append(two_relation_golds_tmp)
                            
                        if k in event_ids:
                            event_golds_.append("{}_{}_{}".format(i, j, k))
                            trigger_1=[]
                            entity_2=[]
                            for t in range(i,j):
                                if [i,t] in trigger_tuples:
                                    trigger_1.append([i,t])
                                    for t in range(j,l):
                                        if [j,t] in entity_tuples:
                                            entity_2.append([j,t])
                                elif [i,t] in entity_tuples:
                                    entity_2.append([i,t])
                                    for t in range(j,l):
                                        if [j,t] in trigger_tuples:
                                            trigger_1.append([j,t])

                            if len(trigger_1)>1 or len(entity_2)>1:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger存在起始位置相同的')
                            if len(trigger_1)==0 or len(entity_2)==0:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger不存在')

                            start_1,end_1=trigger_1[0]
                            start_2,end_2=entity_2[0]
                            event_golds_tmp=[]
                            for s_i in range(start_1,end_1+1):
                                for e_i in range(start_2,end_2+1):
                                    event_golds_tmp.append("{}_{}_{}".format(s_i, e_i, k))
                            event_golds.append(event_golds_tmp)

                            preds_trigger_1=[]
                            preds_entity_2=[]
                            for t in range(i,j):
                                if [i,t] in preds_trigger_tuples:
                                    preds_trigger_1.append([i,t])
                                    for t in range(j,l):
                                        if [j,t] in preds_entity_tuples:
                                            preds_entity_2.append([j,t])
                                elif [i,t] in preds_entity_tuples:
                                    preds_entity_2.append([i,t])
                                    for t in range(j,l):
                                        if [j,t] in preds_trigger_tuples:
                                            preds_trigger_1.append([j,t])
                            if len(preds_trigger_1)>1 or len(preds_entity_2)>1:
                                print(len(preds_trigger_1),len(preds_entity_2))
                                print('事件对应的预测实体或trigger存在起始位置相同的')
                            if len(preds_trigger_1)==0 or len(preds_entity_2)==0:
                                print(len(preds_trigger_1),len(preds_entity_2))
                                print('事件对应的预测实体或trigger不存在')
                            else:
                                start_1,end_1=preds_trigger_1[0]
                                start_2,end_2=preds_entity_2[0]
                                two_event_tmp=[]
                                for s_i in range(start_1,end_1+1):
                                    for e_i in range(start_2,end_2+1):
                                        two_event_tmp.append("{}_{}_{}".format(s_i, e_i, k))
                                two_event_golds.append(two_event_tmp)

        '''对预测总数和正确的预测均去重'''
        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

        trigger_pred_count+=len(trigger_preds)
        trigger_gold_count+=len(trigger_golds)
        trigger_correct+=len(set(trigger_preds).intersection(set(trigger_golds)))

        entity_preds_count+=len(entity_preds)
        entity_gold_count+=len(entity_golds)
        entity_correct+=len(set(entity_preds).intersection(set(entity_golds)))

        relation_preds_count+=len(relation_preds)
        relation_gold_count+=len(relation_golds_)
        for relation_golds_single_set in relation_golds:
            if len(set(relation_preds).intersection(set(relation_golds_single_set)))>0:
                relation_correct+=1
                relation_preds_count=relation_preds_count-len(set(relation_preds).intersection(set(relation_golds_single_set)))+1
            
        event_preds_count+=len(event_preds)
        event_gold_count+=len(event_golds_)
        for event_golds_single_set in event_golds:
            if len(set(event_preds).intersection(set(event_golds_single_set)))>0:
                event_correct+=1
                event_preds_count=event_preds_count-len(set(event_preds).intersection(set(event_golds_single_set)))+1

        two_relation_preds_count+=len(relation_preds)
        two_relation_gold_count+=len(relation_golds_)
        for two_relation_golds_single_set in two_relation_golds:
            if len(set(relation_preds).intersection(set(two_relation_golds_single_set)))>0:
                two_relation_correct+=1
                two_relation_preds_count=two_relation_preds_count-len(set(relation_preds).intersection(set(two_relation_golds_single_set)))+1

        two_event_preds_count+=len(event_preds)
        two_event_gold_count+=len(event_golds_)
        for two_event_golds_single_set in two_event_golds:
            if len(set(event_preds).intersection(set(two_event_golds_single_set)))>0:
                two_event_correct+=1
                two_event_preds_count=two_event_preds_count-len(set(event_preds).intersection(set(two_event_golds)))+1

    return correct, pred_count, gold_count,trigger_correct,trigger_pred_count,trigger_gold_count, \
            entity_correct,entity_preds_count,entity_gold_count,relation_correct,relation_preds_count,relation_gold_count, \
            event_correct,event_preds_count,event_gold_count, two_relation_correct,two_relation_preds_count,two_relation_gold_count, \
            two_event_correct,two_event_preds_count,two_event_gold_count