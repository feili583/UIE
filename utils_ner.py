import torch
from tqdm import tqdm
import sys
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
import random
import numpy as np

'''在39对eval函数做了修改11.18
在39对eval函数的三元组加入了min和max限制11.21
11.23对mask生成函数分成两个mask矩阵
23.10.21加入新的评估函数
23.10.22对评估函数进行修改,对数据输入进行修改,将partial_mask和eval_mask合并
23.10.25加入关系和论元所有的span处理
23.10.28将矩阵改为稀疏矩阵,加载到gpu上再转换
24.1.29将评估函数改为尾-头解析
24.1.31对评估函数进行bug修改,将边界范围修改为包含i-1
24.4.6对评估函数加入标签字典
24.4.7对评估函数进行bug修改,将边界范围修改为包含i
'''

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, pos, label=None, full_label=None):
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
        self.full_label = full_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, gather_ids, gather_masks, partial_masks, eval_masks):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.gather_ids = gather_ids
        self.gather_masks = gather_masks
        self.partial_masks = partial_masks
        self.eval_masks = eval_masks


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

    def __init__(self, logger, dataset, latent_size,trained_models):
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
            # self.labels=['Business.Declare-Bankruptcy', 'r_Prosecutor', 'GEN-AFF.Org-Location', 'r_Destination', 'r_Seller', 'Transaction.Transfer-Ownership', 'Contact.Meet', 'r_ORG-AFF.Investor-Shareholder', 'r_Artifact', 'Instrument', 'r_Target', 'r_Plaintiff', 'Agent', 'r_Buyer', 'FAC', 'r_PER-SOC.Business', 'r_ORG-AFF.Founder', 'Justice.Pardon', 'ART.User-Owner-Inventor-Manufacturer', 'PHYS.Located', 'Transaction.Transfer-Money', 'Org', 'Place', 'GPE', 'PER-SOC.Business', 'Target', 'r_Beneficiary', 'Recipient', 'PART-WHOLE.Geographical', 'Attacker', 'r_Origin', 'r_PHYS.Located', 'Prosecutor', 'VEH', 'Personnel.Nominate', 'PER-SOC.Lasting-Personal', 'r_Instrument', 'Buyer', 'r_Attacker', 'r_PHYS.Near', 'Justice.Release-Parole', 'Beneficiary', 'r_PART-WHOLE.Subsidiary', 'Justice.Trial-Hearing', 'Seller', 'Business.End-Org', 'r_PER-SOC.Family', 'Movement.Transport', 'r_Defendant', 'Life.Marry', 'r_Entity', 'GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'ORG-AFF.Membership', 'Life.Divorce', 'r_ORG-AFF.Sports-Affiliation', 'Contact.Phone-Write', 'r_Place', 'PART-WHOLE.Artifact', 'Plaintiff', 'Justice.Arrest-Jail', 'Justice.Sentence', 'r_Person', 'r_Adjudicator', 'Person', 'WEA', 'r_ORG-AFF.Student-Alum', 'r_Org', 'ORG-AFF.Founder', 'Conflict.Attack', 'PHYS.Near', 'Justice.Fine', 'PER', 'Personnel.End-Position', 'Artifact', 'Entity', 'Vehicle', 'r_ORG-AFF.Membership', 'ORG', 'Destination', 'r_GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'Personnel.Start-Position', 'Justice.Extradite', 'Business.Merge-Org', 'Life.Die', 'Justice.Charge-Indict', 'Adjudicator', 'ORG-AFF.Investor-Shareholder', 'Life.Be-Born', 'r_ART.User-Owner-Inventor-Manufacturer', 'Conflict.Demonstrate', 'r_GEN-AFF.Org-Location', 'Justice.Acquit', 'r_Victim', 'ORG-AFF.Employment', 'r_PART-WHOLE.Artifact', 'ORG-AFF.Ownership', 'Giver', 'r_ORG-AFF.Ownership', 'ORG-AFF.Sports-Affiliation', 'Defendant', 'Justice.Appeal', 'r_Agent', 'Origin', 'PART-WHOLE.Subsidiary', 'r_PART-WHOLE.Geographical', 'r_Giver', 'Justice.Convict', 'Personnel.Elect', 'Business.Start-Org', 'r_Recipient', 'Life.Injure', 'r_PER-SOC.Lasting-Personal', 'ORG-AFF.Student-Alum', 'Justice.Execute', 'PER-SOC.Family', 'Victim', 'LOC', 'r_Vehicle', 'r_ORG-AFF.Employment', 'Justice.Sue']
            # self.labels = ['PHYS.Near', 'r_GEN-AFF.Org-Location', 'ORG-AFF.Membership', 'PER-SOC.Business', 'r_ORG-AFF.Ownership', 'ORG-AFF.Investor-Shareholder', 'PHYS.Located', 'GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'LOC', 'r_ORG-AFF.Employment', 'VEH', 'r_ORG-AFF.Membership', 'r_ORG-AFF.Student-Alum', 'r_ART.User-Owner-Inventor-Manufacturer', 'ORG', 'r_PHYS.Near', 'PART-WHOLE.Artifact', 'PER-SOC.Family', 'r_GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'ORG-AFF.Student-Alum', 'PART-WHOLE.Subsidiary', 'r_PER-SOC.Lasting-Personal', 'ORG-AFF.Employment', 'r_PHYS.Located', 'ART.User-Owner-Inventor-Manufacturer', 'PER', 'GPE', 'WEA', 'r_PER-SOC.Family', 'PER-SOC.Lasting-Personal', 'ORG-AFF.Ownership', 'r_ORG-AFF.Founder', 'r_ORG-AFF.Sports-Affiliation', 'r_PART-WHOLE.Geographical', 'r_ORG-AFF.Investor-Shareholder', 'ORG-AFF.Founder', 'r_PER-SOC.Business', 'r_PART-WHOLE.Subsidiary', 'GEN-AFF.Org-Location', 'PART-WHOLE.Geographical', 'r_PART-WHOLE.Artifact', 'FAC', 'ORG-AFF.Sports-Affiliation']
            #7, 33, 36, 44(40, 80)
            # self.labels = ['GPE', 'PER', 'FAC', 'ORG', 'VEH', 'LOC', 'WEA', \
            #                 'Transaction.Transfer-Money', 'Business.Merge-Org', 'Justice.Release-Parole', 'Justice.Charge-Indict', 'Business.Start-Org', 'Justice.Appeal', 'Justice.Execute', 'Justice.Trial-Hearing', 'Personnel.Start-Position', 'Conflict.Demonstrate', 'Personnel.End-Position', 'Conflict.Attack', 'Justice.Convict', 'Justice.Sentence', 'Justice.Sue', 'Justice.Pardon', 'Life.Marry', 'Life.Be-Born', 'Life.Divorce', 'Movement.Transport', 'Justice.Extradite', 'Business.End-Org', 'Contact.Phone-Write', 'Life.Injure', 'Justice.Arrest-Jail', 'Business.Declare-Bankruptcy', 'Life.Die', 'Justice.Acquit', 'Personnel.Nominate', 'Contact.Meet', 'Personnel.Elect', 'Transaction.Transfer-Ownership', 'Justice.Fine', \
            #                 'ART.User-Owner-Inventor-Manufacturer', 'r_PART-WHOLE.Subsidiary', 'ORG-AFF.Ownership', 'ORG-AFF.Investor-Shareholder', 'ORG-AFF.Employment', 'r_PER-SOC.Family', 'r_ORG-AFF.Founder', 'GEN-AFF.Org-Location', 'GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'r_ORG-AFF.Student-Alum', 'r_ORG-AFF.Sports-Affiliation', 'r_GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'r_PHYS.Located', 'r_PER-SOC.Business', 'PER-SOC.Lasting-Personal', 'r_ORG-AFF.Membership', 'PHYS.Near', 'PART-WHOLE.Geographical', 'PART-WHOLE.Subsidiary', 'ORG-AFF.Sports-Affiliation', 'PHYS.Located', 'ORG-AFF.Student-Alum', 'r_ORG-AFF.Employment', 'PER-SOC.Family', 'r_PHYS.Near', 'PER-SOC.Business', 'r_ORG-AFF.Investor-Shareholder', 'r_PER-SOC.Lasting-Personal', 'r_GEN-AFF.Org-Location', 'r_ORG-AFF.Ownership', 'r_PART-WHOLE.Geographical', 'PART-WHOLE.Artifact', 'ORG-AFF.Membership', 'r_ART.User-Owner-Inventor-Manufacturer', 'r_PART-WHOLE.Artifact', 'ORG-AFF.Founder', \
            #                 'r_Prosecutor', 'r_Giver', 'Place', 'r_Recipient', 'Recipient', 'r_Org', 'r_Origin', 'r_Defendant', 'Prosecutor', 'r_Person', 'Origin', 'Vehicle', 'Attacker', 'Plaintiff', 'r_Victim', 'Org', 'r_Entity', 'r_Adjudicator', 'Artifact', 'Victim', 'Agent', 'r_Seller', 'Instrument', 'r_Place', 'r_Agent', 'r_Vehicle', 'Target', 'r_Buyer', 'r_Target', 'r_Attacker', 'r_Beneficiary', 'Beneficiary', 'r_Instrument', 'r_Artifact', 'r_Destination', 'Seller', 'Adjudicator', 'r_Plaintiff', 'Defendant', 'Giver', 'Buyer', 'Person', 'Entity', 'Destination']
            # self.labels = ['FAC', 'GPE', 'PER', 'ORG', 'VEH', 'LOC', 'WEA', 'ORG-AFF.Founder', 'r_PER-SOC.Family', 'ORG-AFF.Ownership', 'r_PART-WHOLE.Geographical', 'r_ORG-AFF.Employment', 'r_ORG-AFF.Founder', 'r_PHYS.Located', 'GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'PHYS.Near', 'r_PART-WHOLE.Artifact', 'ORG-AFF.Sports-Affiliation', 'PHYS.Located', 'r_ORG-AFF.Membership', 'PART-WHOLE.Geographical', 'PER-SOC.Lasting-Personal', 'ORG-AFF.Membership', 'r_PHYS.Near', 'ORG-AFF.Investor-Shareholder', 'PART-WHOLE.Subsidiary', 'PER-SOC.Business', 'r_GEN-AFF.Org-Location', 'r_ORG-AFF.Student-Alum', 'r_PER-SOC.Lasting-Personal', 'r_GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'GEN-AFF.Org-Location', 'ORG-AFF.Employment', 'ART.User-Owner-Inventor-Manufacturer', 'r_PART-WHOLE.Subsidiary', 'r_ORG-AFF.Investor-Shareholder', 'r_PER-SOC.Business', 'PART-WHOLE.Artifact', 'r_ORG-AFF.Ownership', 'r_ORG-AFF.Sports-Affiliation', 'PER-SOC.Family', 'ORG-AFF.Student-Alum', 'r_ART.User-Owner-Inventor-Manufacturer', 'Justice.Trial-Hearing', 'Business.Declare-Bankruptcy', 'Personnel.Elect', 'Life.Injure', 'Contact.Meet', 'Personnel.Nominate', 'Life.Be-Born', 'Justice.Charge-Indict', 'Personnel.Start-Position', 'Business.Start-Org', 'Justice.Fine', 'Business.End-Org', 'Conflict.Demonstrate', 'Justice.Extradite', 'Justice.Arrest-Jail', 'Transaction.Transfer-Money', 'Personnel.End-Position', 'Justice.Sentence', 'Business.Merge-Org', 'Life.Divorce', 'Justice.Convict', 'Justice.Sue', 'Life.Die', 'Justice.Release-Parole', 'Contact.Phone-Write', 'Justice.Pardon', 'Movement.Transport', 'Conflict.Attack', 'Life.Marry', 'Justice.Acquit', 'Transaction.Transfer-Ownership', 'Justice.Appeal', 'Justice.Execute', 'r_Buyer', 'Origin', 'Destination', 'Seller', 'r_Origin', 'Vehicle', 'Org', 'Agent', 'Beneficiary', 'r_Person', 'Person', 'Target', 'Entity', 'Adjudicator', 'Plaintiff', 'Defendant', 'Buyer', 'r_Place', 'r_Org', 'r_Destination', 'r_Entity', 'r_Agent', 'Recipient', 'Prosecutor', 'Attacker', 'r_Prosecutor', 'r_Giver', 'r_Target', 'Place', 'r_Defendant', 'r_Artifact', 'r_Victim', 'r_Recipient', 'r_Adjudicator', 'r_Plaintiff', 'r_Vehicle', 'r_Seller', 'r_Attacker', 'Victim', 'Artifact', 'r_Beneficiary', 'Instrument', 'Giver', 'r_Instrument']
            #240801
            # self.labels = ['Life.Divorce', 'Justice.Sentence', 'GEN-AFF', 'Conflict.Demonstrate', 'Justice.Extradite', 'Attacker', 'Business.Declare-Bankruptcy', 'Origin', 'Life.Injure', 'Giver', 'Justice.Trial-Hearing', 'PART-WHOLE', 'GPE', 'Entity', 'Destination', 'Life.Die', 'Business.Merge-Org', 'Recipient', 'Justice.Charge-Indict', 'Contact.Meet', 'Victim', 'PHYS', 'Target', 'Life.Marry', 'Business.End-Org', 'VEH', 'Justice.Sue', 'Life.Be-Born', 'Justice.Appeal', 'Personnel.Elect', 'Defendant', 'Personnel.Nominate', 'Agent', 'ORG-AFF', 'Justice.Convict', 'PER-SOC', 'Place', 'Personnel.Start-Position', 'Justice.Pardon', 'PER', 'Justice.Fine', 'Justice.Acquit', 'Person', 'Buyer', 'Justice.Execute', 'Seller', 'Adjudicator', 'Conflict.Attack', 'ART', 'Transaction.Transfer-Money', 'Contact.Phone-Write', 'Artifact', 'WEA', 'LOC', 'Personnel.End-Position', 'Org', 'Beneficiary', 'Justice.Release-Parole', 'Business.Start-Org', 'Vehicle', 'Instrument', 'Prosecutor', 'FAC', 'Plaintiff', 'Transaction.Transfer-Ownership', 'Movement.Transport', 'Justice.Arrest-Jail', 'ORG']
            #ACE_event
            # self.labels = ['Person', 'ORG', 'Vehicle', 'Plaintiff', 'Justice.Sue', 'r_Target', 'Personnel.End-Position', 'Life.Marry', 'VEH', 'r_Origin', 'Personnel.Nominate', 'r_Instrument', 'Giver', 'Contact.Phone-Write', 'GPE', 'Place', 'Conflict.Attack', 'Artifact', 'Business.Start-Org', 'r_Agent', 'r_Attacker', 'Agent', 'r_Buyer', 'r_Org', 'Justice.Convict', 'Justice.Extradite', 'Seller', 'Victim', 'r_Place', 'Instrument', 'PER', 'Attacker', 'Defendant', 'Personnel.Elect', 'Business.Declare-Bankruptcy', 'Justice.Arrest-Jail', 'r_Prosecutor', 'Recipient', 'WEA', 'Justice.Trial-Hearing', 'Justice.Pardon', 'Origin', 'Justice.Fine', 'r_Seller', 'Destination', 'Justice.Charge-Indict', 'Life.Injure', 'Entity', 'r_Artifact', 'Justice.Sentence', 'Org', 'Beneficiary', 'FAC', 'r_Adjudicator', 'r_Recipient', 'LOC', 'Contact.Meet', 'Transaction.Transfer-Money', 'r_Plaintiff', 'Justice.Release-Parole', 'Personnel.Start-Position', 'Justice.Acquit', 'Life.Be-Born', 'Transaction.Transfer-Ownership', 'Life.Divorce', 'r_Victim', 'r_Person', 'Conflict.Demonstrate', 'Target', 'r_Defendant', 'r_Destination', 'Justice.Execute', 'r_Giver', 'Business.End-Org', 'Buyer', 'Justice.Appeal', 'r_Entity', 'Movement.Transport', 'Business.Merge-Org', 'Adjudicator', 'Prosecutor', 'Life.Die', 'r_Beneficiary', 'r_Vehicle']       
            #1224
            self.labels = ['Personnel.End-Position', 'PER-SOC', 'r_Prosecutor', 'r_Adjudicator', 'Business.Merge-Org', 'Org', 'r_PART-WHOLE', 'ART', 'Conflict.Demonstrate', 'Attacker', 'r_Vehicle', 'Justice.Arrest-Jail', 'PART-WHOLE', 'Conflict.Attack', 'Person', 'r_Giver', 'Business.Declare-Bankruptcy', 'r_Plaintiff', 'r_Target', 'Justice.Charge-Indict', 'r_Buyer', 'Defendant', 'r_Defendant', 'Justice.Trial-Hearing', 'Personnel.Nominate', 'r_Entity', 'Justice.Appeal', 'Justice.Extradite', 'Transaction.Transfer-Ownership', 'Justice.Sue', 'ORG-AFF', 'r_Person', 'r_GEN-AFF', 'Contact.Meet', 'Entity', 'WEA', 'r_Place', 'r_Attacker', 'Plaintiff', 'r_Beneficiary', 'Justice.Sentence', 'Life.Divorce', 'r_ART', 'Contact.Phone-Write', 'Movement.Transport', 'Business.Start-Org', 'VEH', 'r_Org', 'Beneficiary', 'Personnel.Elect', 'Justice.Execute', 'LOC', 'Justice.Release-Parole', 'Buyer', 'PHYS', 'Justice.Fine', 'Personnel.Start-Position', 'Giver', 'Life.Marry', 'r_PER-SOC', 'r_Origin', 'r_Instrument', 'Agent', 'Target', 'Justice.Acquit', 'Justice.Convict', 'r_Victim', 'r_Artifact', 'Origin', 'Vehicle', 'Justice.Pardon', 'r_Seller', 'ORG', 'Business.End-Org', 'r_ORG-AFF', 'Destination', 'Prosecutor', 'Seller', 'Life.Be-Born', 'GEN-AFF', 'r_Recipient', 'Adjudicator', 'Recipient', 'Life.Injure', 'r_Agent', 'PER', 'Transaction.Transfer-Money', 'Artifact', 'FAC', 'r_Destination', 'Life.Die', 'r_PHYS', 'Instrument', 'GPE', 'Place', 'Victim']
        elif dataset=='JSON_ACE':
            # self.labels=['Transaction:Transfer-Ownership', 'Region-International', 'Individual', 'Business:Declare-Bankruptcy', 'Plaintiff', 'Underspecified', 'Agent', 'Exploding', 'Sharp', 'Justice:Appeal', 'Life:Injure', 'Educational', 'Boundary', 'Shooting', 'Beneficiary', 'Org', 'Biological', 'Life:Die', 'Conflict:Demonstrate', 'Air', 'ART:User-Owner-Inventor-Manufacturer', 'Water-Body', 'Subarea-Vehicle', 'Nuclear', 'Instrument', 'Destination', 'Celestial', 'Commercial', 'Justice:Pardon', 'Business:Merge-Org', 'PER-SOC:Family', 'GEN-AFF:Org-Location', 'Defendant', 'Sports', 'Religious', 'Justice:Release-Parole', 'Life:Marry', 'Justice:Sue', 'Personnel:Elect', 'Indeterminate', 'Personnel:Nominate', 'Land', 'Subarea-Facility', 'Entertainment', 'Personnel:Start-Position', 'ORG-AFF:Sports-Affiliation', 'Airport', 'Transaction:Transfer-Money', 'PER-SOC:Lasting-Personal', 'Business:Start-Org', 'Non-Governmental', 'Origin', 'Special', 'Person', 'Justice:Fine', 'Justice:Trial-Hearing', 'Justice:Charge-Indict', 'Projectile', 'GEN-AFF:Citizen-Resident-Religion-Ethnicity', 'Movement:Transport', 'PART-WHOLE:Artifact', 'PHYS:Near', 'Government', 'Contact:Phone-Write', 'Population-Center', 'Entity', 'PHYS:Located', 'Region-General', 'Vehicle', 'Building-Grounds', 'Address', 'Path', 'Attacker', 'Group', 'Water', 'PART-WHOLE:Subsidiary', 'PART-WHOLE:Geographical', 'County-or-District', 'Land-Region-Natural', 'Blunt', 'Place', 'PER-SOC:Business', 'Giver', 'Life:Divorce', 'Life:Be-Born', 'GPE-Cluster', 'Personnel:End-Position', 'Prosecutor', 'Justice:Sentence', 'Artifact', 'Seller', 'Conflict:Attack', 'ORG-AFF:Founder', 'Adjudicator', 'ORG-AFF:Membership', 'ORG-AFF:Ownership', 'Buyer', 'Justice:Execute', 'Justice:Extradite', 'Justice:Convict', 'Continent', 'ORG-AFF:Investor-Shareholder', 'Medical-Science', 'Victim', 'Recipient', 'Justice:Acquit', 'Business:End-Org', 'State-or-Province', 'Plant', 'Media', 'Target', 'ORG-AFF:Student-Alum', 'ORG-AFF:Employment', 'Contact:Meet', 'Nation', 'Chemical', 'Justice:Arrest-Jail']
            # self.labels = ['Target', 'Justice:Sentence', 'r_Buyer', 'Entity', 'Buyer', 'Path', 'Conflict:Attack', 'Justice:Appeal', 'Blunt', 'Region-General', 'r_Prosecutor', 'Justice:Sue', 'ORG-AFF:Employment', 'Subarea-Vehicle', 'Transaction:Transfer-Money', 'PHYS:Located', 'r_Victim', 'Chemical', 'r_ORG-AFF:Ownership', 'Justice:Charge-Indict', 'r_ORG-AFF:Founder', 'r_PART-WHOLE:Artifact', 'Beneficiary', 'Prosecutor', 'Justice:Arrest-Jail', 'Attacker', 'Entertainment', 'Artifact', 'Educational', 'PART-WHOLE:Artifact', 'Justice:Acquit', 'Movement:Transport', 'Place', 'r_Entity', 'Sports', 'Personnel:Elect', 'Life:Injure', 'Recipient', 'Boundary', 'r_PHYS:Near', 'Victim', 'r_Place', 'Justice:Trial-Hearing', 'PER-SOC:Family', 'Airport', 'PHYS:Near', 'Justice:Fine', 'r_PHYS:Located', 'r_PER-SOC:Family', 'Giver', 'GEN-AFF:Org-Location', 'r_ORG-AFF:Membership', 'Sharp', 'r_ORG-AFF:Investor-Shareholder', 'r_Plaintiff', 'r_PER-SOC:Business', 'r_Giver', 'Contact:Phone-Write', 'Region-International', 'Life:Divorce', 'Water', 'r_PER-SOC:Lasting-Personal', 'Conflict:Demonstrate', 'Seller', 'Religious', 'Org', 'ORG-AFF:Investor-Shareholder', 'r_Instrument', 'Special', 'ORG-AFF:Student-Alum', 'Shooting', 'Land-Region-Natural', 'r_Person', 'r_Defendant', 'Life:Die', 'Land', 'ORG-AFF:Membership', 'GEN-AFF:Citizen-Resident-Religion-Ethnicity', 'Indeterminate', 'r_ORG-AFF:Employment', 'r_GEN-AFF:Org-Location', 'County-or-District', 'ART:User-Owner-Inventor-Manufacturer', 'Agent', 'Address', 'Plant', 'Business:Declare-Bankruptcy', 'Nuclear', 'Destination', 'r_Adjudicator', 'Personnel:End-Position', 'Defendant', 'Vehicle', 'Government', 'Projectile', 'GPE-Cluster', 'r_Origin', 'r_Target', 'Transaction:Transfer-Ownership', 'r_Artifact', 'Non-Governmental', 'PER-SOC:Lasting-Personal', 'r_Org', 'Water-Body', 'ORG-AFF:Ownership', 'Commercial', 'r_PART-WHOLE:Geographical', 'Life:Marry', 'r_ORG-AFF:Student-Alum', 'Origin', 'Contact:Meet', 'r_Destination', 'r_GEN-AFF:Citizen-Resident-Religion-Ethnicity', 'r_ART:User-Owner-Inventor-Manufacturer', 'Exploding', 'Justice:Release-Parole', 'PER-SOC:Business', 'Medical-Science', 'State-or-Province', 'Plaintiff', 'Continent', 'Group', 'Population-Center', 'Justice:Execute', 'Nation', 'r_ORG-AFF:Sports-Affiliation', 'Personnel:Nominate', 'Instrument', 'Life:Be-Born', 'r_Recipient', 'Adjudicator', 'r_Agent', 'Building-Grounds', 'r_Beneficiary', 'ORG-AFF:Sports-Affiliation', 'Business:Merge-Org', 'Business:Start-Org', 'Person', 'Business:End-Org', 'Media', 'r_Attacker', 'r_Vehicle', 'Air', 'Justice:Pardon', 'Celestial', 'Underspecified', 'Personnel:Start-Position', 'PART-WHOLE:Geographical', 'r_Seller', 'Biological', 'r_PART-WHOLE:Subsidiary', 'ORG-AFF:Founder', 'Justice:Convict', 'Justice:Extradite', 'PART-WHOLE:Subsidiary', 'Subarea-Facility', 'Individual']
            #240801
            # self.labels = ['Media', 'Group', 'Exploding', 'Buyer', 'Life:Marry', 'Airport', 'Religious', 'PER-SOC', 'Subarea-Vehicle', 'Destination', 'Prosecutor', 'Justice:Fine', 'Nuclear', 'Entertainment', 'Org', 'Plant', 'Population-Center', 'Water-Body', 'Justice:Sue', 'Shooting', 'Justice:Execute', 'ORG-AFF', 'Commercial', 'PART-WHOLE', 'Defendant', 'Conflict:Demonstrate', 'Justice:Pardon', 'Special', 'Attacker', 'Non-Governmental', 'Person', 'Medical-Science', 'Origin', 'Life:Injure', 'Projectile', 'State-or-Province', 'Subarea-Facility', 'Individual', 'Justice:Extradite', 'Justice:Acquit', 'Business:Declare-Bankruptcy', 'Justice:Convict', 'Government', 'Justice:Arrest-Jail', 'Underspecified', 'County-or-District', 'Agent', 'Place', 'Contact:Meet', 'Personnel:Nominate', 'Seller', 'Artifact', 'Address', 'Life:Divorce', 'Justice:Charge-Indict', 'Continent', 'Business:Start-Org', 'Victim', 'GEN-AFF', 'Biological', 'Region-International', 'Beneficiary', 'Sharp', 'Justice:Trial-Hearing', 'Plaintiff', 'Land', 'Entity', 'Justice:Release-Parole', 'Justice:Appeal', 'Instrument', 'Path', 'Movement:Transport', 'Life:Be-Born', 'Recipient', 'Educational', 'PHYS', 'Region-General', 'Land-Region-Natural', 'Transaction:Transfer-Ownership', 'Transaction:Transfer-Money', 'Celestial', 'ART', 'Contact:Phone-Write', 'Blunt', 'Business:Merge-Org', 'Chemical', 'Vehicle', 'Nation', 'Business:End-Org', 'Personnel:Elect', 'Giver', 'Personnel:End-Position', 'Life:Die', 'Boundary', 'Target', 'Adjudicator', 'GPE-Cluster', 'Air', 'Water', 'Indeterminate', 'Personnel:Start-Position', 'Conflict:Attack', 'Sports', 'Building-Grounds', 'Justice:Sentence']
            #ACE_event
            # self.labels = ['Exploding', 'Water', 'Justice:Release-Parole', 'Attacker', 'Justice:Pardon', 'Conflict:Attack', 'Region-International', 'Life:Die', 'Religious', 'r_Vehicle', 'Chemical', 'r_PHYS:Near', 'r_Victim', 'r_Plaintiff', 'r_PER-SOC:Lasting-Personal', 'Celestial', 'Plant', 'r_Instrument', 'r_ORG-AFF:Founder', 'Water-Body', 'r_ART:User-Owner-Inventor-Manufacturer', 'Business:Start-Org', 'Life:Injure', 'Vehicle', 'ORG-AFF:Founder', 'Shooting', 'r_ORG-AFF:Ownership', 'r_Prosecutor', 'r_PART-WHOLE:Geographical', 'r_PER-SOC:Business', 'Biological', 'r_Adjudicator', 'Org', 'Medical-Science', 'Nuclear', 'r_PART-WHOLE:Subsidiary', 'Special', 'Person', 'PART-WHOLE:Artifact', 'Land-Region-Natural', 'r_Beneficiary', 'ORG-AFF:Sports-Affiliation', 'Life:Marry', 'PART-WHOLE:Geographical', 'Justice:Extradite', 'GEN-AFF:Org-Location', 'Personnel:Nominate', 'r_Artifact', 'Building-Grounds', 'Entertainment', 'r_ORG-AFF:Sports-Affiliation', 'Airport', 'PART-WHOLE:Subsidiary', 'Business:End-Org', 'GEN-AFF:Citizen-Resident-Religion-Ethnicity', 'Justice:Trial-Hearing', 'Business:Declare-Bankruptcy', 'Government', 'r_Buyer', 'PER-SOC:Lasting-Personal', 'Commercial', 'Personnel:Elect', 'r_Destination', 'Justice:Acquit', 'Justice:Fine', 'Subarea-Vehicle', 'Sharp', 'r_PART-WHOLE:Artifact', 'Indeterminate', 'r_Seller', 'Educational', 'County-or-District', 'r_ORG-AFF:Investor-Shareholder', 'Justice:Convict', 'r_Place', 'Continent', 'r_Person', 'r_Giver', 'Transaction:Transfer-Ownership', 'r_Agent', 'ORG-AFF:Membership', 'Buyer', 'Justice:Appeal', 'ORG-AFF:Employment', 'Subarea-Facility', 'Transaction:Transfer-Money', 'Place', 'Artifact', 'Justice:Sentence', 'Giver', 'GPE-Cluster', 'Seller', 'Justice:Execute', 'Nation', 'Defendant', 'r_PER-SOC:Family', 'ORG-AFF:Student-Alum', 'r_Origin', 'PER-SOC:Family', 'Contact:Phone-Write', 'r_PHYS:Located', 'Boundary', 'PER-SOC:Business', 'Beneficiary', 'Individual', 'r_GEN-AFF:Citizen-Resident-Religion-Ethnicity', 'Victim', 'Justice:Charge-Indict', 'PHYS:Near', 'Plaintiff', 'r_Target', 'r_ORG-AFF:Student-Alum', 'Conflict:Demonstrate', 'Destination', 'Adjudicator', 'Sports', 'Instrument', 'Entity', 'Agent', 'Underspecified', 'Prosecutor', 'Movement:Transport', 'r_Entity', 'Land', 'ORG-AFF:Ownership', 'Personnel:Start-Position', 'Justice:Sue', 'r_Defendant', 'State-or-Province', 'Blunt', 'Non-Governmental', 'Recipient', 'r_ORG-AFF:Membership', 'Justice:Arrest-Jail', 'r_Attacker', 'Region-General', 'Life:Be-Born', 'Business:Merge-Org', 'Target', 'r_Recipient', 'Path', 'r_GEN-AFF:Org-Location', 'r_ORG-AFF:Employment', 'ART:User-Owner-Inventor-Manufacturer', 'Group', 'Life:Divorce', 'Air', 'Contact:Meet', 'Media', 'r_Org', 'Projectile', 'ORG-AFF:Investor-Shareholder', 'Origin', 'Address', 'PHYS:Located', 'Population-Center', 'Personnel:End-Position']
            #1224
            self.labels = ['GPE', 'Plaintiff', 'Life:Die', 'r_Beneficiary', 'Conflict:Demonstrate', 'r_Place', 'Attacker', 'Justice:Sue', 'LOC', 'Business:Merge-Org', 'r_Destination', 'Justice:Release-Parole', 'Buyer', 'Life:Divorce', 'Life:Injure', 'r_Instrument', 'VEH', 'Business:Start-Org', 'Giver', 'PART-WHOLE', 'Transaction:Transfer-Ownership', 'PHYS', 'Prosecutor', 'r_Recipient', 'r_PHYS', 'Justice:Appeal', 'PER-SOC', 'Justice:Acquit', 'r_Seller', 'ART', 'Justice:Fine', 'r_Defendant', 'Place', 'Justice:Pardon', 'Contact:Meet', 'Origin', 'Movement:Transport', 'r_Agent', 'Justice:Trial-Hearing', 'Personnel:End-Position', 'Recipient', 'r_PER-SOC', 'Justice:Arrest-Jail', 'WEA', 'Justice:Convict', 'PER', 'r_Buyer', 'Person', 'ORG-AFF', 'r_ORG-AFF', 'Instrument', 'r_Prosecutor', 'Transaction:Transfer-Money', 'r_Entity', 'Adjudicator', 'r_Artifact', 'r_Giver', 'Business:End-Org', 'r_Target', 'Personnel:Nominate', 'ORG', 'r_Victim', 'Vehicle', 'Defendant', 'r_Origin', 'Victim', 'r_Org', 'Personnel:Start-Position', 'Seller', 'Personnel:Elect', 'Conflict:Attack', 'GEN-AFF', 'r_ART', 'r_Attacker', 'r_Vehicle', 'Target', 'Agent', 'r_Adjudicator', 'r_PART-WHOLE', 'Entity', 'r_Plaintiff', 'Beneficiary', 'FAC', 'Justice:Charge-Indict', 'Justice:Sentence', 'Artifact', 'Justice:Extradite', 'Contact:Phone-Write', 'r_Person', 'Org', 'Life:Be-Born', 'r_GEN-AFF', 'Life:Marry', 'Destination', 'Business:Declare-Bankruptcy', 'Justice:Execute']
        elif dataset == 'ACE_ADD_EVENT':
            self.labels = ['r_Defendant', 'Victim', 'Buyer', 'Justice:Fine', 'Target', 'Justice:Charge-Indict', 'r_Entity', 'r_Attacker', 'Conflict:Attack', 'Business:End-Org', 'Instrument', 'r_Buyer', 'Business:Start-Org', 'Defendant', 'r_Beneficiary', 'Justice:Sue', 'Recipient', 'PER', 'Artifact', 'Place', 'Justice:Execute', 'Plaintiff', 'Personnel:Nominate', 'Contact:Meet', 'Life:Divorce', 'Destination', 'Transaction:Transfer-Money', 'Life:Be-Born', 'r_Seller', 'r_Origin', 'Life:Injure', 'Justice:Acquit', 'Entity', 'Personnel:End-Position', 'Attacker', 'r_Agent', 'WEA', 'r_Place', 'Life:Marry', 'Personnel:Elect', 'Transaction:Transfer-Ownership', 'Contact:Phone-Write', 'Origin', 'ORG', 'r_Vehicle', 'Beneficiary', 'r_Giver', 'r_Instrument', 'r_Victim', 'r_Recipient', 'Justice:Release-Parole', 'Business:Declare-Bankruptcy', 'Person', 'r_Plaintiff', 'Adjudicator', 'Justice:Pardon', 'Org', 'Personnel:Start-Position', 'Business:Merge-Org', 'Justice:Appeal', 'Agent', 'r_Artifact', 'Giver', 'Vehicle', 'Conflict:Demonstrate', 'Prosecutor', 'r_Person', 'VEH', 'GPE', 'r_Prosecutor', 'Justice:Arrest-Jail', 'Movement:Transport', 'r_Target', 'r_Org', 'r_Adjudicator', 'Justice:Trial-Hearing', 'Life:Die', 'Justice:Convict', 'FAC', 'LOC', 'Seller', 'Justice:Sentence', 'Justice:Extradite', 'r_Destination']
        elif dataset == 'ACE_ADD_RELATION':
            self.labels = ['ORG-AFF', 'WEA', 'r_PHYS', 'r_PART-WHOLE', 'r_ART', 'ORG', 'PHYS', 'VEH', 'ART', 'GPE', 'GEN-AFF', 'r_ORG-AFF', 'PER-SOC', 'PER', 'PART-WHOLE', 'r_GEN-AFF', 'LOC', 'r_PER-SOC', 'FAC']
        elif dataset == 'ACE_ADD_ENTITY':
            self.labels = ['ORG', 'LOC', 'WEA', 'GPE', 'PER', 'FAC', 'VEH']
        elif dataset == 'ACE_RELATION':
            self.labels = ['r_PART-WHOLE', 'GPE', 'WEA', 'r_GEN-AFF', 'r_ORG-AFF', 'FAC', 'ART', 'r_PHYS', 'r_PER-SOC', 'PER-SOC', 'PART-WHOLE', 'r_ART', 'VEH', 'ORG', 'ORG-AFF', 'PHYS', 'PER', 'LOC', 'GEN-AFF']
        elif dataset == 'ACE_EVENT':
            self.labels = ['r_Giver', 'r_Artifact', 'Seller', 'Target', 'r_Beneficiary', 'Business.End-Org', 'Justice.Release-Parole', 'Personnel.Nominate', 'Justice.Sentence', 'Destination', 'Life.Marry', 'Victim', 'Life.Be-Born', 'Agent', 'Attacker', 'r_Buyer', 'r_Vehicle', 'Justice.Arrest-Jail', 'r_Place', 'Justice.Execute', 'r_Org', 'Org', 'r_Origin', 'Plaintiff', 'Origin', 'Transaction.Transfer-Money', 'Justice.Pardon', 'Beneficiary', 'r_Seller', 'r_Entity', 'r_Instrument', 'Life.Injure', 'Recipient', 'Business.Start-Org', 'FAC', 'Life.Divorce', 'Movement.Transport', 'r_Defendant', 'r_Plaintiff', 'Justice.Sue', 'Justice.Appeal', 'Justice.Fine', 'Conflict.Demonstrate', 'VEH', 'Giver', 'Instrument', 'Justice.Extradite', 'LOC', 'Entity', 'Personnel.Elect', 'r_Agent', 'Contact.Meet', 'Business.Declare-Bankruptcy', 'r_Victim', 'Personnel.End-Position', 'Prosecutor', 'r_Adjudicator', 'Justice.Charge-Indict', 'Adjudicator', 'r_Person', 'Transaction.Transfer-Ownership', 'Buyer', 'Place', 'GPE', 'r_Target', 'Personnel.Start-Position', 'Artifact', 'Justice.Trial-Hearing', 'r_Destination', 'Contact.Phone-Write', 'r_Attacker', 'Business.Merge-Org', 'Person', 'ORG', 'r_Recipient', 'Justice.Convict', 'WEA', 'Justice.Acquit', 'PER', 'Conflict.Attack', 'r_Prosecutor', 'Life.Die', 'Defendant', 'Vehicle']
        elif dataset == 'ACE_ENTITY':
            self.labels = ['FAC', 'GPE', 'VEH', 'ORG', 'WEA', 'PER', 'LOC']
        elif dataset == 'conll04':
            self.labels = ['people', 'location', 'organization', 'other', 'located-in', 'organization-in', 'live-in', 'work-for', 'kill', 'r_organization-in', 'r_located-in', 'r_work-for', 'r_kill', 'r_live-in']
        elif dataset == 'nyt':
            self.labels = ['location', 'person', 'organization', 'place-of-birth', 'country', 'major-shareholder-of', 'capital', 'ethnicity', 'teams', 'industry', 'people', 'major-shareholders', 'founders', 'profession', 'advisors', 'religion', 'contains', 'children', 'neighborhood-of', 'place-founded', 'nationality', 'place-of-death', 'company', 'location', 'geographic-distribution', 'place-lived', 'administrative-divisions', 'r_contains', 'r_country', 'r_children', 'r_administrative-divisions', 'r_capital', 'r_company', 'r_place-of-death', 'r_place-of-birth', 'r_nationality', 'r_founders', 'r_neighborhood-of', 'r_place-lived', 'r_advisors', 'r_location', 'r_place-founded', 'r_major-shareholders', 'r_major-shareholder-of', 'r_teams', 'r_religion', 'r_geographic-distribution', 'r_people', 'r_ethnicity']
        elif dataset == 'scierc':
            self.labels = ['method', 'task', 'other-scientific-term', 'metric', 'material', 'generic', 'evaluate-for', 'compare', 'used-for', 'feature-of', 'conjunction', 'part-of', 'hyponym-of', 'r_used-for', 'r_feature-of', 'r_evaluate-for', 'r_conjunction', 'r_hyponym-of', 'r_part-of', 'r_compare']
        elif dataset == 'casie':
            self.labels = ['geopolitical-entity', 'time', 'file', 'website', 'data', 'common-vulnerabilities-and-exposures', 'money', 'patch', 'malware', 'person', 'purpose', 'number', 'vulnerability', 'version', 'capabilities', 'payment-method', 'system', 'software', 'personally-identifiable-information', 'organization', 'device', 'patch-number', 'patch', 'number-of-data', 'discoverer', 'common-vulnerabilities-and-exposures', 'vulnerable-system', 'vulnerable-system-version', 'supported-platform', 'payment-method', 'issues-addressed', 'purpose', 'damage-amount', 'attacker', 'releaser', 'tool', 'number-of-victim', 'trusted-entity', 'capabilities', 'place', 'attack-pattern', 'compromised-data', 'price', 'victim', 'vulnerable-system-owner', 'vulnerability', 'time', 'r_tool', 'r_trusted-entity', 'r_attack-pattern', 'r_victim', 'r_attacker', 'r_time', 'r_place', 'r_vulnerable-system-owner', 'r_discoverer', 'r_releaser', 'r_vulnerability', 'r_vulnerable-system', 'r_vulnerable-system-version', 'r_patch', 'r_compromised-data', 'r_number-of-victim', 'r_purpose', 'r_common-vulnerabilities-and-exposures', 'r_number-of-data', 'r_price', 'r_payment-method', 'r_capabilities', 'r_patch-number', 'r_damage-amount', 'r_issues-addressed', 'r_supported-platform', 'phishing', 'databreach', 'ransom', 'discover-vulnerability', 'patch-vulnerability']
        elif dataset == 'cadec':
            self.labels = ['NA', 'adverse-drug-reaction', 'r_adverse-drug-reaction']
        elif dataset == 'absa':
            self.labels = ['opinion', 'aspect', 'negative', 'neutral', 'positive', 'r_positive', 'r_negative', 'r_neutral']
        elif dataset == 'ace2004':
            self.labels = ['GPE', 'ORG', 'PER', 'FAC', 'VEH', 'LOC', 'WEA']
        elif dataset == '14lap':
            self.labels = ["opinion", "aspect", "negative", "neutral", "positive", "r_positive", "r_negative", "r_neutral"]
        elif datasets == '14res':
            self.labels = ["opinion", "aspect", "negative", "neutral", "positive", "r_positive", "r_negative", "r_neutral"]
        elif dataset == '15res':
            self.labels = ["opinion", "aspect", "negative", "neutral", "positive", "r_positive", "r_negative", "r_neutral"]
        elif dataset == '16res':
            self.labels = ["opinion", "aspect", "negative", "neutral", "positive", "r_positive", "r_negative", "r_neutral"]
        else:
            raise NotImplementedError()

        if dataset == "ACE05" or dataset == "GENIA" or dataset == "ACE04" or dataset == "CONLL" \
                or dataset=='WEIBO_++' or dataset=='WEIBO_++MERGE' or dataset=='YOUKU' or dataset=='ECOMMERCE' \
                or dataset=='ACE' or dataset=='JSON_ACE':
            self.interval = 4
        else:
            self.interval = 4
            # raise NotImplementedError()

        self.trained_model_path=trained_models
        print(self.trained_model_path)
        if dataset == "CONLL":
            self.true_mask='CONLL'
            # print('加载模型')
            # sys.path.append("..")
            # from bert_en.model import Net
            # self.trained_model=Net(False,10,'cuda',False).cuda()
            # self.trained_model=nn.DataParallel(self.trained_model)
            # self.trained_model.load_state_dict(torch.load(self.trained_model_path))
            # self.trained_model.eval()
            # self.trained_tokenizer=BertTokenizer.from_pretrained('./bert-base-cased')
        elif dataset=='WEIBO_++':
            self.true_mask='WEIBO_++'
            print('加载模型')
            # sys.path.append("..") 
            sys.path.append("..")
            from bert_ch.model import Net
            self.trained_model=Net(False,16,'cuda',False).cuda()
            self.trained_model=nn.DataParallel(self.trained_model)
            self.trained_model.load_state_dict(torch.load(self.trained_model_path))
            self.trained_model.eval()
            self.trained_tokenizer=BertTokenizer.from_pretrained('./bert-chinese-wwm')
        elif dataset=='WEIBO_++MERGE':
            self.true_mask='WEIBO_++MERGE'
            print('加载模型')
            # sys.path.append("..") 
            sys.path.append("..")
            from bert_ch.model import Net
            self.trained_model=Net(False,10,'cuda',False).cuda()
            self.trained_model=nn.DataParallel(self.trained_model)
            self.trained_model.load_state_dict(torch.load(self.trained_model_path))
            self.trained_model.eval()
            self.trained_tokenizer=BertTokenizer.from_pretrained('./bert-chinese-wwm')
        elif dataset=='YOUKU':
            self.true_mask='YOUKU'
            print('加载模型')
            # sys.path.append("..") 
            sys.path.append("..")
            from bert_ch.model import Net
            self.trained_model=Net(False,11,'cuda',False).cuda()
            self.trained_model=nn.DataParallel(self.trained_model)
            self.trained_model.load_state_dict(torch.load(self.trained_model_path))
            self.trained_model.eval()
            self.trained_tokenizer=BertTokenizer.from_pretrained('./bert-chinese-wwm')
        elif dataset=='ECOMMERCE':
            self.true_mask='ECOMMERCE'
            print('加载模型')
            # sys.path.append("..") 
            sys.path.append("..")
            from bert_ch.model import Net
            self.trained_model=Net(False,10,'cuda',False).cuda()
            self.trained_model=nn.DataParallel(self.trained_model)
            self.trained_model.load_state_dict(torch.load(self.trained_model_path))
            self.trained_model.eval()
            self.trained_tokenizer=BertTokenizer.from_pretrained('./bert-chinese-wwm')
        else:
            self.true_mask='ACE'
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
            full_label = lines[i + 2]
            label = lines[i + 2]

            examples.append(
                InputExample(guid=len(examples), text_a=text_a, pos=None, label=label, full_label = full_label))
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

            partial_masks, eval_masks = self.generate_partial_masks(example.text_a.split(' '), max_seq_length, example.label,
                                                            self.labels, example.full_label)
            # torch.set_printoptions(profile="full")
            # print()
            # print(partial_masks)
            # stop
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
                # self.logger.info(
                #     "eval_masks: %s" % " ".join([str(x) for x in eval_masks]))
                # self.logger.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              eval_masks=eval_masks))

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

    def get_probs(self,x,is_heads,y):
        '''11.22lwc:获取一条句子的预测概率，batch设置成1，不需要负采样'''
        # print('开始预测')
        # print(x)
        # print(y)
        with torch.no_grad():
            if self.true_mask=='CONLL':
                _,logits,y,y_hat=self.trained_model(x.cuda(),y.cuda())
            else:
                logits,y,y_hat=self.trained_model(x.cuda(),y.cuda())
            logits=torch.softmax(logits,-1)
            # probs,_=torch.max(logits,-1)
            # print(logits.cpu())
            # print(is_heads)
            probs=[logit.numpy().tolist() for logit,is_head in zip(logits.cpu()[0],is_heads[0]) if is_head==1]
            # print(len(x[0]),len(y[0]),len(probs.cpu().numpy().tolist()[0]))
            # print(probs.cpu().numpy().tolist()[0])
            # stop
        # print('保存概率')
        # with open(save_path,'wb') as fw:
        #     pickle.dump(predict_probs,fw)
        return probs

    def generate_partial_masks_v1(self, tokens, max_seq_length, labels, tags):
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
            
        #初始化所有节点为隐藏节点,生成评估矩阵
        mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection

        for start, end, tag in label_list:
            #初始化观测到的节点,使用没被tokenizer的原始token
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
                            if mask[i][j][k]!=1:
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
        
        #初始化所有节点为隐藏节点,生成掩码矩阵
        true_mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection

        if self.true_mask=='CONLL' or self.true_mask=='ECOMMERCE' or self.true_mask=='WEIBO_++' \
            or self.true_mask=='WEIBO_++MERGE' or self.true_mask=='YOUKU':
            if self.true_mask=='CONLL':
                bio_labels=['<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG']
            elif self.true_mask=='ECOMMERCE':
                bio_labels=['<PAD>', 'B-HCCX', 'I-HPPX', 'O', 'I-XH', 'I-HCCX', 'B-XH', 'I-MISC', 'B-HPPX', 'B-MISC']
            elif self.true_mask=='WEIBO_++':
                bio_labels=['<PAD>', 'B-LOC.NAM', 'I-GPE.NAM', 'I-ORG.NOM', 'I-LOC.NOM', 'B-ORG.NAM', 'I-LOC.NAM', 'B-GPE.NAM', 'I-ORG.NAM', 'B-LOC.NOM', 'I-PER.NAM', 'B-PER.NAM', 'B-PER.NOM', 'O', 'B-ORG.NOM', 'I-PER.NOM']
            elif self.true_mask=='WEIBO_++MERGE':
                bio_labels=['<PAD>', 'I-LOC', 'B-ORG', 'B-PER', 'I-GPE', 'I-PER', 'B-LOC', 'O', 'B-GPE', 'I-ORG']
            elif self.true_mask=='YOUKU':
                bio_labels=['<PAD>', 'E-PER', 'B-PER', 'I-PER', 'E-NUM', 'B-TV', 'I-NUM', 'O', 'I-TV', 'B-NUM', 'E-TV']
            
            is_heads=[]
            input_ids=[]
            count=0
            for tok in tokens:
                toks=self.trained_tokenizer.tokenize(tok)
                if count==0:
                    input_ids=torch.tensor(self.trained_tokenizer.convert_tokens_to_ids(toks))
                    is_heads=torch.tensor([1]+[0]*(len(toks)-1))
                else:
                    input_ids=torch.cat((input_ids,torch.tensor(self.trained_tokenizer.convert_tokens_to_ids(toks))),0)
                    is_heads=torch.cat((is_heads,torch.tensor([1]+[0]*(len(toks)-1))),0)
                count+=1
            # print(input_ids)
            # print(is_heads)
            probs=self.get_probs(input_ids.unsqueeze(0),is_heads.unsqueeze(0),is_heads.unsqueeze(0))

            for start, end, tag in label_list:
                #初始化观测到的节点,使用没被tokenizer的原始token
                if start < max_seq_length and end < max_seq_length:
                    probs_sum=0
                    for k in range(total_tags_num):
                        # if k != tag_idx:
                        if k<len(tags) and 'B-'+tags[k] in bio_labels and 'I-'+tags[k] in bio_labels:
                            start_bio_idx=bio_labels.index('B-'+tags[k])
                            end_bio_idx=bio_labels.index('I-'+tags[k])
                            # print(probs[start][start_bio_idx])
                            true_mask[start][end][k] = probs[start][start_bio_idx]+probs[start][end_bio_idx]
                            probs_sum+=true_mask[start][end][k]
                            # print(probs[start][k])
                        else:
                            true_mask[start][end][k]=1-probs_sum

                #交叉边界
                for i in range(l):
                    if i > end:
                        continue
                    for j in range(i, l):
                        if j < start:
                            continue
                        if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
                            for k in range(total_tags_num):
                                true_mask[i][j][k] = 0

            #终点大于起点，（2，0）
            for i in range(l):
                for j in range(0, i):
                    for k in range(total_tags_num):
                        true_mask[i][j][k] = 0

            #L0=0,L1=1
            for i in range(l):
                for j in range(i, l):
                    probs_sum=0
                    for k in range(total_tags_num):
                        if true_mask[i][j][k] == 2:
                            if k < len(tags) and 'B-'+tags[k] in bio_labels and 'I-'+tags[k] in bio_labels:
                                start_bio_idx=bio_labels.index('B-'+tags[k])
                                end_bio_idx=bio_labels.index('I-'+tags[k])
                                true_mask[i][j][k] = probs[start][start_bio_idx]+probs[start][end_bio_idx]
                                probs_sum+=true_mask[i][j][k]
                            else:
                                #可能在其他标签集中
                                true_mask[i][j][k] = 1-probs_sum
            #将无用的拒绝
            for i in range(max_seq_length):
                for j in range(max_seq_length):
                    for k in range(total_tags_num):
                        if true_mask[i][j][k] == 2:
                            true_mask[i][j][k] = 0
        else:
            true_mask=mask
        # print(true_mask==mask)
        # for i in range(len(mask)):
        #     for j in range(len(mask[i])):
        #         for k in range(len(mask[i][j])):
        #             if mask[i][j][k]==1:
        #                 if k<len(tags):
        #                     print(i,j,tags[k])
        # stop
        #mask是评估矩阵,true_mask是掩码矩阵

        return mask,true_mask
    
    def generate_partial_masks(self, tokens, max_seq_length, labels, tags, full_labels):
        #所有标签
        total_tags_num = len(tags) + self.latent_size
        #L0，已知标签序列
        labels = labels.split('|')
        label_list = list()
        full_labels = full_labels.split('|')
        full_label_list= []

        for label in labels:
            if not label:
                continue
            sp = label.strip().split(' ')
            start, end = sp[0].split(',')[:2]
            start = int(start)
            end = int(end) - 1
            label_list.append((start, end, sp[1]))

        for label in full_labels:
            if not label:
                continue
            sp = label.strip().split(' ')
            start, end = sp[0].split(',')[:2]
            start = int(start)
            end = int(end) - 1
            full_label_list.append((start, end, sp[1]))

        #初始化所有节点为隐藏节点
        mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        full_mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection

        for start, end, tag in label_list:
            #初始化观测到的节点,使用没被tokenizer的原始token
            if start < max_seq_length and end < max_seq_length:
                # print(tags)
                tag_idx = tags.index(tag)
                mask[start][end][tag_idx] = 1
                for k in range(total_tags_num):
                    if k != tag_idx:
                        mask[start][end][k] = 0
            # #交叉边界
            # for i in range(l):
            #     if i > end:
            #         continue
            #     for j in range(i, l):
            #         if j < start:
            #             continue
            #         if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
            #             for k in range(total_tags_num):
            #                 mask[i][j][k] = 0

        for start, end, tag in full_label_list:
            #初始化观测到的节点,使用没被tokenizer的原始token
            if start < max_seq_length and end < max_seq_length:
                tag_idx = tags.index(tag)
                full_mask[start][end][tag_idx] = 1
                for k in range(total_tags_num):
                    if k != tag_idx:
                        full_mask[start][end][k] = 0
            # #交叉边界
            # for i in range(l):
            #     if i > end:
            #         continue
            #     for j in range(i, l):
            #         if j < start:
            #             continue
            #         if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
            #             for k in range(total_tags_num):
            #                 full_mask[i][j][k] = 0

        #终点大于起点，（2，0）
        for i in range(l):
            for j in range(0, i):
                for k in range(total_tags_num):
                    mask[i][j][k] = 0
                    full_mask[i][j][k] = 0

        all_mask_count = 0
        all_full_mask_count = 0
        #L0=0,L1=1
        for i in range(l):
            for j in range(i, l):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        all_mask_count += 1
                    if full_mask[i][j][k] == 2:
                        all_full_mask_count += 1
        
        # sample_mask = random.sample(list(range(0, all_mask_count)), min(len(label_list)+1, all_mask_count))
        # sample_full_mask = random.sample(list(range(0, all_full_mask_count)), min(len(label_list)+1, all_full_mask_count))

        # mask_start = 0
        # full_mask_start = 0
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
                    # if mask[i][j][k] == 2:
                    #     mask_start += 1
                    if full_mask[i][j][k] == 2:
                        if k < len(tags):
                            #不在观察序列中必然不在观测序列标签集中
                            full_mask[i][j][k] = 0
                        else:
                            #可能在其他标签集中
                            full_mask[i][j][k] = 1
                    # if full_mask[i][j][k] == 2:
                    #     full_mask_start += 1
        #将无用的拒绝
        full_mask_col = []
        full_mask_pos = []
        full_mask_label = []
        mask_col = []
        mask_pos = []
        mask_label = []
        for i in range(max_seq_length):
            for j in range(max_seq_length):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        mask[i][j][k] = 0
                    if full_mask[i][j][k] == 2:
                        full_mask[i][j][k] = 0
                    if full_mask[i][j][k] == 1:
                        full_mask_col.append(i)
                        full_mask_pos.append(j)
                        full_mask_label.append(k)
                    if mask[i][j][k] == 1:
                        mask_col.append(i)
                        mask_pos.append(j)
                        mask_label.append(k)
        # import numpy as np
        # torch.set_printoptions(threshold=np.inf)
        # print(mask[2][0])
        # stop
        full_mask = [full_mask_col,full_mask_pos,full_mask_label]
        mask = [mask_col,mask_pos,mask_label]
        
        return full_mask, mask


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

def eval_v0(args, outputs, partial_masks, label_size, gather_masks):

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

def eval_v1(args, outputs, partial_masks, label_size, gather_masks):

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
    label_size=len(label_size)

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
    print(pred_count)
    return correct, pred_count, gold_count,trigger_correct,trigger_pred_count,trigger_gold_count, \
            entity_correct,entity_preds_count,entity_gold_count,relation_correct,relation_preds_count,relation_gold_count, \
            event_correct,event_preds_count,event_gold_count, two_relation_correct,two_relation_preds_count,two_relation_gold_count, \
            two_event_correct,two_event_preds_count,two_event_gold_count

def eval_v3(args, outputs, partial_masks, labels, gather_masks, valid_pattern):
    '''
    relation和event
    (只统计relation：在span里面，使用gold entity判断预测的关系是否正确，
    测试的时候使用新的标注方法，标注所有可能的关系，只要预测对其中的一个就算正确，要去重

    relation和event(包含entity，
    预测的entity判断预测的关系是否正确

    '''
    # #ACE
    if args.dataset=='ACE':
        entity_types=['FAC', 'PER', 'VEH', 'GPE', 'WEA', 'LOC', 'ORG']
        relation_types=['Membership', 'Investor-Shareholder', 'Employment', 'Lasting-Personal', 'Founder', 'Student-Alum', 'Located', 'Business', 'Sports-Affiliation', 'Subsidiary', 'Ownership', 'Org-Location', 'Citizen-Resident-Religion-Ethnicity', 'Family', 'Near', 'User-Owner-Inventor-Manufacturer', 're_Artifact', 'Geographical']
        event_types=['Merge-Org', 'Release-Parole', 'Declare-Bankruptcy', 'Extradite', 'Marry', 'Transfer-Money', 'End-Org', 'Sentence', 'Start-Position', 'Fine', 'Acquit', 'End-Position', 'Injure', 'Sue', 'Transport', 'Die', 'Be-Born', 'Execute', 'Meet', 'Convict', 'Phone-Write', 'Appeal', 'Charge-Indict', 'Divorce', 'Demonstrate', 'Start-Org', 'Attack', 'Arrest-Jail', 'Transfer-Ownership', 'Trial-Hearing', 'Pardon', 'Elect', 'Nominate']
        role_types=['Beneficiary', 'Target', 'Adjudicator', 'Vehicle', 'Victim', 'eve_Artifact', 'Defendant', 'Prosecutor', 'Agent', 'Giver', 'Org', 'Place', 'Attacker', 'Origin', 'Person', 'Buyer', 'Destination', 'Plaintiff', 'Entity', 'Seller', 'Recipient', 'Instrument']
    elif args.dataset=='JSON_ACE':
        #ACE+
        entity_types=['Chemical', 'Blunt', 'Continent', 'Non-Governmental', 'Population-Center', 'Government', 'Educational', 'Airport', 'Region-International', 'Nuclear', 'Address', 'County-or-District', 'Path', 'Celestial', 'Water-Body', 'Land', 'Underspecified', 'Water', 'Air', 'Plant', 'Sports', 'Nation', 'Special', 'State-or-Province', 'Commercial', 'Projectile', 'Subarea-Facility', 'Building-Grounds', 'GPE-Cluster', 'Land-Region-Natural', 'Boundary', 'Religious', 'Indeterminate', 'Group', 'Sharp', 'Biological', 'Shooting', 'Entertainment', 'Region-General', 'Exploding', 'Individual', 'Subarea-Vehicle', 'Media', 'Medical-Science']
        relation_types=['Membership', 'Investor-Shareholder', 'Employment', 'Lasting-Personal', 'Founder', 'Located', 'Student-Alum', 'Business', 'Sports-Affiliation', 'Subsidiary', 'Ownership', 'Org-Location', 'Citizen-Resident-Religion-Ethnicity', 'Family', 'Near', 'User-Owner-Inventor-Manufacturer', 're_Artifact', 'Geographical']
        event_types=['Merge-Org', 'Release-Parole', 'Declare-Bankruptcy', 'Extradite', 'Marry', 'Transfer-Money', 'End-Org', 'Sentence', 'Start-Position', 'Fine', 'Acquit', 'End-Position', 'Injure', 'Sue', 'Transport', 'Die', 'Be-Born', 'Execute', 'Meet', 'Convict', 'Phone-Write', 'Appeal', 'Charge-Indict', 'Divorce', 'Demonstrate', 'Start-Org', 'Attack', 'Arrest-Jail', 'Transfer-Ownership', 'Trial-Hearing', 'Pardon', 'Elect', 'Nominate']
        role_types=['Beneficiary', 'Target', 'Adjudicator', 'Vehicle', 'Victim', 'Defendant', 'eve_Artifact', 'Prosecutor', 'Agent', 'Giver', 'Org', 'Place', 'Attacker', 'Origin', 'Person', 'Buyer', 'Destination', 'Plaintiff', 'Entity', 'Seller', 'Recipient', 'Instrument']

    # relation_types=[]
    label_size=len(labels)
    #trigger和role组成的，一个事件+一个实体
    role_ids=[labels.index(item) for item in role_types]
    entity_ids=[labels.index(item) for item in entity_types]
    #两个实体
    relation_ids=[labels.index(item) for item in relation_types]
    #trigger对应
    event_ids=[labels.index(item) for item in event_types]

    correct, pred_count, gold_count = 0, 0, 0
    role_preds_count,role_gold_count,role_correct=0,0,0
    relation_preds_count,relation_gold_count,relation_correct=0,0,0
    event_preds_count,event_gold_count,event_correct=0,0,0
    entity_preds_count,entity_gold_count,entity_correct=0,0,0
    two_relation_preds_count,two_relation_gold_count,two_relation_correct=0,0,0
    two_role_preds_count,two_role_gold_count,two_role_correct=0,0,0

    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()
    num=0

    # # print(partial_masks)
    # for partial_mask in partial_masks:
    #     for i in range(len(partial_mask)):
    #         for j in range(len(partial_mask[i])):
    #             for k in range(len(partial_mask[i][j])):
    #                 if partial_mask[i][j][k]==1:
    #                     if k<len(labels):
    #                         print(i,j,labels[k])
 
    for output, partial_mask, l in zip(outputs, partial_masks, gather_masks):
        # print(num)
        num+=1
        torch.set_printoptions(profile="full")
        import numpy as np
        import sys
        np.set_printoptions(threshold=sys.maxsize)

        golds = list()
        preds = list()
        #全部
        role_golds=list()
        role_preds=list()
        entity_golds=list()
        entity_preds=list()
        #全部
        relation_golds=list()
        relation_preds=list()
        event_golds=list()
        event_preds=list()
        two_relation_preds=[]
        two_relation_golds=[]
        two_role_golds=[]
        two_role_preds=[]
        #单一
        relation_golds_=[]
        #单一
        role_golds_=[]
        entity_tuples=[]
        event_tuples=[]
        preds_entity_tuples=[]
        preds_event_tuples=[]

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0:
                    if output[i][j] < label_size:
                        preds.append("{}_{}_{}".format(i, j, int(output[i][j])))

                    if output[i][j] in event_ids:
                        event_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        preds_event_tuples.append([i,j])

                    if output[i][j] in entity_ids:
                        entity_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        preds_entity_tuples.append([i,j])
                    
                    if output[i][j] in relation_ids:
                        relation_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                    
                    if output[i][j] in role_ids:
                        role_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))

                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
                        if k in event_ids:
                            event_golds.append("{}_{}_{}".format(i, j, k))
                            event_tuples.append([i,j])
                        
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
                            for t in range(i,j+1):
                                if [t,j] in entity_tuples:
                                    entity_2.append([t,j])
                            if len(entity_1)>1 or len(entity_2)>1:
                                print(len(entity_1),len(entity_2))
                                print('关系对应的实体存在起始或终止位置相同的')
                            if len(entity_1)==0 or len(entity_2)==0:
                                # print(len(entity_1),len(entity_2))
                                # print('关系对应的实体不存在')
                                # print(entity_tuples)
                                # print(i,j,labels[k])
                                a=1
                            else:
                                start_1,end_1=entity_1[0]
                                start_2,end_2=entity_2[0]
                                relation_golds_tmp=[]
                                for s_i in range(start_1,end_1+1):
                                    for e_i in range(start_2,end_2+1):
                                        relation_golds_tmp.append("{}_{}_{}".format(s_i, e_i, k))
                                relation_golds.append(relation_golds_tmp)


                        if k in role_ids:
                            role_golds_.append("{}_{}_{}".format(i, j, k))
                            trigger_1=[]
                            entity_2=[]
                            for t in range(i,j+1):
                                if [i,t] in event_tuples:
                                    trigger_1.append([i,t])
                            for m in range(i,j+1):
                                if [m,j] in entity_tuples:
                                    entity_2.append([m,j])
                            if len(trigger_1)==0 and len(entity_2)==0:
                                for t in range(i,j+1):
                                    if [i,t] in entity_tuples:
                                        entity_2.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in event_tuples:
                                        trigger_1.append([m,j])
                            if len(trigger_1)>1 or len(entity_2)>1:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger存在起始或终止位置相同的')
                            if len(trigger_1)==0 or len(entity_2)==0:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger不存在')
                                print(entity_tuples,event_tuples)
                                print(i,j,labels[k])
                            else:
                                start_1,end_1=trigger_1[0]
                                start_2,end_2=entity_2[0]
                                role_golds_tmp=[]
                                for s_i in range(start_1,end_1+1):
                                    for e_i in range(start_2,end_2+1):
                                        role_golds_tmp.append("{}_{}_{}".format(min(s_i,e_i), max(s_i,e_i), k))
                                role_golds.append(role_golds_tmp)

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0 and output[i][j]<label_size:
                    k=output[i][j]  
                    if k in relation_ids:
                        preds_entity_1=[]
                        preds_entity_2=[]
                        # for t in range(i,j):
                        #     if [i,t] in preds_entity_tuples:
                        #         preds_entity_1.append([i,t])
                        # for t in range(j,l):
                        #     if [j,t] in preds_entity_tuples:
                        #         preds_entity_2.append([j,t])
                        # # if len(preds_entity_1)>1 or len(preds_entity_2)>1:
                        # #     print(len(preds_entity_1),len(preds_entity_2))
                        # #     print('关系对应的预测实体存在起始位置相同的')
                        # # if len(preds_entity_1)==0 or len(preds_entity_2)==0:
                        # #     print(len(preds_entity_1),len(preds_entity_2))
                        # #     print('关系对应的预测实体不存在')
                        # # else:
                        # #     start_1,end_1=preds_entity_1[0]
                        # #     start_2,end_2=preds_entity_2[0]
                        # #     two_relation_golds_tmp=[]
                        # #     for s_i in range(start_1,end_1+1):
                        # #         for e_i in range(start_2,end_2+1):
                        # #             two_relation_golds_tmp.append("{}_{}_{}".format(s_i, e_i, k))
                        # #     two_relation_golds.append(two_relation_golds_tmp)
                        # if len(preds_entity_1)>=1 and len(preds_entity_2)>=1:
                        #     two_relation_preds.append("{}_{}_{}".format(i, j, k))
                        
                        for [m,n] in preds_entity_tuples:
                            if m<=i and i<=n:
                                preds_entity_1.append([m,n])
                            if m<=j and j<=n and [m,n] not in preds_entity_1:
                                preds_entity_2.append([m,n])
                        if len(preds_entity_1)>=1 and len(preds_entity_2)>=1:
                            two_relation_preds.append("{}_{}_{}".format(i, j, k))

                    if k in role_ids:
                        preds_trigger_1=[]
                        preds_entity_2=[]
                        # for t in range(i,j):
                        #     if [i,t] in preds_event_tuples:
                        #         preds_trigger_1.append([i,t])
                        #         for t in range(j,l):
                        #             if [j,t] in preds_entity_tuples:
                        #                 preds_entity_2.append([j,t])
                        #     elif [i,t] in preds_entity_tuples:
                        #         preds_entity_2.append([i,t])
                        #         for t in range(j,l):
                        #             if [j,t] in preds_event_tuples:
                        #                 preds_trigger_1.append([j,t])
                        # # if len(preds_trigger_1)>1 or len(preds_entity_2)>1:
                        # #     print(len(preds_trigger_1),len(preds_entity_2))
                        # #     print('事件对应的预测实体或trigger存在起始位置相同的')
                        # # if len(preds_trigger_1)==0 or len(preds_entity_2)==0:
                        # #     print(len(preds_trigger_1),len(preds_entity_2))
                        # #     print('事件对应的预测实体或trigger不存在')
                        # # else:
                        # #     start_1,end_1=preds_trigger_1[0]
                        # #     start_2,end_2=preds_entity_2[0]
                        # #     two_event_tmp=[]
                        # #     for s_i in range(start_1,end_1+1):
                        # #         for e_i in range(start_2,end_2+1):
                        # #             two_event_tmp.append("{}_{}_{}".format(min(s_i,e_i), max(s_i,e_i), k))
                        # #     two_event_golds.append(two_event_tmp)
                        # if len(preds_trigger_1)>=1 and len(preds_entity_2)>=1:
                        #     two_role_preds.append("{}_{}_{}".format(i, j, k))
                        for [m,n] in preds_event_tuples:
                            if m<=i and i<=n:
                                preds_trigger_1.append([m,n])
                        for [m,n] in preds_entity_tuples:
                            if m<=j and j<=n:
                                preds_entity_2.append([m,n])
                        if len(preds_trigger_1)==0 and len(preds_entity_2)==0:
                            for [m,n] in preds_event_tuples:
                                if m<=j and j<=n:
                                    preds_trigger_1.append([m,n])
                            for [m,n] in preds_entity_tuples:
                                if m<=i and i<=n:
                                    preds_entity_2.append([m,n])
                        if len(preds_trigger_1)>=1 and len(preds_entity_2)>=1:
                            two_role_preds.append("{}_{}_{}".format(i, j, k))

        '''对预测总数和正确的预测均去重'''
        # print('start')
        # print(len(entity_golds),entity_golds)
        # print(len(event_golds),event_golds)
        # print(len(relation_golds_),relation_golds_)
        # print(len(role_golds_),role_golds_)
        # print(relation_golds)
        # print(relation_preds)
        # print(role_golds)
        # print(role_preds)
        # print('...........')

        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

        event_preds_count+=len(event_preds)
        event_gold_count+=len(event_golds)
        event_correct+=len(set(event_preds).intersection(set(event_golds)))

        entity_preds_count+=len(entity_preds)
        entity_gold_count+=len(entity_golds)
        entity_correct+=len(set(entity_preds).intersection(set(entity_golds)))

        relation_preds_count+=len(relation_preds)
        relation_gold_count+=len(relation_golds_)
        for relation_golds_single_set in relation_golds:
            if len(set(relation_preds).intersection(set(relation_golds_single_set)))>0:
                relation_correct+=1
                relation_preds_count=relation_preds_count-len(set(relation_preds).intersection(set(relation_golds_single_set)))+1

        role_preds_count+=len(role_preds)
        role_gold_count+=len(role_golds_)
        for role_golds_single_set in role_golds:
            if len(set(role_preds).intersection(set(role_golds_single_set)))>0:
                role_correct+=1
                role_preds_count=role_preds_count-len(set(role_preds).intersection(set(role_golds_single_set)))+1

        two_relation_preds_count+=len(two_relation_preds)
        two_relation_gold_count+=len(relation_golds_)
        for two_relation_golds_single_set in relation_golds:
            if len(set(two_relation_preds).intersection(set(two_relation_golds_single_set)))>0:
                two_relation_correct+=1
                two_relation_preds_count=two_relation_preds_count-len(set(two_relation_preds).intersection(set(two_relation_golds_single_set)))+1

        two_role_preds_count+=len(two_role_preds)
        two_role_gold_count+=len(role_golds_)
        for two_role_golds_single_set in role_golds:
            if len(set(two_role_preds).intersection(set(two_role_golds_single_set)))>0:
                two_role_correct+=1
                two_role_preds_count=two_role_preds_count-len(set(two_role_preds).intersection(set(two_role_golds_single_set)))+1

    return correct, pred_count, gold_count,event_correct,event_preds_count,event_gold_count, \
            entity_correct,entity_preds_count,entity_gold_count,relation_correct,relation_preds_count,relation_gold_count, \
            role_correct,role_preds_count,role_gold_count, two_relation_correct,two_relation_preds_count,two_relation_gold_count, \
            two_role_correct,two_role_preds_count,two_role_gold_count

def eval_v6(args, outputs, partial_masks, labels, gather_masks, valid_pattern):
    '''
    头-尾,解析tree
    '''
    label_size=len(labels)
    argu_ids=[labels.index(item) for item in valid_pattern['role']]
    entity_ids=[labels.index(item) for item in valid_pattern['ner']]
    relation_ids=[labels.index(item) for item in valid_pattern['relation']]
    trigger_ids=[labels.index(item) for item in valid_pattern['event']]

    correct, pred_count, gold_count = 0, 0, 0
    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0

    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()

    label_num_dict = dict()
    label_num_dict['entity'] = dict()
    label_num_dict['relation'] = dict()
    label_num_dict['role'] = dict()
    label_num_dict['trigger'] = dict()
    for key in valid_pattern['ner']:
        label_num_dict['entity'][key] = dict()
        label_num_dict['entity'][key]['correct'] = 0
        label_num_dict['entity'][key]['pred'] = 0
        label_num_dict['entity'][key]['gold'] = 0
    for key in valid_pattern['role']:
        label_num_dict['role'][key] = dict()
        label_num_dict['role'][key]['correct'] = 0
        label_num_dict['role'][key]['pred'] = 0
        label_num_dict['role'][key]['gold'] = 0
    for key in valid_pattern['relation']:
        label_num_dict['relation'][key] = dict()
        label_num_dict['relation'][key]['correct'] = 0
        label_num_dict['relation'][key]['pred'] = 0
        label_num_dict['relation'][key]['gold'] = 0
    for key in valid_pattern['event']:
        label_num_dict['trigger'][key] = dict()
        label_num_dict['trigger'][key]['correct'] = 0
        label_num_dict['trigger'][key]['pred'] = 0
        label_num_dict['trigger'][key]['gold'] = 0
    # import numpy as np  
    # np.set_printoptions(threshold=np.inf)
    # print(partial_masks)
    # stop

    for output, partial_mask, l in zip(outputs, partial_masks, gather_masks):
        # torch.set_printoptions(profile="full")
        # import numpy as np
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        golds = list()
        preds = list()
        entity_golds=list()
        entity_preds=list()
        relation_golds=list()
        relation_preds=list()
        argu_golds=list()
        argu_preds=list()
        trigger_golds=list()
        trigger_preds=list()

        pred_argu_tuples = []
        pred_argu_str = []
        gold_argu_tuples = []
        gold_argu_str = []
        pred_entity_tuples = []
        gold_entity_tuples = []
        pred_trigger_tuples = []
        pred_trigger_str = []
        gold_trigger_tuples = []
        gold_trigger_str = []
        gold_relation_tuples = []
        pred_relation_tuples = []

        gold_label_dict = dict()
        pred_label_dict = dict()
        gold_label_dict['entity'] = dict()
        gold_label_dict['relation'] = dict()
        gold_label_dict['role'] = dict()
        gold_label_dict['trigger'] = dict()
        pred_label_dict['entity']  = dict()
        pred_label_dict['relation'] = dict()
        pred_label_dict['role'] = dict()
        pred_label_dict['trigger'] = dict()

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0:
                    if output[i][j] < label_size:
                        preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                    if output[i][j] in trigger_ids:
                        trigger_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        pred_trigger_tuples.append([i,j])
                        pred_trigger_str.append("{}_{}".format(i, j))
                        if labels[int(output[i][j])] not in pred_label_dict['trigger'].keys():
                            pred_label_dict['trigger'][labels[int(output[i][j])]] = []
                        pred_label_dict['trigger'][labels[int(output[i][j])]].append("{}_{}_{}".format(i, j, int(output[i][j])))

                    if output[i][j] in entity_ids:
                        entity_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        pred_entity_tuples.append([i,j])
                        if labels[int(output[i][j])] not in pred_label_dict['entity'].keys():
                            pred_label_dict['entity'][labels[int(output[i][j])]] = []
                        pred_label_dict['entity'][labels[int(output[i][j])]].append("{}_{}_{}".format(i, j, int(output[i][j])))
                    
                    if output[i][j] in relation_ids:
                        relation_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                    
                    if output[i][j] in argu_ids:
                        argu_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))

                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
                        if k in trigger_ids:
                            trigger_golds.append("{}_{}_{}".format(i, j, k))
                            gold_trigger_tuples.append([i,j])
                            gold_trigger_str.append("{}_{}".format(i, j))
                            if labels[k] not in gold_label_dict['trigger'].keys():
                                gold_label_dict['trigger'][labels[k]] = []
                            gold_label_dict['trigger'][labels[k]].append("{}_{}_{}".format(i, j, k))
                        
                        if k in entity_ids:
                            entity_golds.append("{}_{}_{}".format(i, j, k))
                            gold_entity_tuples.append([i,j])
                            if labels[k] not in gold_label_dict['entity'].keys():
                                gold_label_dict['entity'][labels[k]] = []
                            gold_label_dict['entity'][labels[k]].append("{}_{}_{}".format(i, j, k))

        for i in range(l):
            for j in range(l):
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        if k in relation_ids:
                            relation_golds.append("{}_{}_{}".format(i, j, k))
                            entity_1=[]
                            entity_2=[]
                            for t in range(i,j):
                                if [i,t] in gold_entity_tuples:
                                    entity_1.append([i,t])
                            for t in range(i,j+1):
                                if [t,j] in gold_entity_tuples:
                                    entity_2.append([t,j])
                            if len(entity_1)>1 or len(entity_2)>1:
                                print(len(entity_1),len(entity_2),i,j,k,labels[k])
                                print('关系对应的实体存在起始或终止位置相同的')
                                print(entity_1,entity_2)
                            if len(entity_1)==0 or len(entity_2)==0:
                                print('关系错误，存在非实体')
                                print(len(entity_1),len(entity_2),i,j,k,labels[k])
                            else:
                                gold_relation_tuples.append("{}_{}_{}_{}_{}".format(entity_1[0][0], entity_1[0][1], entity_2[0][0], entity_2[0][1], k))
                                if labels[k] not in gold_label_dict['relation'].keys():
                                    gold_label_dict['relation'][labels[k]] = []
                                gold_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity_1[0][0], entity_1[0][1], entity_2[0][0], entity_2[0][1], k))

                        if k in argu_ids:
                            argu_golds.append("{}_{}_{}".format(i, j, k))
                            trigger_1=[]
                            entity_2=[]

                            for t in range(i,j):
                                if [i,t] in gold_trigger_tuples:
                                    trigger_1.append([i,t])
                            for m in range(i,j+1):
                                if [m,j] in gold_entity_tuples:
                                    entity_2.append([m,j])
                            if len(trigger_1)==0 and len(entity_2)==0:
                                for t in range(i,j):
                                    if [i,t] in gold_entity_tuples:
                                        entity_2.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_trigger_tuples:
                                        trigger_1.append([m,j])
                            if len(trigger_1)==0 or len(entity_2)==0:
                                for t in range(i,j+1):
                                    if [i,t] in gold_entity_tuples:
                                        entity_2.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_trigger_tuples:
                                        trigger_1.append([m,j])
                            elif len(trigger_1)==0 or len(entity_2)==0:
                                for t in range(i,j+1):
                                    if [i,t] in gold_trigger_tuples:
                                        trigger_1.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_entity_tuples:
                                        entity_2.append([m,j])
                            if len(trigger_1)>1 or len(entity_2)>1:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger存在起始或终止位置相同的')
                            if len(trigger_1)==0 or len(entity_2)==0:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger不存在')
                                print(gold_entity_tuples,gold_trigger_tuples)
                                print(i,j,labels[k])
                            else:
                            # gold_argu_tuples.append([trigger_1[0],entity_2[0],k])
                                gold_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1],k))
                                gold_argu_str.append("{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1]))
                                if labels[k] not in gold_label_dict['role'].keys():
                                    gold_label_dict['role'][labels[k]] = []
                                gold_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1],k))

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0 and output[i][j]<label_size:
                    k = int(output[i][j])
                    if k in relation_ids:
                        preds_entity_1 = []
                        preds_entity_2 = []
                        for [m,n] in pred_entity_tuples:
                            if i >= m and i <= n and (j < m or j > n):
                                preds_entity_1.append([m,n])
                            if j >= m and j <= n and (i < m or i > n):
                                preds_entity_2.append([m,n])
                        if len(preds_entity_1) > 1 or len(preds_entity_2) > 1:
                            print('预测的关系对应多个实体')
                        if len(preds_entity_1) >= 1 and len(preds_entity_2) >= 1:
                            for entity1_start, entity1_end in preds_entity_1:
                                for entity2_start, entity2_end in preds_entity_2:
                                    if entity1_start < entity2_start:
                                        pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                                        if labels[k] not in pred_label_dict['relation'].keys():
                                            pred_label_dict['relation'][labels[k]] = []
                                        pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                                    else:
                                        pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                                        if labels[k] not in pred_label_dict['relation'].keys():
                                            pred_label_dict['relation'][labels[k]] = []
                                        pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                            # if preds_entity_1[0][0] < preds_entity_2[0][0]:
                            #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_1[0][0], preds_entity_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                            # else:
                            #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_2[0][0], preds_entity_2[0][1], preds_entity_1[0][0], preds_entity_1[0][1], k))

                    if k in argu_ids:
                        preds_trigger_1=[]
                        preds_entity_2=[]
                        if 'r_' in labels[k]:
                            for [m,n] in pred_trigger_tuples:
                                if m<=j and j<=n:
                                    preds_trigger_1.append([m,n])
                            for [m,n] in pred_entity_tuples:
                                if m<=i and i<=n:
                                    preds_entity_2.append([m,n])
                        else:
                            for [m,n] in pred_trigger_tuples:
                                if m<=i and i<=n:
                                    preds_trigger_1.append([m,n])
                            for [m,n] in pred_entity_tuples:
                                if m<=j and j<=n:
                                    preds_entity_2.append([m,n])
   
                        if len(preds_trigger_1) > 1 or len(preds_entity_2) > 1:
                            print('预测的事件对应多个触发器或实体')
                            print(i,j,preds_trigger_1,preds_entity_2)
                        if len(preds_trigger_1) >= 1 and len(preds_entity_2) >= 1:
                            for trigger_start, trigger_end in preds_trigger_1:
                                for entity_start, entity_end in preds_entity_2:
                                    pred_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                                    pred_argu_str.append("{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end))
                                    if labels[k] not in pred_label_dict['role'].keys():
                                        pred_label_dict['role'][labels[k]] = []
                                    pred_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                            # pred_argu_tuples.append("{}_{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                            # pred_argu_str.append("{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1]))

        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

        pred_trigger_num += len(trigger_preds)
        gold_trigger_num += len(trigger_golds)
        trigger_class_num += len(set(trigger_preds).intersection(set(trigger_golds)))
        trigger_idn_num += len(set(pred_trigger_str).intersection(set(gold_trigger_str)))

        pred_ent_num += len(entity_preds)
        gold_ent_num += len(entity_golds)
        ent_match_num += len(set(entity_preds).intersection(set(entity_golds)))

        pred_rel_num += len(relation_preds)
        gold_rel_num += len(relation_golds)
        rel_match_num += len(set(gold_relation_tuples).intersection(set(pred_relation_tuples)))

        pred_arg_num += len(argu_preds)
        gold_arg_num += len(argu_golds)
        arg_class_num += len(set(gold_argu_tuples).intersection(set(pred_argu_tuples)))
        arg_idn_num += len(set(pred_argu_str).intersection(set(gold_argu_str)))

        for key in pred_label_dict['entity'].keys():
            label_num_dict['entity'][key]['pred'] += len(pred_label_dict['entity'][key])
        for key in gold_label_dict['entity'].keys():
            label_num_dict['entity'][key]['gold'] += len(gold_label_dict['entity'][key])
            if key in list(pred_label_dict['entity'].keys()):
                label_num_dict['entity'][key]['correct'] += len(set(pred_label_dict['entity'][key]).intersection(set(gold_label_dict['entity'][key])))


        for key in pred_label_dict['relation'].keys():
            label_num_dict['relation'][key]['pred'] += len(pred_label_dict['relation'][key])
        for key in gold_label_dict['relation'].keys():
            label_num_dict['relation'][key]['gold'] += len(gold_label_dict['relation'][key])
            if key in list(pred_label_dict['relation'].keys()):
                label_num_dict['relation'][key]['correct'] += len(set(pred_label_dict['relation'][key]).intersection(set(gold_label_dict['relation'][key])))


        for key in pred_label_dict['role'].keys():
            label_num_dict['role'][key]['pred'] += len(pred_label_dict['role'][key])
        for key in gold_label_dict['role'].keys():
            label_num_dict['role'][key]['gold'] += len(gold_label_dict['role'][key])
            if key in list(pred_label_dict['role'].keys()):
                label_num_dict['role'][key]['correct'] += len(set(pred_label_dict['role'][key]).intersection(set(gold_label_dict['role'][key])))

        for key in pred_label_dict['trigger'].keys():
            label_num_dict['trigger'][key]['pred'] += len(pred_label_dict['trigger'][key])
        for key in gold_label_dict['trigger'].keys():
            label_num_dict['trigger'][key]['gold'] += len(gold_label_dict['trigger'][key])
            if key in list(pred_label_dict['trigger'].keys()):
                label_num_dict['trigger'][key]['correct'] += len(set(pred_label_dict['trigger'][key]).intersection(set(gold_label_dict['trigger'][key])))
        
        # print(trigger_preds)
        # print(trigger_golds)
        # print(pred_trigger_str)
        # print(gold_trigger_str)
        # print(entity_preds)
        # print(entity_golds)
        # print(gold_relation_tuples)
        # print(pred_relation_tuples)
        # print(relation_preds)
        # print(argu_golds)
        # print(argu_preds)
        # print(pred_argu_str)
        # print(gold_argu_str)
        # print(pred_argu_tuples)
        # print(gold_argu_tuples)
        # print('trigger_pred',trigger_preds)
        # print('trigger_gold',trigger_golds)
        # print('entity_pred',entity_preds)
        # print('entity_gold',entity_golds)
        # print('relation_preds',relation_preds)
        # print('relation_golds',relation_golds)
        # print('pred_relation_tuples',pred_relation_tuples)
        # print('gold_relation_tuples',gold_relation_tuples)
        # print('argu_preds',argu_preds)
        # print('argu_golds',argu_golds)
        # print('pred_argu_tuples',pred_argu_tuples)
        # print('gold_argu_tuples',gold_argu_tuples)
    
    for key in ['entity', 'trigger', 'relation', 'role']:
        for lab in label_num_dict[key].keys():
            if label_num_dict[key][lab]['pred'] == 0 :
                label_num_dict[key][lab]['precision'] = 0
            else:
                label_num_dict[key][lab]['precision'] = label_num_dict[key][lab]['correct'] / label_num_dict[key][lab]['pred']
            if label_num_dict[key][lab]['gold'] == 0:
                label_num_dict[key][lab]['recall'] = 0
            else:
                label_num_dict[key][lab]['recall'] = label_num_dict[key][lab]['correct'] / label_num_dict[key][lab]['gold']
            if label_num_dict[key][lab]['precision'] + label_num_dict[key][lab]['recall'] == 0:
                label_num_dict[key][lab]['f1'] = 0
            else:
                label_num_dict[key][lab]['f1'] = 2 * label_num_dict[key][lab]['precision'] * label_num_dict[key][lab]['recall'] / \
                                        (label_num_dict[key][lab]['precision'] + label_num_dict[key][lab]['recall'])
            print(key, ' ', lab,' ', 'precision', label_num_dict[key][lab]['precision'])
            print(key, ' ', lab, ' ', 'recall', label_num_dict[key][lab]['recall'])
            print(key, ' ', lab, ' ', 'f1', label_num_dict[key][lab]['f1'])

    return correct, pred_count, gold_count, pred_ent_num, gold_ent_num, ent_match_num, pred_trigger_num, gold_trigger_num, trigger_idn_num, \
    pred_trigger_num, gold_trigger_num, trigger_class_num, \
    pred_rel_num, gold_rel_num, rel_match_num, pred_arg_num, gold_arg_num, arg_idn_num, \
    arg_class_num
        # golds = list()所有的实体
        # preds = list()预测的所有实体

        # entity_golds=list()实体gold三元组
        # entity_preds=list()预测的实体三元组
        # pred_entity_tuples = []预测的实体二元组列表
        # gold_entity_tuples = []实体gold二元组列表

        # relation_golds=list()关系gold三元组
        # relation_preds=list()预测的关系三元组
        # gold_relation_tuples = []关系gold五元组
        # pred_relation_tuples = []预测关系五元组

        # pred_argu_tuples = []预测论元五元组
        # pred_argu_str = []预测论元四元组
        # gold_argu_tuples = []论元gold五元组
        # gold_argu_str = []论元gold四元组
        # argu_golds=list()论元gold三元组
        # argu_preds=list()预测的论元三元组

        # pred_trigger_tuples = []预测的触发器二元组列表
        # pred_trigger_str = []预测的触发器二元组字符串
        # trigger_preds=list()预测的触发器三元组
        # gold_trigger_tuples = []触发器gold二元组列表
        # gold_trigger_str = []触发器gold二元组字符串
        # trigger_golds=list()触发器gold三元组

def eval(args, outputs, partial_masks, labels, gather_masks, valid_pattern):
    '''
    头-尾，解析双仿射+掩码
    '''
    label_size=len(labels)
    argu_ids=[labels.index(item) for item in valid_pattern['role']]
    entity_ids=[labels.index(item) for item in valid_pattern['ner']]
    relation_ids=[labels.index(item) for item in valid_pattern['relation']]
    trigger_ids=[labels.index(item) for item in valid_pattern['event']]

    correct, pred_count, gold_count = 0, 0, 0
    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0

    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()

    label_num_dict = dict()
    label_num_dict['entity'] = dict()
    label_num_dict['relation'] = dict()
    label_num_dict['role'] = dict()
    label_num_dict['trigger'] = dict()
    for key in valid_pattern['ner']:
        label_num_dict['entity'][key] = dict()
        label_num_dict['entity'][key]['correct'] = 0
        label_num_dict['entity'][key]['pred'] = 0
        label_num_dict['entity'][key]['gold'] = 0
    for key in valid_pattern['role']:
        label_num_dict['role'][key] = dict()
        label_num_dict['role'][key]['correct'] = 0
        label_num_dict['role'][key]['pred'] = 0
        label_num_dict['role'][key]['gold'] = 0
    for key in valid_pattern['relation']:
        label_num_dict['relation'][key] = dict()
        label_num_dict['relation'][key]['correct'] = 0
        label_num_dict['relation'][key]['pred'] = 0
        label_num_dict['relation'][key]['gold'] = 0
    for key in valid_pattern['event']:
        label_num_dict['trigger'][key] = dict()
        label_num_dict['trigger'][key]['correct'] = 0
        label_num_dict['trigger'][key]['pred'] = 0
        label_num_dict['trigger'][key]['gold'] = 0
    # import numpy as np  
    # np.set_printoptions(threshold=np.inf)
    # print(partial_masks)
    # stop

    for output, partial_mask, l in zip(outputs, partial_masks, gather_masks):
        # torch.set_printoptions(profile="full")
        # import numpy as np
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        golds = list()
        preds = list()
        entity_golds=list()
        entity_preds=list()
        relation_golds=list()
        relation_preds=list()
        argu_golds=list()
        argu_preds=list()
        trigger_golds=list()
        trigger_preds=list()

        pred_argu_tuples = []
        pred_argu_str = []
        gold_argu_tuples = []
        gold_argu_str = []
        pred_entity_tuples = []
        gold_entity_tuples = []
        pred_trigger_tuples = []
        pred_trigger_str = []
        gold_trigger_tuples = []
        gold_trigger_str = []
        gold_relation_tuples = []
        pred_relation_tuples = []

        gold_label_dict = dict()
        pred_label_dict = dict()
        gold_label_dict['entity'] = dict()
        gold_label_dict['relation'] = dict()
        gold_label_dict['role'] = dict()
        gold_label_dict['trigger'] = dict()
        pred_label_dict['entity']  = dict()
        pred_label_dict['relation'] = dict()
        pred_label_dict['role'] = dict()
        pred_label_dict['trigger'] = dict()

        for i in range(l):
            for j in range(l):
                # pred_labels = output[i][j]
                # for k, _ in enumerate(pred_labels):
                #     if _ <= 0:
                #         continue
                #     if k < label_size:
                #         preds.append("{}_{}_{}".format(i, j, int(k)))
                #     if k in trigger_ids:
                #         trigger_preds.append("{}_{}_{}".format(i, j, int(k)))
                #         pred_trigger_tuples.append([i,j])
                #         pred_trigger_str.append("{}_{}".format(i, j))
                #         if labels[int(k)] not in pred_label_dict['trigger'].keys():
                #             pred_label_dict['trigger'][labels[int(k)]] = []
                #         pred_label_dict['trigger'][labels[int(k)]].append("{}_{}_{}".format(i, j, int(k)))

                #     if k in entity_ids:
                #         entity_preds.append("{}_{}_{}".format(i, j, int(k)))
                #         pred_entity_tuples.append([i,j])
                #         if labels[int(k)] not in pred_label_dict['entity'].keys():
                #             pred_label_dict['entity'][labels[int(k)]] = []
                #         pred_label_dict['entity'][labels[int(k)]].append("{}_{}_{}".format(i, j, int(k)))
                    
                #     if k in relation_ids:
                #         relation_preds.append("{}_{}_{}".format(i, j, int(k)))
                    
                #     if k in argu_ids:
                #         argu_preds.append("{}_{}_{}".format(i, j, int(k)))

                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
                        if k in trigger_ids:
                            trigger_golds.append("{}_{}_{}".format(i, j, k))
                            gold_trigger_tuples.append([i,j])
                            gold_trigger_str.append("{}_{}".format(i, j))
                            if labels[k] not in gold_label_dict['trigger'].keys():
                                gold_label_dict['trigger'][labels[k]] = []
                            gold_label_dict['trigger'][labels[k]].append("{}_{}_{}".format(i, j, k))
                        
                        if k in entity_ids:
                            entity_golds.append("{}_{}_{}".format(i, j, k))
                            gold_entity_tuples.append([i,j])
                            if labels[k] not in gold_label_dict['entity'].keys():
                                gold_label_dict['entity'][labels[k]] = []
                            gold_label_dict['entity'][labels[k]].append("{}_{}_{}".format(i, j, k))

        for k, i, j in zip(*np.where(output > 0)):
            if k < label_size:
                preds.append("{}_{}_{}".format(i, j, int(k)))
            if k in trigger_ids:
                trigger_preds.append("{}_{}_{}".format(i, j, int(k)))
                pred_trigger_tuples.append([i,j])
                pred_trigger_str.append("{}_{}".format(i, j))
                if labels[int(k)] not in pred_label_dict['trigger'].keys():
                    pred_label_dict['trigger'][labels[int(k)]] = []
                pred_label_dict['trigger'][labels[int(k)]].append("{}_{}_{}".format(i, j, int(k)))

            if k in entity_ids:
                entity_preds.append("{}_{}_{}".format(i, j, int(k)))
                pred_entity_tuples.append([i,j])
                if labels[int(k)] not in pred_label_dict['entity'].keys():
                    pred_label_dict['entity'][labels[int(k)]] = []
                pred_label_dict['entity'][labels[int(k)]].append("{}_{}_{}".format(i, j, int(k)))
            
            if k in relation_ids:
                relation_preds.append("{}_{}_{}".format(i, j, int(k)))
            
            if k in argu_ids:
                argu_preds.append("{}_{}_{}".format(i, j, int(k)))

        for i in range(l):
            for j in range(l):
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        if k in relation_ids:
                            relation_golds.append("{}_{}_{}".format(i, j, k))
                            entity_1=[]
                            entity_2=[]
                            for t in range(i,j):
                                if [i,t] in gold_entity_tuples:
                                    entity_1.append([i,t])
                            for t in range(i,j+1):
                                if [t,j] in gold_entity_tuples:
                                    entity_2.append([t,j])
                            if len(entity_1)>1 or len(entity_2)>1:
                                print(len(entity_1),len(entity_2),i,j,k,labels[k])
                                print('关系对应的实体存在起始或终止位置相同的')
                                print(entity_1,entity_2)
                            if len(entity_1)==0 or len(entity_2)==0:
                                print('关系错误，存在非实体')
                                print(len(entity_1),len(entity_2),i,j,k,labels[k])
                            else:
                                gold_relation_tuples.append("{}_{}_{}_{}_{}".format(entity_1[0][0], entity_1[0][1], entity_2[0][0], entity_2[0][1], k))
                                if labels[k] not in gold_label_dict['relation'].keys():
                                    gold_label_dict['relation'][labels[k]] = []
                                gold_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity_1[0][0], entity_1[0][1], entity_2[0][0], entity_2[0][1], k))

                        if k in argu_ids:
                            argu_golds.append("{}_{}_{}".format(i, j, k))
                            trigger_1=[]
                            entity_2=[]

                            for t in range(i,j):
                                if [i,t] in gold_trigger_tuples:
                                    trigger_1.append([i,t])
                            for m in range(i,j+1):
                                if [m,j] in gold_entity_tuples:
                                    entity_2.append([m,j])
                            if len(trigger_1)==0 and len(entity_2)==0:
                                for t in range(i,j):
                                    if [i,t] in gold_entity_tuples:
                                        entity_2.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_trigger_tuples:
                                        trigger_1.append([m,j])
                            if len(trigger_1)==0 or len(entity_2)==0:
                                for t in range(i,j+1):
                                    if [i,t] in gold_entity_tuples:
                                        entity_2.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_trigger_tuples:
                                        trigger_1.append([m,j])
                            elif len(trigger_1)==0 or len(entity_2)==0:
                                for t in range(i,j+1):
                                    if [i,t] in gold_trigger_tuples:
                                        trigger_1.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_entity_tuples:
                                        entity_2.append([m,j])
                            if len(trigger_1)>1 or len(entity_2)>1:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger存在起始或终止位置相同的')
                            if len(trigger_1)==0 or len(entity_2)==0:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger不存在')
                                print(gold_entity_tuples,gold_trigger_tuples)
                                print(i,j,labels[k])
                            else:
                            # gold_argu_tuples.append([trigger_1[0],entity_2[0],k])
                                gold_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1],k))
                                gold_argu_str.append("{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1]))
                                if labels[k] not in gold_label_dict['role'].keys():
                                    gold_label_dict['role'][labels[k]] = []
                                gold_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1],k))

        for k, i, j in zip(*np.where(output > 0)):
            # print(i,j,k, output[k][i][j],labels[40])
            if k in relation_ids:
                preds_entity_1 = []
                preds_entity_2 = []
                for [m,n] in pred_entity_tuples:
                    if i >= m and i <= n and (j < m or j > n):
                        preds_entity_1.append([m,n])
                    if j >= m and j <= n and (i < m or i > n):
                        preds_entity_2.append([m,n])
                if len(preds_entity_1) > 1 or len(preds_entity_2) > 1:
                    # print('预测的关系对应多个实体')
                    pass
                if len(preds_entity_1) >= 1 and len(preds_entity_2) >= 1:
                    for entity1_start, entity1_end in preds_entity_1:
                        for entity2_start, entity2_end in preds_entity_2:
                            if entity1_start < entity2_start:
                                pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                                if labels[k] not in pred_label_dict['relation'].keys():
                                    pred_label_dict['relation'][labels[k]] = []
                                pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                            else:
                                pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                                if labels[k] not in pred_label_dict['relation'].keys():
                                    pred_label_dict['relation'][labels[k]] = []
                                pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                    # if preds_entity_1[0][0] < preds_entity_2[0][0]:
                    #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_1[0][0], preds_entity_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                    # else:
                    #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_2[0][0], preds_entity_2[0][1], preds_entity_1[0][0], preds_entity_1[0][1], k))

            if k in argu_ids:
                preds_trigger_1=[]
                preds_entity_2=[]
                if 'r_' in labels[k]:
                    for [m,n] in pred_trigger_tuples:
                        if m<=j and j<=n:
                            preds_trigger_1.append([m,n])
                    for [m,n] in pred_entity_tuples:
                        if m<=i and i<=n:
                            preds_entity_2.append([m,n])
                else:
                    for [m,n] in pred_trigger_tuples:
                        if m<=i and i<=n:
                            preds_trigger_1.append([m,n])
                    for [m,n] in pred_entity_tuples:
                        if m<=j and j<=n:
                            preds_entity_2.append([m,n])
                    if len(preds_trigger_1) == 0 and len(preds_entity_2) == 0:
                        for [m,n] in pred_trigger_tuples:
                            if m<=j and j<=n:
                                preds_trigger_1.append([m,n])
                        for [m,n] in pred_entity_tuples:
                            if m<=i and i<=n:
                                preds_entity_2.append([m,n])

                if len(preds_trigger_1) > 1 or len(preds_entity_2) > 1:
                    pass
                    # print('预测的事件对应多个触发器或实体')
                    # print(i,j,preds_trigger_1,preds_entity_2)
                if len(preds_trigger_1) >= 1 and len(preds_entity_2) >= 1:
                    for trigger_start, trigger_end in preds_trigger_1:
                        for entity_start, entity_end in preds_entity_2:
                            pred_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                            pred_argu_str.append("{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end))
                            if labels[k] not in pred_label_dict['role'].keys():
                                pred_label_dict['role'][labels[k]] = []
                            pred_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                    # pred_argu_tuples.append("{}_{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                    # pred_argu_str.append("{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1]))

        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

        pred_trigger_num += len(trigger_preds)
        gold_trigger_num += len(trigger_golds)
        trigger_class_num += len(set(trigger_preds).intersection(set(trigger_golds)))
        trigger_idn_num += len(set(pred_trigger_str).intersection(set(gold_trigger_str)))

        pred_ent_num += len(entity_preds)
        gold_ent_num += len(entity_golds)
        ent_match_num += len(set(entity_preds).intersection(set(entity_golds)))

        pred_rel_num += len(relation_preds)
        gold_rel_num += len(relation_golds)
        rel_match_num += len(set(gold_relation_tuples).intersection(set(pred_relation_tuples)))

        pred_arg_num += len(argu_preds)
        gold_arg_num += len(argu_golds)
        arg_class_num += len(set(gold_argu_tuples).intersection(set(pred_argu_tuples)))
        arg_idn_num += len(set(pred_argu_str).intersection(set(gold_argu_str)))

        for key in pred_label_dict['entity'].keys():
            label_num_dict['entity'][key]['pred'] += len(pred_label_dict['entity'][key])
        for key in gold_label_dict['entity'].keys():
            label_num_dict['entity'][key]['gold'] += len(gold_label_dict['entity'][key])
            if key in list(pred_label_dict['entity'].keys()):
                label_num_dict['entity'][key]['correct'] += len(set(pred_label_dict['entity'][key]).intersection(set(gold_label_dict['entity'][key])))


        for key in pred_label_dict['relation'].keys():
            label_num_dict['relation'][key]['pred'] += len(pred_label_dict['relation'][key])
        for key in gold_label_dict['relation'].keys():
            label_num_dict['relation'][key]['gold'] += len(gold_label_dict['relation'][key])
            if key in list(pred_label_dict['relation'].keys()):
                label_num_dict['relation'][key]['correct'] += len(set(pred_label_dict['relation'][key]).intersection(set(gold_label_dict['relation'][key])))


        for key in pred_label_dict['role'].keys():
            label_num_dict['role'][key]['pred'] += len(pred_label_dict['role'][key])
        for key in gold_label_dict['role'].keys():
            label_num_dict['role'][key]['gold'] += len(gold_label_dict['role'][key])
            if key in list(pred_label_dict['role'].keys()):
                label_num_dict['role'][key]['correct'] += len(set(pred_label_dict['role'][key]).intersection(set(gold_label_dict['role'][key])))

        for key in pred_label_dict['trigger'].keys():
            label_num_dict['trigger'][key]['pred'] += len(pred_label_dict['trigger'][key])
        for key in gold_label_dict['trigger'].keys():
            label_num_dict['trigger'][key]['gold'] += len(gold_label_dict['trigger'][key])
            if key in list(pred_label_dict['trigger'].keys()):
                label_num_dict['trigger'][key]['correct'] += len(set(pred_label_dict['trigger'][key]).intersection(set(gold_label_dict['trigger'][key])))
        
        # print(trigger_preds)
        # print(trigger_golds)
        # print(pred_trigger_str)
        # print(gold_trigger_str)
        # print(entity_preds)
        # print(entity_golds)
        # print(gold_relation_tuples)
        # print(pred_relation_tuples)
        # print(relation_preds)
        # print(argu_golds)
        # print(argu_preds)
        # print(pred_argu_str)
        # print(gold_argu_str)
        # print(pred_argu_tuples)
        # print(gold_argu_tuples)
        # print('trigger_pred',trigger_preds)
        # print('trigger_gold',trigger_golds)
        # print('entity_pred',entity_preds)
        # print('entity_gold',entity_golds)
        # print('relation_preds',relation_preds)
        # print('relation_golds',relation_golds)
        # print('pred_relation_tuples',pred_relation_tuples)
        # print('gold_relation_tuples',gold_relation_tuples)
        # print('argu_preds',argu_preds)
        # print('argu_golds',argu_golds)
        # print('pred_argu_tuples',pred_argu_tuples)
        # print('gold_argu_tuples',gold_argu_tuples)
    
    for key in ['entity', 'trigger', 'relation', 'role']:
        for lab in label_num_dict[key].keys():
            if label_num_dict[key][lab]['pred'] == 0 :
                label_num_dict[key][lab]['precision'] = 0
            else:
                label_num_dict[key][lab]['precision'] = label_num_dict[key][lab]['correct'] / label_num_dict[key][lab]['pred']
            if label_num_dict[key][lab]['gold'] == 0:
                label_num_dict[key][lab]['recall'] = 0
            else:
                label_num_dict[key][lab]['recall'] = label_num_dict[key][lab]['correct'] / label_num_dict[key][lab]['gold']
            if label_num_dict[key][lab]['precision'] + label_num_dict[key][lab]['recall'] == 0:
                label_num_dict[key][lab]['f1'] = 0
            else:
                label_num_dict[key][lab]['f1'] = 2 * label_num_dict[key][lab]['precision'] * label_num_dict[key][lab]['recall'] / \
                                        (label_num_dict[key][lab]['precision'] + label_num_dict[key][lab]['recall'])
            print(key, ' ', lab,' ', 'precision', label_num_dict[key][lab]['precision'])
            print(key, ' ', lab, ' ', 'recall', label_num_dict[key][lab]['recall'])
            print(key, ' ', lab, ' ', 'f1', label_num_dict[key][lab]['f1'])

    return correct, pred_count, gold_count, pred_ent_num, gold_ent_num, ent_match_num, pred_trigger_num, gold_trigger_num, trigger_idn_num, \
    pred_trigger_num, gold_trigger_num, trigger_class_num, \
    pred_rel_num, gold_rel_num, rel_match_num, pred_arg_num, gold_arg_num, arg_idn_num, \
    arg_class_num
        # golds = list()所有的实体
        # preds = list()预测的所有实体

        # entity_golds=list()实体gold三元组
        # entity_preds=list()预测的实体三元组
        # pred_entity_tuples = []预测的实体二元组列表
        # gold_entity_tuples = []实体gold二元组列表

        # relation_golds=list()关系gold三元组
        # relation_preds=list()预测的关系三元组
        # gold_relation_tuples = []关系gold五元组
        # pred_relation_tuples = []预测关系五元组

        # pred_argu_tuples = []预测论元五元组
        # pred_argu_str = []预测论元四元组
        # gold_argu_tuples = []论元gold五元组
        # gold_argu_str = []论元gold四元组
        # argu_golds=list()论元gold三元组
        # argu_preds=list()预测的论元三元组

        # pred_trigger_tuples = []预测的触发器二元组列表
        # pred_trigger_str = []预测的触发器二元组字符串
        # trigger_preds=list()预测的触发器三元组
        # gold_trigger_tuples = []触发器gold二元组列表
        # gold_trigger_str = []触发器gold二元组字符串
        # trigger_golds=list()触发器gold三元组

def eval_5(args, outputs, partial_masks, labels, gather_masks, valid_pattern):
    '''
    尾-头
    '''
    label_size=len(labels)
    argu_ids=[labels.index(item) for item in valid_pattern['role']]
    entity_ids=[labels.index(item) for item in valid_pattern['ner']]
    relation_ids=[labels.index(item) for item in valid_pattern['relation']]
    trigger_ids=[labels.index(item) for item in valid_pattern['event']]

    correct, pred_count, gold_count = 0, 0, 0
    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0

    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()

    label_num_dict = dict()
    label_num_dict['entity'] = dict()
    label_num_dict['relation'] = dict()
    label_num_dict['role'] = dict()
    label_num_dict['trigger'] = dict()
    for key in valid_pattern['ner']:
        label_num_dict['entity'][key] = dict()
        label_num_dict['entity'][key]['correct'] = 0
        label_num_dict['entity'][key]['pred'] = 0
        label_num_dict['entity'][key]['gold'] = 0
    for key in valid_pattern['role']:
        label_num_dict['role'][key] = dict()
        label_num_dict['role'][key]['correct'] = 0
        label_num_dict['role'][key]['pred'] = 0
        label_num_dict['role'][key]['gold'] = 0
    for key in valid_pattern['relation']:
        label_num_dict['relation'][key] = dict()
        label_num_dict['relation'][key]['correct'] = 0
        label_num_dict['relation'][key]['pred'] = 0
        label_num_dict['relation'][key]['gold'] = 0
    for key in valid_pattern['event']:
        label_num_dict['trigger'][key] = dict()
        label_num_dict['trigger'][key]['correct'] = 0
        label_num_dict['trigger'][key]['pred'] = 0
        label_num_dict['trigger'][key]['gold'] = 0

    # import numpy as np  
    # np.set_printoptions(threshold=np.inf)
    # print(partial_masks)
    # stop

    for output, partial_mask, l in zip(outputs, partial_masks, gather_masks):
        # torch.set_printoptions(profile="full")
        # import numpy as np
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        golds = list()
        preds = list()
        entity_golds=list()
        entity_preds=list()
        relation_golds=list()
        relation_preds=list()
        argu_golds=list()
        argu_preds=list()
        trigger_golds=list()
        trigger_preds=list()

        pred_argu_tuples = []
        pred_argu_str = []
        gold_argu_tuples = []
        gold_argu_str = []
        pred_entity_tuples = []
        gold_entity_tuples = []
        pred_trigger_tuples = []
        pred_trigger_str = []
        gold_trigger_tuples = []
        gold_trigger_str = []
        gold_relation_tuples = []
        pred_relation_tuples = []

        gold_label_dict = dict()
        pred_label_dict = dict()
        gold_label_dict['entity'] = dict()
        gold_label_dict['relation'] = dict()
        gold_label_dict['role'] = dict()
        gold_label_dict['trigger'] = dict()
        pred_label_dict['entity']  = dict()
        pred_label_dict['relation'] = dict()
        pred_label_dict['role'] = dict()
        pred_label_dict['trigger'] = dict()

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0:
                    if output[i][j] < label_size:
                        preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                    if output[i][j] in trigger_ids:
                        trigger_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        pred_trigger_tuples.append([i,j])
                        pred_trigger_str.append("{}_{}".format(i, j))
                        if labels[int(output[i][j])] not in pred_label_dict['trigger'].keys():
                            pred_label_dict['trigger'][labels[int(output[i][j])]] = []
                        pred_label_dict['trigger'][labels[int(output[i][j])]].append("{}_{}_{}".format(i, j, int(output[i][j])))

                    if output[i][j] in entity_ids:
                        entity_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        pred_entity_tuples.append([i,j])
                        if labels[int(output[i][j])] not in pred_label_dict['entity'].keys():
                            pred_label_dict['entity'][labels[int(output[i][j])]] = []
                        pred_label_dict['entity'][labels[int(output[i][j])]].append("{}_{}_{}".format(i, j, int(output[i][j])))
                    
                    if output[i][j] in relation_ids:
                        relation_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                    
                    if output[i][j] in argu_ids:
                        argu_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))

                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
                        if k in trigger_ids:
                            trigger_golds.append("{}_{}_{}".format(i, j, k))
                            gold_trigger_tuples.append([i,j])
                            gold_trigger_str.append("{}_{}".format(i, j))
                            if labels[k] not in gold_label_dict['trigger'].keys():
                                gold_label_dict['trigger'][labels[k]] = []
                            gold_label_dict['trigger'][labels[k]].append("{}_{}_{}".format(i, j, k))
                                                
                        if k in entity_ids:
                            entity_golds.append("{}_{}_{}".format(i, j, k))
                            gold_entity_tuples.append([i,j])
                            if labels[k] not in gold_label_dict['entity'].keys():
                                gold_label_dict['entity'][labels[k]] = []
                            gold_label_dict['entity'][labels[k]].append("{}_{}_{}".format(i, j, k))

        for i in range(l):
            for j in range(l):
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        if k in relation_ids:
                            relation_golds.append("{}_{}_{}".format(i, j, k))
                            entity_1=[]
                            entity_2=[]
                            for t in range(0, i + 1):
                                if [t,i] in gold_entity_tuples:
                                    if len(entity_1) > 0:
                                        entity_1.pop(0)
                                    entity_1.append([t,i])
                            for t in range(j, l):
                                if [j,t] in gold_entity_tuples:
                                    entity_2.append([j,t])
                                    break
                            # if len(entity_1)>1 or len(entity_2)>1:
                            #     print(len(entity_1),len(entity_2),i,j,k,labels[k])
                            #     print('关系对应的实体存在起始或终止位置相同的')
                            #     print(entity_1,entity_2)
                            if len(entity_1)==0 or len(entity_2)==0:
                                print('关系错误，存在非实体')
                                print(len(entity_1),len(entity_2),i,j,k,labels[k])
                            else:
                                gold_relation_tuples.append("{}_{}_{}_{}_{}".format(entity_1[0][0], entity_1[0][1], entity_2[0][0], entity_2[0][1], k))
                                if labels[k] not in gold_label_dict['relation'].keys():
                                    gold_label_dict['relation'][labels[k]] = []
                                gold_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity_1[0][0], entity_1[0][1], entity_2[0][0], entity_2[0][1], k))

                        if k in argu_ids:
                            argu_golds.append("{}_{}_{}".format(i, j, k))
                            trigger_1=[]
                            entity_2=[]

                            if 'r_' in labels[int(k)]:
                                for m in range(j, l):
                                    if [j, m] in gold_trigger_tuples:
                                        trigger_1.append([j, m])
                                        break
                                for n in range(0, i + 1):
                                    if [n, i] in gold_entity_tuples:
                                        if len(entity_2) > 0:
                                            entity_2.pop(0)
                                        entity_2.append([n ,i])
                            else:
                                for m in range(0, i + 1):
                                    if [m, i] in gold_trigger_tuples:
                                        if len(trigger_1) > 0:
                                            trigger_1.pop(0)
                                        trigger_1.append([m, i])
                                for n in range(j, l):
                                    if [j, n] in gold_entity_tuples:
                                        entity_2.append([j, n])
                                        break

                            # if len(trigger_1)>1 or len(entity_2)>1:
                            #     print(len(trigger_1),len(entity_2))
                            #     print('事件对应的实体或trigger存在起始或终止位置相同的')
                            if len(trigger_1)==0 or len(entity_2)==0:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger不存在')
                                print(gold_entity_tuples,gold_trigger_tuples)
                                print(i,j,labels[k])
                            else:
                            # gold_argu_tuples.append([trigger_1[0],entity_2[0],k])
                                gold_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1],k))
                                gold_argu_str.append("{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1]))
                                if labels[k] not in gold_label_dict['role'].keys():
                                    gold_label_dict['role'][labels[k]] = []
                                gold_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1],k))

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0 and output[i][j]<label_size:
                    k = int(output[i][j])
                    if k in relation_ids:
                        preds_entity_1 = []
                        preds_entity_2 = []
                        for m in range(0, i + 1):
                            if [m, i] in pred_entity_tuples:
                                if len(preds_entity_1) > 0:
                                    preds_entity_1.pop(0)
                                    print('预测的关系对应多个实体')
                                preds_entity_1.append([m, i])
                        for n in range(j, l):
                            if [j, n] in pred_entity_tuples:
                                preds_entity_2.append([j, n])
                                break
                        if len(preds_entity_1) >= 1 and len(preds_entity_2) >= 1:
                            pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_1[0][0], preds_entity_1[0][-1], preds_entity_2[0][0], preds_entity_2[0][-1], k))
                            if labels[k] not in pred_label_dict['relation'].keys():
                                pred_label_dict['relation'][labels[k]] = []
                            pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(preds_entity_1[0][0], preds_entity_1[0][-1], preds_entity_2[0][0], preds_entity_2[0][-1], k))
                            
                        # for [m,n] in pred_entity_tuples:
                        #     if i >= m and i <= n and (j < m or j > n):
                        #         preds_entity_1.append([m,n])
                        #     if j >= m and j <= n and (i < m or i > n):
                        #         preds_entity_2.append([m,n])
                        # if len(preds_entity_1) > 1 or len(preds_entity_2) > 1:
                        #     print('预测的关系对应多个实体')
                        # if len(preds_entity_1) >= 1 and len(preds_entity_2) >= 1:
                        #     for entity1_start, entity1_end in preds_entity_1:
                        #         for entity2_start, entity2_end in preds_entity_2:
                        #             if entity1_start < entity2_start:
                        #                 pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                        #             else:
                        #                 pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                            # if preds_entity_1[0][0] < preds_entity_2[0][0]:
                            #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_1[0][0], preds_entity_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                            # else:
                            #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_2[0][0], preds_entity_2[0][1], preds_entity_1[0][0], preds_entity_1[0][1], k))

                    if k in argu_ids:
                        preds_trigger_1=[]
                        preds_entity_2=[]
                        if 'r_' in labels[int(k)]:
                            # for [m,n] in pred_trigger_tuples:
                            #     if m<=j and j<=n:
                            #         preds_trigger_1.append([m,n])
                            # for [m,n] in pred_entity_tuples:
                            #     if m<=i and i<=n:
                            #         preds_entity_2.append([m,n])
                            for m in range(j, l):
                                if [j, m] in pred_trigger_tuples:
                                    preds_trigger_1.append([j, m])
                                    break
                            for n in range(0, i + 1):
                                if [n, i] in pred_entity_tuples:
                                    if len(preds_entity_2) > 0:
                                        print('预测的事件对应多个触发器或实体')
                                        preds_entity_2.pop(0)
                                    preds_entity_2.append([n, i])
                        else:
                            # for [m,n] in pred_trigger_tuples:
                            #     if m<=i and i<=n:
                            #         preds_trigger_1.append([m,n])
                            # for [m,n] in pred_entity_tuples:
                            #     if m<=j and j<=n:
                            #         preds_entity_2.append([m,n])
                            for m in range(0, i + 1):
                                if [m, i] in pred_trigger_tuples:
                                    if len(preds_trigger_1) > 0:
                                        preds_trigger_1.pop(0)
                                    preds_trigger_1.append([m, i])
                            for n in range(j, l):
                                if [j, n] in pred_entity_tuples:
                                    preds_entity_2.append([j, n])
                                    break
                        if len(preds_trigger_1) >= 1 and len(preds_entity_2) >= 1:
                            pred_argu_tuples.append("{}_{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][-1], preds_entity_2[0][0], preds_entity_2[0][-1], k))
                            pred_argu_str.append("{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][-1], preds_entity_2[0][0], preds_entity_2[0][-1]))
                            if labels[k] not in pred_label_dict['role'].keys():
                                pred_label_dict['role'][labels[k]] = []
                            pred_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][-1], preds_entity_2[0][0], preds_entity_2[0][-1], k))

   
                        # if len(preds_trigger_1) > 1 or len(preds_entity_2) > 1:
                        #     print('预测的事件对应多个触发器或实体')
                        #     print(i,j,preds_trigger_1,preds_entity_2)
                        # if len(preds_trigger_1) >= 1 and len(preds_entity_2) >= 1:
                        #     for trigger_start, trigger_end in preds_trigger_1:
                        #         for entity_start, entity_end in preds_entity_2:
                        #             pred_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                        #             pred_argu_str.append("{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end))
                            # pred_argu_tuples.append("{}_{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                            # pred_argu_str.append("{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1]))

        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

        pred_trigger_num += len(trigger_preds)
        gold_trigger_num += len(trigger_golds)
        trigger_class_num += len(set(trigger_preds).intersection(set(trigger_golds)))
        trigger_idn_num += len(set(pred_trigger_str).intersection(set(gold_trigger_str)))

        pred_ent_num += len(entity_preds)
        gold_ent_num += len(entity_golds)
        ent_match_num += len(set(entity_preds).intersection(set(entity_golds)))

        pred_rel_num += len(relation_preds)
        gold_rel_num += len(relation_golds)
        rel_match_num += len(set(gold_relation_tuples).intersection(set(pred_relation_tuples)))

        pred_arg_num += len(argu_preds)
        gold_arg_num += len(argu_golds)
        arg_class_num += len(set(gold_argu_tuples).intersection(set(pred_argu_tuples)))
        arg_idn_num += len(set(pred_argu_str).intersection(set(gold_argu_str)))

        # print(trigger_preds)
        # print(trigger_golds)
        # print(pred_trigger_str)
        # print(gold_trigger_str)
        # print(entity_preds)
        # print(entity_golds)
        # print(gold_relation_tuples)
        # print(pred_relation_tuples)
        # print(relation_preds)
        # print(argu_golds)
        # print(argu_preds)
        # print(pred_argu_str)
        # print(gold_argu_str)
        # print(pred_argu_tuples)
        # print(gold_argu_tuples)
        print('trigger_pred',trigger_preds)
        print('trigger_gold',trigger_golds)
        print('entity_pred',entity_preds)
        print('entity_gold',entity_golds)
        print('relation_preds',relation_preds)
        print('relation_golds',relation_golds)
        print('pred_relation_tuples',pred_relation_tuples)
        print('gold_relation_tuples',gold_relation_tuples)
        print('argu_preds',argu_preds)
        print('argu_golds',argu_golds)
        print('pred_argu_tuples',pred_argu_tuples)
        print('gold_argu_tuples',gold_argu_tuples)
        
        for key in pred_label_dict['entity'].keys():
            label_num_dict['entity'][key]['pred'] += len(pred_label_dict['entity'][key])
        for key in gold_label_dict['entity'].keys():
            label_num_dict['entity'][key]['gold'] += len(gold_label_dict['entity'][key])
            if key in list(pred_label_dict['entity'].keys()):
                label_num_dict['entity'][key]['correct'] += len(set(pred_label_dict['entity'][key]).intersection(set(gold_label_dict['entity'][key])))


        for key in pred_label_dict['relation'].keys():
            label_num_dict['relation'][key]['pred'] += len(pred_label_dict['relation'][key])
        for key in gold_label_dict['relation'].keys():
            label_num_dict['relation'][key]['gold'] += len(gold_label_dict['relation'][key])
            if key in list(pred_label_dict['relation'].keys()):
                label_num_dict['relation'][key]['correct'] += len(set(pred_label_dict['relation'][key]).intersection(set(gold_label_dict['relation'][key])))


        for key in pred_label_dict['role'].keys():
            label_num_dict['role'][key]['pred'] += len(pred_label_dict['role'][key])
        for key in gold_label_dict['role'].keys():
            label_num_dict['role'][key]['gold'] += len(gold_label_dict['role'][key])
            if key in list(pred_label_dict['role'].keys()):
                label_num_dict['role'][key]['correct'] += len(set(pred_label_dict['role'][key]).intersection(set(gold_label_dict['role'][key])))

        for key in pred_label_dict['trigger'].keys():
            label_num_dict['trigger'][key]['pred'] += len(pred_label_dict['trigger'][key])
        for key in gold_label_dict['trigger'].keys():
            label_num_dict['trigger'][key]['gold'] += len(gold_label_dict['trigger'][key])
            if key in list(pred_label_dict['trigger'].keys()):
                label_num_dict['trigger'][key]['correct'] += len(set(pred_label_dict['trigger'][key]).intersection(set(gold_label_dict['trigger'][key])))
        
    for key in ['entity', 'trigger', 'relation', 'role']:
        for lab in label_num_dict[key].keys():
            if label_num_dict[key][lab]['pred'] == 0 :
                label_num_dict[key][lab]['precision'] = 0
            else:
                label_num_dict[key][lab]['precision'] = label_num_dict[key][lab]['correct'] / label_num_dict[key][lab]['pred']
            if label_num_dict[key][lab]['gold'] == 0:
                label_num_dict[key][lab]['recall'] = 0
            else:
                label_num_dict[key][lab]['recall'] = label_num_dict[key][lab]['correct'] / label_num_dict[key][lab]['gold']
            if label_num_dict[key][lab]['precision'] + label_num_dict[key][lab]['recall'] == 0:
                label_num_dict[key][lab]['f1'] = 0
            else:
                label_num_dict[key][lab]['f1'] = 2 * label_num_dict[key][lab]['precision'] * label_num_dict[key][lab]['recall'] / \
                                        (label_num_dict[key][lab]['precision'] + label_num_dict[key][lab]['recall'])
            print(key, ' ', lab,' ', 'precision', label_num_dict[key][lab]['precision'])
            print(key, ' ', lab, ' ', 'recall', label_num_dict[key][lab]['recall'])
            print(key, ' ', lab, ' ', 'f1', label_num_dict[key][lab]['f1'])

    return correct, pred_count, gold_count, pred_ent_num, gold_ent_num, ent_match_num, pred_trigger_num, gold_trigger_num, trigger_idn_num, \
    pred_trigger_num, gold_trigger_num, trigger_class_num, \
    pred_rel_num, gold_rel_num, rel_match_num, pred_arg_num, gold_arg_num, arg_idn_num, \
    arg_class_num

def eval_2(args, outputs, partial_masks, labels, gather_masks, valid_pattern):
    '''
    头-尾，解析双仿射和树
    '''
    label_size=len(labels)
    argu_ids=[labels.index(item) for item in valid_pattern['role']]
    entity_ids=[labels.index(item) for item in valid_pattern['ner']]
    relation_ids=[labels.index(item) for item in valid_pattern['relation']]
    trigger_ids=[labels.index(item) for item in valid_pattern['event']]

    correct, pred_count, gold_count = 0, 0, 0
    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0

    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs_ = outputs[0].cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()

    label_num_dict = dict()
    label_num_dict['entity'] = dict()
    label_num_dict['relation'] = dict()
    label_num_dict['role'] = dict()
    label_num_dict['trigger'] = dict()
    for key in valid_pattern['ner']:
        label_num_dict['entity'][key] = dict()
        label_num_dict['entity'][key]['correct'] = 0
        label_num_dict['entity'][key]['pred'] = 0
        label_num_dict['entity'][key]['gold'] = 0
    for key in valid_pattern['role']:
        label_num_dict['role'][key] = dict()
        label_num_dict['role'][key]['correct'] = 0
        label_num_dict['role'][key]['pred'] = 0
        label_num_dict['role'][key]['gold'] = 0
    for key in valid_pattern['relation']:
        label_num_dict['relation'][key] = dict()
        label_num_dict['relation'][key]['correct'] = 0
        label_num_dict['relation'][key]['pred'] = 0
        label_num_dict['relation'][key]['gold'] = 0
    for key in valid_pattern['event']:
        label_num_dict['trigger'][key] = dict()
        label_num_dict['trigger'][key]['correct'] = 0
        label_num_dict['trigger'][key]['pred'] = 0
        label_num_dict['trigger'][key]['gold'] = 0
    # import numpy as np  
    # np.set_printoptions(threshold=np.inf)
    # print(partial_masks)
    # stop
    start = 0
    for output, partial_mask, l in zip(outputs_, partial_masks, gather_masks):
        # torch.set_printoptions(profile="full")
        # import numpy as np
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        golds = list()
        preds = list()
        entity_golds=list()
        entity_preds=list()
        relation_golds=list()
        relation_preds=list()
        argu_golds=list()
        argu_preds=list()
        trigger_golds=list()
        trigger_preds=list()

        pred_argu_tuples = []
        pred_argu_str = []
        gold_argu_tuples = []
        gold_argu_str = []
        pred_entity_tuples = []
        gold_entity_tuples = []
        pred_trigger_tuples = []
        pred_trigger_str = []
        gold_trigger_tuples = []
        gold_trigger_str = []
        gold_relation_tuples = []
        pred_relation_tuples = []

        gold_label_dict = dict()
        pred_label_dict = dict()
        gold_label_dict['entity'] = dict()
        gold_label_dict['relation'] = dict()
        gold_label_dict['role'] = dict()
        gold_label_dict['trigger'] = dict()
        pred_label_dict['entity']  = dict()
        pred_label_dict['relation'] = dict()
        pred_label_dict['role'] = dict()
        pred_label_dict['trigger'] = dict()

        for i in range(l):
            for j in range(l):
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
                        if k in trigger_ids:
                            trigger_golds.append("{}_{}_{}".format(i, j, k))
                            gold_trigger_tuples.append([i,j])
                            gold_trigger_str.append("{}_{}".format(i, j))
                            if labels[k] not in gold_label_dict['trigger'].keys():
                                gold_label_dict['trigger'][labels[k]] = []
                            gold_label_dict['trigger'][labels[k]].append("{}_{}_{}".format(i, j, k))
                        
                        if k in entity_ids:
                            entity_golds.append("{}_{}_{}".format(i, j, k))
                            gold_entity_tuples.append([i,j])
                            if labels[k] not in gold_label_dict['entity'].keys():
                                gold_label_dict['entity'][labels[k]] = []
                            gold_label_dict['entity'][labels[k]].append("{}_{}_{}".format(i, j, k))

        for k, i, j in zip(*np.where(output > 0)):
            if k < label_size:
                preds.append("{}_{}_{}".format(i, j, int(k)))
            if k in trigger_ids:
                trigger_preds.append("{}_{}_{}".format(i, j, int(k)))
                pred_trigger_tuples.append([i,j])
                pred_trigger_str.append("{}_{}".format(i, j))
                if labels[int(k)] not in pred_label_dict['trigger'].keys():
                    pred_label_dict['trigger'][labels[int(k)]] = []
                pred_label_dict['trigger'][labels[int(k)]].append("{}_{}_{}".format(i, j, int(k)))

            if k in entity_ids:
                entity_preds.append("{}_{}_{}".format(i, j, int(k)))
                pred_entity_tuples.append([i,j])
                if labels[int(k)] not in pred_label_dict['entity'].keys():
                    pred_label_dict['entity'][labels[int(k)]] = []
                pred_label_dict['entity'][labels[int(k)]].append("{}_{}_{}".format(i, j, int(k)))
            
            if k in relation_ids:
                relation_preds.append("{}_{}_{}".format(i, j, int(k)))
            
            if k in argu_ids:
                argu_preds.append("{}_{}_{}".format(i, j, int(k)))

        for i in range(l):
            for j in range(l):
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        if k in relation_ids:
                            relation_golds.append("{}_{}_{}".format(i, j, k))
                            entity_1=[]
                            entity_2=[]
                            for t in range(i,j):
                                if [i,t] in gold_entity_tuples:
                                    entity_1.append([i,t])
                            for t in range(i,j+1):
                                if [t,j] in gold_entity_tuples:
                                    entity_2.append([t,j])
                            if len(entity_1)>1 or len(entity_2)>1:
                                print(len(entity_1),len(entity_2),i,j,k,labels[k])
                                print('关系对应的实体存在起始或终止位置相同的')
                                print(entity_1,entity_2)
                            if len(entity_1)==0 or len(entity_2)==0:
                                print('关系错误，存在非实体')
                                print(len(entity_1),len(entity_2),i,j,k,labels[k])
                            else:
                                gold_relation_tuples.append("{}_{}_{}_{}_{}".format(entity_1[0][0], entity_1[0][1], entity_2[0][0], entity_2[0][1], k))
                                if labels[k] not in gold_label_dict['relation'].keys():
                                    gold_label_dict['relation'][labels[k]] = []
                                gold_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity_1[0][0], entity_1[0][1], entity_2[0][0], entity_2[0][1], k))

                        if k in argu_ids:
                            argu_golds.append("{}_{}_{}".format(i, j, k))
                            trigger_1=[]
                            entity_2=[]

                            for t in range(i,j):
                                if [i,t] in gold_trigger_tuples:
                                    trigger_1.append([i,t])
                            for m in range(i,j+1):
                                if [m,j] in gold_entity_tuples:
                                    entity_2.append([m,j])
                            if len(trigger_1)==0 and len(entity_2)==0:
                                for t in range(i,j):
                                    if [i,t] in gold_entity_tuples:
                                        entity_2.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_trigger_tuples:
                                        trigger_1.append([m,j])
                            if len(trigger_1)==0 or len(entity_2)==0:
                                for t in range(i,j+1):
                                    if [i,t] in gold_entity_tuples:
                                        entity_2.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_trigger_tuples:
                                        trigger_1.append([m,j])
                            elif len(trigger_1)==0 or len(entity_2)==0:
                                for t in range(i,j+1):
                                    if [i,t] in gold_trigger_tuples:
                                        trigger_1.append([i,t])
                                for m in range(i,j+1):
                                    if [m,j] in gold_entity_tuples:
                                        entity_2.append([m,j])
                            if len(trigger_1)>1 or len(entity_2)>1:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger存在起始或终止位置相同的')
                            if len(trigger_1)==0 or len(entity_2)==0:
                                print(len(trigger_1),len(entity_2))
                                print('事件对应的实体或trigger不存在')
                                print(gold_entity_tuples,gold_trigger_tuples)
                                print(i,j,labels[k])
                            else:
                            # gold_argu_tuples.append([trigger_1[0],entity_2[0],k])
                                gold_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1],k))
                                gold_argu_str.append("{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1]))
                                if labels[k] not in gold_label_dict['role'].keys():
                                    gold_label_dict['role'][labels[k]] = []
                                gold_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(trigger_1[0][0],trigger_1[0][1],entity_2[0][0],entity_2[0][1],k))

        for k, i, j in zip(*np.where(output > 0)):
            # print(i,j,k, output[k][i][j],labels[40])
            if k in relation_ids:
                preds_entity_1 = []
                preds_entity_2 = []
                for [m,n] in pred_entity_tuples:
                    if i >= m and i <= n and (j < m or j > n):
                        preds_entity_1.append([m,n])
                    if j >= m and j <= n and (i < m or i > n):
                        preds_entity_2.append([m,n])
                if len(preds_entity_1) > 1 or len(preds_entity_2) > 1:
                    # print('预测的关系对应多个实体')
                    pass
                if len(preds_entity_1) >= 1 and len(preds_entity_2) >= 1:
                    for entity1_start, entity1_end in preds_entity_1:
                        for entity2_start, entity2_end in preds_entity_2:
                            if entity1_start < entity2_start:
                                pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                                if labels[k] not in pred_label_dict['relation'].keys():
                                    pred_label_dict['relation'][labels[k]] = []
                                pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                            else:
                                pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                                if labels[k] not in pred_label_dict['relation'].keys():
                                    pred_label_dict['relation'][labels[k]] = []
                                pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                    # if preds_entity_1[0][0] < preds_entity_2[0][0]:
                    #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_1[0][0], preds_entity_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                    # else:
                    #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_2[0][0], preds_entity_2[0][1], preds_entity_1[0][0], preds_entity_1[0][1], k))

            if k in argu_ids:
                preds_trigger_1=[]
                preds_entity_2=[]
                if 'r_' in labels[k]:
                    for [m,n] in pred_trigger_tuples:
                        if m<=j and j<=n:
                            preds_trigger_1.append([m,n])
                    for [m,n] in pred_entity_tuples:
                        if m<=i and i<=n:
                            preds_entity_2.append([m,n])
                else:
                    for [m,n] in pred_trigger_tuples:
                        if m<=i and i<=n:
                            preds_trigger_1.append([m,n])
                    for [m,n] in pred_entity_tuples:
                        if m<=j and j<=n:
                            preds_entity_2.append([m,n])
                    if len(preds_trigger_1) == 0 and len(preds_entity_2) == 0:
                        for [m,n] in pred_trigger_tuples:
                            if m<=j and j<=n:
                                preds_trigger_1.append([m,n])
                        for [m,n] in pred_entity_tuples:
                            if m<=i and i<=n:
                                preds_entity_2.append([m,n])

                if len(preds_trigger_1) > 1 or len(preds_entity_2) > 1:
                    pass
                    # print('预测的事件对应多个触发器或实体')
                    # print(i,j,preds_trigger_1,preds_entity_2)
                if len(preds_trigger_1) >= 1 and len(preds_entity_2) >= 1:
                    for trigger_start, trigger_end in preds_trigger_1:
                        for entity_start, entity_end in preds_entity_2:
                            pred_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                            pred_argu_str.append("{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end))
                            if labels[k] not in pred_label_dict['role'].keys():
                                pred_label_dict['role'][labels[k]] = []
                            pred_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                    # pred_argu_tuples.append("{}_{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                    # pred_argu_str.append("{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1]))
        '''树'''
        def get_tree(args, outputs, partial_masks, labels, gather_masks, valid_pattern, start):
            outputs = outputs[1].cpu().numpy()
            preds = list()
            entity_preds=list()
            relation_preds=list()
            argu_preds=list()
            trigger_preds=list()

            pred_argu_tuples = []
            pred_argu_str = []
            pred_entity_tuples = []
            pred_trigger_tuples = []
            pred_trigger_str = []
            pred_relation_tuples = []

            pred_label_dict = dict()
            pred_label_dict['entity']  = dict()
            pred_label_dict['relation'] = dict()
            pred_label_dict['role'] = dict()
            pred_label_dict['trigger'] = dict()
            output, l = outputs[start], gather_masks[start]
            for i in range(l):
                for j in range(l):
                    if output[i][j] >= 0:
                        if output[i][j] < label_size:
                            preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        if output[i][j] in trigger_ids:
                            trigger_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                            pred_trigger_tuples.append([i,j])
                            pred_trigger_str.append("{}_{}".format(i, j))
                            if labels[int(output[i][j])] not in pred_label_dict['trigger'].keys():
                                pred_label_dict['trigger'][labels[int(output[i][j])]] = []
                            pred_label_dict['trigger'][labels[int(output[i][j])]].append("{}_{}_{}".format(i, j, int(output[i][j])))

                        if output[i][j] in entity_ids:
                            entity_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                            pred_entity_tuples.append([i,j])
                            if labels[int(output[i][j])] not in pred_label_dict['entity'].keys():
                                pred_label_dict['entity'][labels[int(output[i][j])]] = []
                            pred_label_dict['entity'][labels[int(output[i][j])]].append("{}_{}_{}".format(i, j, int(output[i][j])))
                        
                        if output[i][j] in relation_ids:
                            relation_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        
                        if output[i][j] in argu_ids:
                            argu_preds.append("{}_{}_{}".format(i, j, int(output[i][j])))

            for i in range(l):
                for j in range(l):
                    if output[i][j] >= 0 and output[i][j]<label_size:
                        k = int(output[i][j])
                        if k in relation_ids:
                            preds_entity_1 = []
                            preds_entity_2 = []
                            for [m,n] in pred_entity_tuples:
                                if i >= m and i <= n and (j < m or j > n):
                                    preds_entity_1.append([m,n])
                                if j >= m and j <= n and (i < m or i > n):
                                    preds_entity_2.append([m,n])
                            if len(preds_entity_1) > 1 or len(preds_entity_2) > 1:
                                print('预测的关系对应多个实体')
                            if len(preds_entity_1) >= 1 and len(preds_entity_2) >= 1:
                                for entity1_start, entity1_end in preds_entity_1:
                                    for entity2_start, entity2_end in preds_entity_2:
                                        if entity1_start < entity2_start:
                                            pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                                            if labels[k] not in pred_label_dict['relation'].keys():
                                                pred_label_dict['relation'][labels[k]] = []
                                            pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity1_start, entity1_end, entity2_start, entity2_end, k))
                                        else:
                                            pred_relation_tuples.append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                                            if labels[k] not in pred_label_dict['relation'].keys():
                                                pred_label_dict['relation'][labels[k]] = []
                                            pred_label_dict['relation'][labels[k]].append("{}_{}_{}_{}_{}".format(entity2_start, entity2_end, entity1_start, entity1_end, k))
                                # if preds_entity_1[0][0] < preds_entity_2[0][0]:
                                #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_1[0][0], preds_entity_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                                # else:
                                #     pred_relation_tuples.append("{}_{}_{}_{}_{}".format(preds_entity_2[0][0], preds_entity_2[0][1], preds_entity_1[0][0], preds_entity_1[0][1], k))

                        if k in argu_ids:
                            preds_trigger_1=[]
                            preds_entity_2=[]
                            if 'r_' in labels[k]:
                                for [m,n] in pred_trigger_tuples:
                                    if m<=j and j<=n:
                                        preds_trigger_1.append([m,n])
                                for [m,n] in pred_entity_tuples:
                                    if m<=i and i<=n:
                                        preds_entity_2.append([m,n])
                            else:
                                for [m,n] in pred_trigger_tuples:
                                    if m<=i and i<=n:
                                        preds_trigger_1.append([m,n])
                                for [m,n] in pred_entity_tuples:
                                    if m<=j and j<=n:
                                        preds_entity_2.append([m,n])
    
                            if len(preds_trigger_1) > 1 or len(preds_entity_2) > 1:
                                print('预测的事件对应多个触发器或实体')
                                print(i,j,preds_trigger_1,preds_entity_2)
                            if len(preds_trigger_1) >= 1 and len(preds_entity_2) >= 1:
                                for trigger_start, trigger_end in preds_trigger_1:
                                    for entity_start, entity_end in preds_entity_2:
                                        pred_argu_tuples.append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                                        pred_argu_str.append("{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end))
                                        if labels[k] not in pred_label_dict['role'].keys():
                                            pred_label_dict['role'][labels[k]] = []
                                        pred_label_dict['role'][labels[k]].append("{}_{}_{}_{}_{}".format(trigger_start, trigger_end, entity_start, entity_end, k))
                                # pred_argu_tuples.append("{}_{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1], k))
                                # pred_argu_str.append("{}_{}_{}_{}".format(preds_trigger_1[0][0], preds_trigger_1[0][1], preds_entity_2[0][0], preds_entity_2[0][1]))
            return preds, trigger_preds, pred_trigger_str, entity_preds, relation_preds, pred_relation_tuples, argu_preds, pred_argu_tuples, pred_argu_str

        preds_tree, trigger_preds_tree, pred_trigger_str_tree, entity_preds_tree, relation_preds_tree, pred_relation_tuples_tree, argu_preds_tree, pred_argu_tuples_tree, pred_argu_str_tree \
         = get_tree(args, outputs, partial_masks, labels, gather_masks, valid_pattern, start)
        start += 1

        preds = set(preds + preds_tree)
        trigger_preds = set(trigger_preds + trigger_preds_tree)
        pred_trigger_str = set(pred_trigger_str + pred_trigger_str_tree)
        entity_preds = set(entity_preds + entity_preds_tree)
        relation_preds = set(relation_preds + relation_preds_tree)
        pred_relation_tuples = set(pred_relation_tuples + pred_relation_tuples_tree)
        argu_preds = set(argu_preds + argu_preds_tree)
        pred_argu_tuples = set(pred_argu_tuples + pred_argu_tuples_tree)
        pred_argu_str = set(pred_argu_str + pred_argu_str_tree)

        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

        pred_trigger_num += len(trigger_preds)
        gold_trigger_num += len(trigger_golds)
        trigger_class_num += len(set(trigger_preds).intersection(set(trigger_golds)))
        trigger_idn_num += len(set(pred_trigger_str).intersection(set(gold_trigger_str)))

        pred_ent_num += len(entity_preds)
        gold_ent_num += len(entity_golds)
        ent_match_num += len(set(entity_preds).intersection(set(entity_golds)))

        pred_rel_num += len(relation_preds)
        gold_rel_num += len(relation_golds)
        rel_match_num += len(set(gold_relation_tuples).intersection(set(pred_relation_tuples)))

        pred_arg_num += len(argu_preds)
        gold_arg_num += len(argu_golds)
        arg_class_num += len(set(gold_argu_tuples).intersection(set(pred_argu_tuples)))
        arg_idn_num += len(set(pred_argu_str).intersection(set(gold_argu_str)))

        for key in pred_label_dict['entity'].keys():
            label_num_dict['entity'][key]['pred'] += len(pred_label_dict['entity'][key])
        for key in gold_label_dict['entity'].keys():
            label_num_dict['entity'][key]['gold'] += len(gold_label_dict['entity'][key])
            if key in list(pred_label_dict['entity'].keys()):
                label_num_dict['entity'][key]['correct'] += len(set(pred_label_dict['entity'][key]).intersection(set(gold_label_dict['entity'][key])))


        for key in pred_label_dict['relation'].keys():
            label_num_dict['relation'][key]['pred'] += len(pred_label_dict['relation'][key])
        for key in gold_label_dict['relation'].keys():
            label_num_dict['relation'][key]['gold'] += len(gold_label_dict['relation'][key])
            if key in list(pred_label_dict['relation'].keys()):
                label_num_dict['relation'][key]['correct'] += len(set(pred_label_dict['relation'][key]).intersection(set(gold_label_dict['relation'][key])))


        for key in pred_label_dict['role'].keys():
            label_num_dict['role'][key]['pred'] += len(pred_label_dict['role'][key])
        for key in gold_label_dict['role'].keys():
            label_num_dict['role'][key]['gold'] += len(gold_label_dict['role'][key])
            if key in list(pred_label_dict['role'].keys()):
                label_num_dict['role'][key]['correct'] += len(set(pred_label_dict['role'][key]).intersection(set(gold_label_dict['role'][key])))

        for key in pred_label_dict['trigger'].keys():
            label_num_dict['trigger'][key]['pred'] += len(pred_label_dict['trigger'][key])
        for key in gold_label_dict['trigger'].keys():
            label_num_dict['trigger'][key]['gold'] += len(gold_label_dict['trigger'][key])
            if key in list(pred_label_dict['trigger'].keys()):
                label_num_dict['trigger'][key]['correct'] += len(set(pred_label_dict['trigger'][key]).intersection(set(gold_label_dict['trigger'][key])))
        
        # print(trigger_preds)
        # print(trigger_golds)
        # print(pred_trigger_str)
        # print(gold_trigger_str)
        # print(entity_preds)
        # print(entity_golds)
        # print(gold_relation_tuples)
        # print(pred_relation_tuples)
        # print(relation_preds)
        # print(argu_golds)
        # print(argu_preds)
        # print(pred_argu_str)
        # print(gold_argu_str)
        # print(pred_argu_tuples)
        # print(gold_argu_tuples)
        # print('trigger_pred',trigger_preds)
        # print('trigger_gold',trigger_golds)
        # print('entity_pred',entity_preds)
        # print('entity_gold',entity_golds)
        # print('relation_preds',relation_preds)
        # print('relation_golds',relation_golds)
        # print('pred_relation_tuples',pred_relation_tuples)
        # print('gold_relation_tuples',gold_relation_tuples)
        # print('argu_preds',argu_preds)
        # print('argu_golds',argu_golds)
        # print('pred_argu_tuples',pred_argu_tuples)
        # print('gold_argu_tuples',gold_argu_tuples)
    
    for key in ['entity', 'trigger', 'relation', 'role']:
        for lab in label_num_dict[key].keys():
            if label_num_dict[key][lab]['pred'] == 0 :
                label_num_dict[key][lab]['precision'] = 0
            else:
                label_num_dict[key][lab]['precision'] = label_num_dict[key][lab]['correct'] / label_num_dict[key][lab]['pred']
            if label_num_dict[key][lab]['gold'] == 0:
                label_num_dict[key][lab]['recall'] = 0
            else:
                label_num_dict[key][lab]['recall'] = label_num_dict[key][lab]['correct'] / label_num_dict[key][lab]['gold']
            if label_num_dict[key][lab]['precision'] + label_num_dict[key][lab]['recall'] == 0:
                label_num_dict[key][lab]['f1'] = 0
            else:
                label_num_dict[key][lab]['f1'] = 2 * label_num_dict[key][lab]['precision'] * label_num_dict[key][lab]['recall'] / \
                                        (label_num_dict[key][lab]['precision'] + label_num_dict[key][lab]['recall'])
            print(key, ' ', lab,' ', 'precision', label_num_dict[key][lab]['precision'])
            print(key, ' ', lab, ' ', 'recall', label_num_dict[key][lab]['recall'])
            print(key, ' ', lab, ' ', 'f1', label_num_dict[key][lab]['f1'])

    return correct, pred_count, gold_count, pred_ent_num, gold_ent_num, ent_match_num, pred_trigger_num, gold_trigger_num, trigger_idn_num, \
    pred_trigger_num, gold_trigger_num, trigger_class_num, \
    pred_rel_num, gold_rel_num, rel_match_num, pred_arg_num, gold_arg_num, arg_idn_num, \
    arg_class_num