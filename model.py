import torch
import torch.nn.functional as F
import torch_model_utils as tmu

from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel, BertForMaskedLM, BertTokenizer, RobertaForMaskedLM, RobertaTokenizer
from torch import nn

from tree_crf_layer import TreeCRFLayer
from parser import Bilinear, BiAffine, DeepBiaffine
from torch.cuda.amp import autocast, GradScaler
import json

class Fusion(nn.Module):
    def __init__(self, n_conditions, seq_len, hidden_size):
        super().__init__()
        self.weight_classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_conditions),
            nn.Softmax(dim=-1)
        )
        self.masks = torch.nn.Embedding(n_conditions, hidden_size)
    
    def forward(self, x, c):
        weight = self.weight_classifier(c)
        x = x.unsqueeze(2).expand(-1, -1, 5, -1)
        x = x * self.masks.weight.unsqueeze(0)
        x = x* weight.unsqueeze(-1)
        return x.mean(dim=2) 

def partial_mask_to_targets(mask):
    device = mask.device
    label_size = mask.size(-1)
    ind = 1 + torch.arange(label_size).to(device).view(1, 1, 1, -1)
    trees = (mask * ind).sum(dim=-1)
    trees = trees - 1
    tree_rej_ind = trees == -1
    # tree_rej_ind = trees < 40
    trees[tree_rej_ind] = label_size - 1
    return trees

# class PartialPCFG(BertPreTrainedModel):
class PartialPCFG(RobertaPreTrainedModel):

    def __init__(self, config):
        super(PartialPCFG, self).__init__(config)

        self.lambda_ent = config.lambda_ent  # try [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        self.label_size = config.label_size
        self.structure_smoothing = config.structure_smoothing_p < 1.0

        self.use_crf = config.use_crf
        if (self.use_crf is False): assert (config.latent_label_size == 1)

        self.bert = BertModel(config)
        self.tokenizer=BertTokenizer.from_pretrained('./bert-base-cased')
        self.mlm = BertForMaskedLM.from_pretrained('./bert-base-cased')
        # self.bert = RobertaModel(config)
        # self.bert = RobertaModel.from_pretrained('./roberta-large')
        # self.tokenizer = RobertaTokenizer.from_pretrained('./roberta-large')
        # self.mlm = RobertaForMaskedLM.from_pretrained('./roberta-large')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fusion_module = Fusion(
            5,
            config.max_seq_length,
            config.hidden_size
        )
        # self.types = []
        if 'genia' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['None', 'G#RNA', 'G#protein', 'G#DNA', 'G#cell_type', 'G#cell_line']
        elif 'conll2003' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['LOC', 'PER', 'ORG', 'MISC']
        elif 'ACE' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['Personnel.End-Position', 'PER-SOC', 'r_Prosecutor', 'r_Adjudicator', 'Business.Merge-Org', 'Org', 'r_PART-WHOLE', 'ART', 'Conflict.Demonstrate', 'Attacker', 'r_Vehicle', 'Justice.Arrest-Jail', 'PART-WHOLE', 'Conflict.Attack', 'Person', 'r_Giver', 'Business.Declare-Bankruptcy', 'r_Plaintiff', 'r_Target', 'Justice.Charge-Indict', 'r_Buyer', 'Defendant', 'r_Defendant', 'Justice.Trial-Hearing', 'Personnel.Nominate', 'r_Entity', 'Justice.Appeal', 'Justice.Extradite', 'Transaction.Transfer-Ownership', 'Justice.Sue', 'ORG-AFF', 'r_Person', 'r_GEN-AFF', 'Contact.Meet', 'Entity', 'WEA', 'r_Place', 'r_Attacker', 'Plaintiff', 'r_Beneficiary', 'Justice.Sentence', 'Life.Divorce', 'r_ART', 'Contact.Phone-Write', 'Movement.Transport', 'Business.Start-Org', 'VEH', 'r_Org', 'Beneficiary', 'Personnel.Elect', 'Justice.Execute', 'LOC', 'Justice.Release-Parole', 'Buyer', 'PHYS', 'Justice.Fine', 'Personnel.Start-Position', 'Giver', 'Life.Marry', 'r_PER-SOC', 'r_Origin', 'r_Instrument', 'Agent', 'Target', 'Justice.Acquit', 'Justice.Convict', 'r_Victim', 'r_Artifact', 'Origin', 'Vehicle', 'Justice.Pardon', 'r_Seller', 'ORG', 'Business.End-Org', 'r_ORG-AFF', 'Destination', 'Prosecutor', 'Seller', 'Life.Be-Born', 'GEN-AFF', 'r_Recipient', 'Adjudicator', 'Recipient', 'Life.Injure', 'r_Agent', 'PER', 'Transaction.Transfer-Money', 'Artifact', 'FAC', 'r_Destination', 'Life.Die', 'r_PHYS', 'Instrument', 'GPE', 'Place', 'Victim']
        elif 'ACE_add' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['GPE', 'Plaintiff', 'Life:Die', 'r_Beneficiary', 'Conflict:Demonstrate', 'r_Place', 'Attacker', 'Justice:Sue', 'LOC', 'Business:Merge-Org', 'r_Destination', 'Justice:Release-Parole', 'Buyer', 'Life:Divorce', 'Life:Injure', 'r_Instrument', 'VEH', 'Business:Start-Org', 'Giver', 'PART-WHOLE', 'Transaction:Transfer-Ownership', 'PHYS', 'Prosecutor', 'r_Recipient', 'r_PHYS', 'Justice:Appeal', 'PER-SOC', 'Justice:Acquit', 'r_Seller', 'ART', 'Justice:Fine', 'r_Defendant', 'Place', 'Justice:Pardon', 'Contact:Meet', 'Origin', 'Movement:Transport', 'r_Agent', 'Justice:Trial-Hearing', 'Personnel:End-Position', 'Recipient', 'r_PER-SOC', 'Justice:Arrest-Jail', 'WEA', 'Justice:Convict', 'PER', 'r_Buyer', 'Person', 'ORG-AFF', 'r_ORG-AFF', 'Instrument', 'r_Prosecutor', 'Transaction:Transfer-Money', 'r_Entity', 'Adjudicator', 'r_Artifact', 'r_Giver', 'Business:End-Org', 'r_Target', 'Personnel:Nominate', 'ORG', 'r_Victim', 'Vehicle', 'Defendant', 'r_Origin', 'Victim', 'r_Org', 'Personnel:Start-Position', 'Seller', 'Personnel:Elect', 'Conflict:Attack', 'GEN-AFF', 'r_ART', 'r_Attacker', 'r_Vehicle', 'Target', 'Agent', 'r_Adjudicator', 'r_PART-WHOLE', 'Entity', 'r_Plaintiff', 'Beneficiary', 'FAC', 'Justice:Charge-Indict', 'Justice:Sentence', 'Artifact', 'Justice:Extradite', 'Contact:Phone-Write', 'r_Person', 'Org', 'Life:Be-Born', 'r_GEN-AFF', 'Life:Marry', 'Destination', 'Business:Declare-Bankruptcy', 'Justice:Execute']
        elif 'ACE_add_event' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['r_Defendant', 'Victim', 'Buyer', 'Justice:Fine', 'Target', 'Justice:Charge-Indict', 'r_Entity', 'r_Attacker', 'Conflict:Attack', 'Business:End-Org', 'Instrument', 'r_Buyer', 'Business:Start-Org', 'Defendant', 'r_Beneficiary', 'Justice:Sue', 'Recipient', 'PER', 'Artifact', 'Place', 'Justice:Execute', 'Plaintiff', 'Personnel:Nominate', 'Contact:Meet', 'Life:Divorce', 'Destination', 'Transaction:Transfer-Money', 'Life:Be-Born', 'r_Seller', 'r_Origin', 'Life:Injure', 'Justice:Acquit', 'Entity', 'Personnel:End-Position', 'Attacker', 'r_Agent', 'WEA', 'r_Place', 'Life:Marry', 'Personnel:Elect', 'Transaction:Transfer-Ownership', 'Contact:Phone-Write', 'Origin', 'ORG', 'r_Vehicle', 'Beneficiary', 'r_Giver', 'r_Instrument', 'r_Victim', 'r_Recipient', 'Justice:Release-Parole', 'Business:Declare-Bankruptcy', 'Person', 'r_Plaintiff', 'Adjudicator', 'Justice:Pardon', 'Org', 'Personnel:Start-Position', 'Business:Merge-Org', 'Justice:Appeal', 'Agent', 'r_Artifact', 'Giver', 'Vehicle', 'Conflict:Demonstrate', 'Prosecutor', 'r_Person', 'VEH', 'GPE', 'r_Prosecutor', 'Justice:Arrest-Jail', 'Movement:Transport', 'r_Target', 'r_Org', 'r_Adjudicator', 'Justice:Trial-Hearing', 'Life:Die', 'Justice:Convict', 'FAC', 'LOC', 'Seller', 'Justice:Sentence', 'Justice:Extradite', 'r_Destination']
        elif 'ACE_add_relation' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['ORG-AFF', 'WEA', 'r_PHYS', 'r_PART-WHOLE', 'r_ART', 'ORG', 'PHYS', 'VEH', 'ART', 'GPE', 'GEN-AFF', 'r_ORG-AFF', 'PER-SOC', 'PER', 'PART-WHOLE', 'r_GEN-AFF', 'LOC', 'r_PER-SOC', 'FAC']
        elif 'ACE_add_entity' ==config.valid_pattern_path.split('/')[-2]:
            self.types = ['ORG', 'LOC', 'WEA', 'GPE', 'PER', 'FAC', 'VEH']
        elif 'ACE_relation' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['r_PART-WHOLE', 'GPE', 'WEA', 'r_GEN-AFF', 'r_ORG-AFF', 'FAC', 'ART', 'r_PHYS', 'r_PER-SOC', 'PER-SOC', 'PART-WHOLE', 'r_ART', 'VEH', 'ORG', 'ORG-AFF', 'PHYS', 'PER', 'LOC', 'GEN-AFF']
        elif 'ACE_event' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['r_Giver', 'r_Artifact', 'Seller', 'Target', 'r_Beneficiary', 'Business.End-Org', 'Justice.Release-Parole', 'Personnel.Nominate', 'Justice.Sentence', 'Destination', 'Life.Marry', 'Victim', 'Life.Be-Born', 'Agent', 'Attacker', 'r_Buyer', 'r_Vehicle', 'Justice.Arrest-Jail', 'r_Place', 'Justice.Execute', 'r_Org', 'Org', 'r_Origin', 'Plaintiff', 'Origin', 'Transaction.Transfer-Money', 'Justice.Pardon', 'Beneficiary', 'r_Seller', 'r_Entity', 'r_Instrument', 'Life.Injure', 'Recipient', 'Business.Start-Org', 'FAC', 'Life.Divorce', 'Movement.Transport', 'r_Defendant', 'r_Plaintiff', 'Justice.Sue', 'Justice.Appeal', 'Justice.Fine', 'Conflict.Demonstrate', 'VEH', 'Giver', 'Instrument', 'Justice.Extradite', 'LOC', 'Entity', 'Personnel.Elect', 'r_Agent', 'Contact.Meet', 'Business.Declare-Bankruptcy', 'r_Victim', 'Personnel.End-Position', 'Prosecutor', 'r_Adjudicator', 'Justice.Charge-Indict', 'Adjudicator', 'r_Person', 'Transaction.Transfer-Ownership', 'Buyer', 'Place', 'GPE', 'r_Target', 'Personnel.Start-Position', 'Artifact', 'Justice.Trial-Hearing', 'r_Destination', 'Contact.Phone-Write', 'r_Attacker', 'Business.Merge-Org', 'Person', 'ORG', 'r_Recipient', 'Justice.Convict', 'WEA', 'Justice.Acquit', 'PER', 'Conflict.Attack', 'r_Prosecutor', 'Life.Die', 'Defendant', 'Vehicle']
        elif 'ACE_entity' == config.valid_pattern_path.split('/')[-2]:
            self.types = ['FAC', 'GPE', 'VEH', 'ORG', 'WEA', 'PER', 'LOC']
        elif 'conll04' == config.valid_pattern_path.split('/')[-3]:
            self.types = ['people', 'location', 'organization', 'other', 'located-in', 'organization-in', 'live-in', 'work-for', 'kill', 'r_organization-in', 'r_located-in', 'r_work-for', 'r_kill', 'r_live-in']
        elif 'nyt' == config.valid_pattern_path.split('/')[-3]:
            self.types = ['location', 'person', 'organization', 'place-of-birth', 'country', 'major-shareholder-of', 'capital', 'ethnicity', 'teams', 'industry', 'people', 'major-shareholders', 'founders', 'profession', 'advisors', 'religion', 'contains', 'children', 'neighborhood-of', 'place-founded', 'nationality', 'place-of-death', 'company', 'location', 'geographic-distribution', 'place-lived', 'administrative-divisions', 'r_contains', 'r_country', 'r_children', 'r_administrative-divisions', 'r_capital', 'r_company', 'r_place-of-death', 'r_place-of-birth', 'r_nationality', 'r_founders', 'r_neighborhood-of', 'r_place-lived', 'r_advisors', 'r_location', 'r_place-founded', 'r_major-shareholders', 'r_major-shareholder-of', 'r_teams', 'r_religion', 'r_geographic-distribution', 'r_people', 'r_ethnicity']
        elif 'scierc' == config.valid_pattern_path.split('/')[-3]:
            self.types = ['method', 'task', 'other-scientific-term', 'metric', 'material', 'generic', 'evaluate-for', 'compare', 'used-for', 'feature-of', 'conjunction', 'part-of', 'hyponym-of', 'r_used-for', 'r_feature-of', 'r_evaluate-for', 'r_conjunction', 'r_hyponym-of', 'r_part-of', 'r_compare']
        elif 'casie' == config.valid_pattern_path.split('/')[-3]:
            self.types = ['geopolitical-entity', 'time', 'file', 'website', 'data', 'common-vulnerabilities-and-exposures', 'money', 'patch', 'malware', 'person', 'purpose', 'number', 'vulnerability', 'version', 'capabilities', 'payment-method', 'system', 'software', 'personally-identifiable-information', 'organization', 'device', 'victim', 'patch-number', 'tool', 'vulnerable-system', 'vulnerable-system-owner', 'releaser', 'discoverer', 'time', 'capabilities', 'purpose', 'supported-platform', 'issues-addressed', 'common-vulnerabilities-and-exposures', 'attacker', 'trusted-entity', 'vulnerability', 'vulnerable-system-version', 'number-of-victim', 'payment-method', 'compromised-data', 'price', 'place', 'attack-pattern', 'number-of-data', 'patch', 'damage-amount', 'r_tool', 'r_trusted-entity', 'r_attack-pattern', 'r_victim', 'r_attacker', 'r_time', 'r_place', 'r_vulnerable-system-owner', 'r_discoverer', 'r_releaser', 'r_vulnerability', 'r_vulnerable-system', 'r_vulnerable-system-version', 'r_patch', 'r_compromised-data', 'r_number-of-victim', 'r_purpose', 'r_common-vulnerabilities-and-exposures', 'r_number-of-data', 'r_price', 'r_payment-method', 'r_capabilities', 'r_patch-number', 'r_damage-amount', 'r_issues-addressed', 'r_supported-platform', 'phishing', 'databreach', 'ransom', 'discover-vulnerability', 'patch-vulnerability']
        elif 'cadec' ==config.valid_pattern_path.split('/')[-3]:
            self.types = ['NA', 'adverse-drug-reaction', 'r_adverse-drug-reaction']
        elif 'absa' == config.valid_pattern_path.split('/')[-4]:
            self.types = ['opinion', 'aspect', 'negative', 'neutral', 'positive', 'r_positive', 'r_negative', 'r_neutral']
        elif 'ace2004' == config.valid_pattern_path.split('/')[-3]:
            self.types = ['GPE', 'ORG', 'PER', 'FAC', 'VEH', 'LOC', 'WEA']
        elif '14lap' == config.valid_pattern_path.split('/')[-3]:
            self.types = ["opinion", "aspect", "negative", "neutral", "positive", "r_positive", "r_negative", "r_neutral"]
        elif '14res' == config.valid_pattern_path.split('/')[-3]:
            self.types = ["opinion", "aspect", "negative", "neutral", "positive", "r_positive", "r_negative", "r_neutral"]
        elif '15res' == config.valid_pattern_path.split('/')[-3]:
            self.types = ["opinion", "aspect", "negative", "neutral", "positive", "r_positive", "r_negative", "r_neutral"]
        elif '16res' == config.valid_pattern_path.split('/')[-3]:
            self.types = ["opinion", "aspect", "negative", "neutral", "positive", "r_positive", "r_negative", "r_neutral"]
        elif 'all' == config.valid_pattern_path.split('/')[-2]:
            self.types =['method', 'geopolitical-entity', 'website', 'aspect', 'MISC', 'patch', 'other-scientific-term', 'G#cell_line', 'generic', 'device', 'other', 'file', 'task', 'money', 'data', 'G#cell_type', 'metric', 'VEH', 'time', 'GPE', 'PER', 'G#protein', 'people', 'location', 'capabilities', 'purpose', 'None', 'opinion', 'personally-identifiable-information', 'malware', 'FAC', 'system', 'number', 'G#RNA', 'WEA', 'LOC', 'ORG', 'G#DNA', 'version', 'NA', 'common-vulnerabilities-and-exposures', 'vulnerability', 'material', 'payment-method', 'software', 'person', 'organization', 'r_ORG-AFF', 'major-shareholder-of', 'r_positive', 'country', 'religion', 'positive', 'r_contains', 'r_people', 'r_place-lived', 'ethnicity', 'r_location', 'r_place-founded', 'place-of-death', 'r_place-of-birth', 'founders', 'r_part-of', 'r_PART-WHOLE', 'r_capital', 'r_neutral', 'PER-SOC', 'r_major-shareholder-of', 'negative', 'r_feature-of', 'teams', 'compare', 'part-of', 'people', 'r_founders', 'r_advisors', 'location', 'r_hyponym-of', 'r_religion', 'r_negative', 'ORG-AFF', 'used-for', 'place-founded', 'r_place-of-death', 'feature-of', 'r_neighborhood-of', 'r_evaluate-for', 'children', 'place-of-birth', 'organization-in', 'capital', 'PHYS', 'work-for', 'r_company', 'hyponym-of', 'advisors', 'r_administrative-divisions', 'evaluate-for', 'r_adverse-drug-reaction', 'administrative-divisions', 'r_PER-SOC', 'r_compare', 'r_PHYS', 'r_live-in', 'r_major-shareholders', 'profession', 'place-lived', 'r_nationality', 'contains', 'neighborhood-of', 'r_country', 'r_work-for', 'major-shareholders', 'geographic-distribution', 'GEN-AFF', 'located-in', 'r_organization-in', 'conjunction', 'company', 'r_geographic-distribution', 'adverse-drug-reaction', 'ART', 'industry', 'nationality', 'r_used-for', 'live-in', 'PART-WHOLE', 'r_conjunction', 'r_teams', 'neutral', 'r_kill', 'r_ART', 'kill', 'r_GEN-AFF', 'r_located-in', 'r_children', 'r_ethnicity', 'Life.Marry', 'Business:End-Org', 'Justice:Pardon', 'Transaction:Transfer-Ownership', 'Business:Start-Org', 'Justice:Arrest-Jail', 'Movement:Transport', 'Business.End-Org', 'Justice:Fine', 'Justice:Extradite', 'databreach', 'Conflict:Attack', 'Justice:Appeal', 'Justice:Sue', 'Life.Divorce', 'Justice.Acquit', 'Justice:Trial-Hearing', 'Life.Be-Born', 'Justice.Trial-Hearing', 'Conflict.Attack', 'Life:Marry', 'Contact:Meet', 'Personnel:Start-Position', 'Business:Declare-Bankruptcy', 'Justice.Fine', 'Personnel:End-Position', 'Personnel.End-Position', 'Transaction.Transfer-Money', 'Justice.Extradite', 'Business:Merge-Org', 'phishing', 'Justice:Charge-Indict', 'Personnel:Elect', 'Personnel.Nominate', 'Justice.Appeal', 'Life:Injure', 'patch-vulnerability', 'Contact.Meet', 'Justice.Convict', 'Conflict.Demonstrate', 'Contact.Phone-Write', 'Justice.Pardon', 'Justice.Charge-Indict', 'Personnel.Start-Position', 'Life:Die', 'Transaction:Transfer-Money', 'Justice:Release-Parole', 'Justice.Sentence', 'Justice.Sue', 'Movement.Transport', 'Personnel:Nominate', 'Life:Be-Born', 'Justice:Convict', 'Justice:Acquit', 'Transaction.Transfer-Ownership', 'discover-vulnerability', 'Business.Start-Org', 'Justice.Arrest-Jail', 'ransom', 'Business.Declare-Bankruptcy', 'Conflict:Demonstrate', 'Business.Merge-Org', 'Justice:Execute', 'Life:Divorce', 'Justice.Release-Parole', 'Personnel.Elect', 'Life.Die', 'Justice:Sentence', 'Justice.Execute', 'Life.Injure', 'Contact:Phone-Write', 'r_number-of-data', 'r_patch-number', 'Recipient', 'r_Target', 'Beneficiary', 'Agent', 'r_number-of-victim', 'Entity', 'r_Beneficiary', 'r_Adjudicator', 'supported-platform', 'trusted-entity', 'Attacker', 'r_time', 'r_attacker', 'Person', 'r_Entity', 'common-vulnerabilities-and-exposures', 'r_Giver', 'victim', 'r_Org', 'payment-method', 'vulnerable-system-version', 'r_compromised-data', 'r_vulnerable-system-owner', 'damage-amount', 'Vehicle', 'r_trusted-entity', 'r_Buyer', 'Artifact', 'Plaintiff', 'Origin', 'compromised-data', 'number-of-victim', 'discoverer', 'releaser', 'r_tool', 'Prosecutor', 'Place', 'r_Instrument', 'r_purpose', 'vulnerable-system-owner', 'r_Destination', 'r_Recipient', 'r_payment-method', 'r_Attacker', 'r_attack-pattern', 'tool', 'r_victim', 'issues-addressed', 'r_Defendant', 'capabilities', 'purpose', 'Adjudicator', 'Buyer', 'place', 'r_place', 'r_price', 'number-of-data', 'patch-number', 'attacker', 'r_discoverer', 'r_releaser', 'r_vulnerability', 'r_supported-platform', 'Org', 'r_Plaintiff', 'r_Seller', 'r_Origin', 'Instrument', 'Seller', 'r_Prosecutor', 'r_Place', 'patch', 'Target', 'r_patch', 'r_Victim', 'Defendant', 'r_Artifact', 'r_vulnerable-system', 'r_Vehicle', 'time', 'r_damage-amount', 'vulnerable-system', 'attack-pattern', 'r_Person', 'Giver', 'r_common-vulnerabilities-and-exposures', 'r_vulnerable-system-version', 'r_Agent', 'r_issues-addressed', 'Destination', 'r_capabilities', 'Victim', 'vulnerability', 'price']
        
        self.types_tokens = [["[CLS]"] + self.tokenizer.tokenize(type_token) + ["[SEP]"] for type_token in self.types]
        self.types_inputs = [torch.tensor([self.tokenizer.convert_tokens_to_ids(type_token)])  for type_token in self.types_tokens]

        if (config.parser_type == 'bilinear'):
            self.parser = Bilinear(config)
        elif (config.parser_type == 'biaffine'):
            self.parser = BiAffine(config)
        elif (config.parser_type == 'deepbiaffine'):
            self.parser = DeepBiaffine(config)
        else:
            raise NotImplementedError('illegal parser type %s not implemented!' % config.parser_type)
        self.tree_crf = TreeCRFLayer(config)

        self.init_weights()

    def forward_1(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks, eval_masks):
        """

        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
            partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
                label_size = observed_label_size + latent_label_size

            input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              eval_masks=eval_masks
                              
        Returns:
            outputs: list 
        """
        inspect = {}
        label_size = self.label_size

        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape

        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]


        # prepare for tree CRF

        log_potentials = self.parser(gather_output)

        # lengths = gather_masks.sum(1)
        # max_len = log_potentials.size(1)
        # # TODO: use vanilla span classification 
        # if (self.use_crf is False):
        #     # [batch * max_len * max_len]
        #     targets = partial_mask_to_targets(partial_masks).view(-1)
        #     # [batch * max_len * max_len, label_size]
        #     prob = log_potentials.reshape(-1, label_size)
        #     loss = F.cross_entropy(prob, targets, reduction='none')

        #     # [batch, max_len, max_len]
        #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
        #     # [batch, max_len, max_len] -> [batch * max_len * max_len]
        #     mask = torch.triu(mask.float()).view(-1)
        #     loss = (loss * mask).sum() / mask.sum()

        # else:
        #     # log_prob_sum_partial.size = [batch]
        #     # TODO: check partial_masks boundary, Done

        #     log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
        #         self.tree_crf(log_potentials, partial_masks, lengths, eval_masks)

        #     if (self.structure_smoothing):
        #         loss = -log_prob_smooth.mean()
        #     else:
        #         loss = -log_prob_sum_partial.mean()
        #     loss -= self.lambda_ent * entropy.mean()

        # outputs = [loss, inspect]
        # import  torch.nn as nn
        # target = torch.zeros(eval_masks.shape[:-1])
        # for i in range(eval_masks.shape[0]):
        #     for j in range(eval_masks[i].shape[0]):
        #         for k in range(eval_masks[i][j].shape[0]):
        #             for t in range(eval_masks[i][j][k].shape[0]):
        #                 if eval_masks[i][j][k][t].item() == 1 and t != eval_masks.shape[-1]:
        #                     target[i][j][k] = torch.tensor(t)
        # print(eval_masks.shape[0])
        # print(eval_masks[0].shape[0])
        # print(eval_masks[0][0].shape[0])
        # print(eval_masks[0][0][0].shape[0])
        # print(target.shape)
        # print(target)
        # stop
        # torch.set_printoptions(profile="full")
        # print(eval_masks)
        # print(eval_masks.argmax(-1)[0])
        # stop
        target = eval_masks.argmax(-1)
        # target[torch.where(target == 0)] = log_potentials.shape[-1]-1
        criterion = nn.CrossEntropyLoss(ignore_index = log_potentials.shape[-1]-1)
        loss = criterion(log_potentials.reshape(-1,log_potentials.shape[-1]), target.view(-1))
        outputs = [loss, inspect]
        return outputs

    def multilabel_categorical_crossentropy(self, y_pred, y_true, bit_mask=None):
        """
        https://kexue.fm/archives/7359
        https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/common/utils.py
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e6  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e6  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        if bit_mask is None:
            return neg_loss + pos_loss
        else:
            raise NotImplementedError

    def forward_0(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks, eval_masks):
        """
        原始
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
            partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
                label_size = observed_label_size + latent_label_size

            input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              eval_masks=eval_masks
                              
        Returns:
            outputs: list 
        """
        inspect = {}
        label_size = self.label_size

        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape

        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]


        # prepare for tree CRF

        log_potentials = self.parser(gather_output)

        lengths = gather_masks.sum(1)
        max_len = log_potentials.size(1)
        # TODO: use vanilla span classification 
        if (self.use_crf is False):
            # [batch * max_len * max_len]
            targets = partial_mask_to_targets(partial_masks).view(-1)
            # [batch * max_len * max_len, label_size]
            prob = log_potentials.reshape(-1, label_size)
            loss = F.cross_entropy(prob, targets, reduction='none')

            # [batch, max_len, max_len]
            mask = tmu.lengths_to_squared_mask(lengths, max_len)
            # [batch, max_len, max_len] -> [batch * max_len * max_len]
            mask = torch.triu(mask.float()).view(-1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            # log_prob_sum_partial.size = [batch]
            # TODO: check partial_masks boundary, Done

            log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
                self.tree_crf(log_potentials, partial_masks, lengths, eval_masks)

            if (self.structure_smoothing):
                loss = -log_prob_smooth.mean()
            else:
                loss = -log_prob_sum_partial.mean()
            loss -= self.lambda_ent * entropy.mean()

        outputs = [loss, inspect]
        return outputs

    def forward_1(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks, eval_masks):
        """
        添加掩码语言损失
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
            partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
                label_size = observed_label_size + latent_label_size

            input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              eval_masks=eval_masks
                              
        Returns:
            outputs: list 
        """
        inspect = {}
        label_size = self.label_size

        outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs_bert[0]

        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape

        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]


        # prepare for tree CRF

        log_potentials = self.parser(gather_output)

        lengths = gather_masks.sum(1)
        max_len = log_potentials.size(1)
        # TODO: use vanilla span classification 
        # if (self.use_crf is False):
        #     # [batch * max_len * max_len]
        #     targets = partial_mask_to_targets(partial_masks).view(-1)
        #     # [batch * max_len * max_len, label_size]
        #     prob = log_potentials.reshape(-1, label_size)
        #     loss = F.cross_entropy(prob, targets, reduction='none')

        #     # [batch, max_len, max_len]
        #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
        #     # [batch, max_len, max_len] -> [batch * max_len * max_len]
        #     mask = torch.triu(mask.float()).view(-1)
        #     loss = (loss * mask).sum() / mask.sum()

        # else:
        #     # log_prob_sum_partial.size = [batch]
        #     # TODO: check partial_masks boundary, Done

        #     log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
        #         self.tree_crf(log_potentials, partial_masks, lengths, eval_masks)

        #     if (self.structure_smoothing):
        #         loss = -log_prob_smooth.mean()
        #     else:
        #         loss = -log_prob_sum_partial.mean()
        #     loss -= self.lambda_ent * entropy.mean()

        # tensor_a = eval_masks.clone()
        # indices = (tensor_a == 1).nonzero(as_tuple=True)
        # last_index = indices[3].max().item()
        # last_indices_mask = indices[3] == last_index
        # tensor_a[indices[0][last_indices_mask], indices[1][last_indices_mask], indices[2][last_indices_mask], last_index] = 0
        # torch.set_printoptions(profile="full")
        # print(log_potentials)
        # stop
        # padding mask
        batch_size, max_len, _, label_size = log_potentials.shape
        pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        log_potentials = log_potentials.permute(0, 3, 1, 2)
        log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

        # 排除下三角
        mask = torch.tril(torch.ones_like(log_potentials), -1)
        log_potentials = log_potentials - mask * 1e6
        log_potentials = log_potentials / 64 ** 0.5
        
        # [24, 121, 64, 64]
        logits = log_potentials[:,:-1,:,:]
        tensor_a = partial_masks[:,:,:,:-1].permute(0, 3, 1, 2)
        prob = logits.reshape(batch_size * (label_size - 1), -1)
        targets = tensor_a.reshape(batch_size * (label_size - 1), -1)
        loss_2 = self.multilabel_categorical_crossentropy(prob, targets).mean()

        outputs = [loss_2, inspect]
        # print(loss_2)
        return outputs

    def kl_div_loss(self, x1, x2):

        batch_dist = F.softmax(x1, dim=-1)
        temp_dist = F.log_softmax(x2, dim=-1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
        loss /= 12
        return loss

    def forward_now(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks, eval_masks):
        """
        添加掩码语言损失+掩码插值表达的损失+KL损失
        将self.bert换成self.mlm
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
            partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
                label_size = observed_label_size + latent_label_size

            input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              eval_masks=eval_masks
                              
        Returns:
            outputs: list 
        """
        inspect = {}
        label_size = self.label_size

        outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        outputs_bert_embedding = self.bert.get_input_embeddings()(input_ids)

        sequence_output_bert = outputs_bert[0]
        # outputs_bert = self.mlm(input_ids, position_ids=None, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, output_hidden_states=True)
        # sequence_output_bert = outputs_bert.hidden_states[-1]
        batch_size, seq_len = input_ids.size()
        mask_token_id = self.tokenizer.mask_token_id

        # Create masked input_ids
        masked_input_ids = input_ids.unsqueeze(1).repeat(1, seq_len, 1)
        mask = torch.eye(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)
        masked_input_ids = masked_input_ids.masked_fill(mask == 1, mask_token_id).to(input_ids.device)

        # Expand token_type_ids and attention_mask to match masked_input_ids
        token_type_ids_ = token_type_ids.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        attention_mask_ = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        masked_input_ids = masked_input_ids.view(batch_size * seq_len, -1)

        # Get BERT outputs
        # with autocast():
        outputs = self.mlm(masked_input_ids, attention_mask=attention_mask_, token_type_ids=token_type_ids_)
        mlm_logits = outputs.logits

        # Get probabilities through softmax
        mlm_probs = torch.softmax(torch.relu(mlm_logits) + 1, dim=-1)
        
        # Extract masked token probabilities
        mlm_probs = mlm_probs.view(batch_size, seq_len, seq_len, -1)
        mask_positions = mask.nonzero(as_tuple=True)
        masked_probs = mlm_probs[mask_positions].view(batch_size, seq_len, -1)

        # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        word_embeddings = self.mlm.get_input_embeddings().weight
        sequence_output_mlm = torch.matmul(masked_probs.to(input_ids.device), word_embeddings)

        loss_4 = self.kl_div_loss(outputs_bert_embedding, sequence_output_mlm)

        # print(sequence_output_mlm.shape)
        sequence_output_mlm = self.mlm.roberta(inputs_embeds=sequence_output_mlm,attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        # outputs = self.mlm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # mlm_logits = outputs.logits

        # # for name, param in self.bert.named_parameters():
        # #     if not param.requires_grad:
        # #         print(f"Parameter {name} is frozen and will not be updated during training.")
        # #     else:
        # #         print(f"Parameter {name} is trainable and will be updated during training.")

        # # Get probabilities through softmax
        # mlm_probs = torch.softmax(mlm_logits, dim=-1)

        # # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        # sequence_output_mlm = torch.matmul(mlm_probs.to(input_ids.device), word_embeddings)
        # print(sequence_output_mlm.shape)
        sequence_output = sequence_output_bert * 1 + sequence_output_mlm * 0.5

        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape

        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]


        # prepare for tree CRF

        log_potentials = self.parser(gather_output)

        lengths = gather_masks.sum(1)
        max_len = log_potentials.size(1)
        # TODO: use vanilla span classification 
        # if (self.use_crf is False):
        #     # [batch * max_len * max_len]
        #     targets = partial_mask_to_targets(partial_masks).view(-1)
        #     # [batch * max_len * max_len, label_size]
        #     prob = log_potentials.reshape(-1, label_size)
        #     loss = F.cross_entropy(prob, targets, reduction='none')

        #     # [batch, max_len, max_len]
        #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
        #     # [batch, max_len, max_len] -> [batch * max_len * max_len]
        #     mask = torch.triu(mask.float()).view(-1)
        #     loss = (loss * mask).sum() / mask.sum()

        # else:
        #     # log_prob_sum_partial.size = [batch]
        #     # TODO: check partial_masks boundary, Done

        #     log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
        #         self.tree_crf(log_potentials, partial_masks, lengths, eval_masks)

        #     if (self.structure_smoothing):
        #         loss = -log_prob_smooth.mean()
        #     else:
        #         loss = -log_prob_sum_partial.mean()
        #     loss -= self.lambda_ent * entropy.mean()

        # tensor_a = eval_masks.clone()
        # indices = (tensor_a == 1).nonzero(as_tuple=True)
        # last_index = indices[3].max().item()
        # last_indices_mask = indices[3] == last_index
        # tensor_a[indices[0][last_indices_mask], indices[1][last_indices_mask], indices[2][last_indices_mask], last_index] = 0
        # torch.set_printoptions(profile="full")
        # print(log_potentials)
        # stop
        # padding mask
        batch_size, max_len, _, label_size = log_potentials.shape
        pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        log_potentials = log_potentials.permute(0, 3, 1, 2)
        log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

        # 排除下三角
        mask = torch.tril(torch.ones_like(log_potentials), -1)
        log_potentials = log_potentials - mask * 1e6
        log_potentials = log_potentials / 64 ** 0.5
        
        # [24, 121, 64, 64]
        logits = log_potentials[:,:-1,:,:]
        tensor_a = eval_masks[:,:,:,:-1].permute(0, 3, 1, 2)
        prob = logits.reshape(batch_size * (label_size - 1), -1)
        targets = tensor_a.reshape(batch_size * (label_size - 1), -1)
        loss_2 = self.multilabel_categorical_crossentropy(prob, targets).mean()

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        # print(masked_probs.shape, input_ids.shape)
        loss_3 = criterion(masked_probs.view(-1, masked_probs.shape[-1]), input_ids.view(-1))

        # torch.set_printoptions(profile="full")
        # print(logits)
        # print(eval_masks)
        # print(tensor_a)
        # print(loss_2)
        # stop

        # criterion = ATLoss()
        # loss_2 = criterion(prob[:,40:], targets[:,40:])
        # loss_2 = (loss_2.view(input_ids.size(0), -1)).sum(dim=-1).mean()

        # # [batch * max_len * max_len]
        # targets = partial_mask_to_targets(eval_masks).view(-1)
        # # [batch * max_len * max_len, label_size]
        # prob = log_potentials.reshape(-1, label_size)
        # loss_2 = F.cross_entropy(prob, targets, reduction='none')
        # # [batch, max_len, max_len]
        # mask = tmu.lengths_to_squared_mask(lengths, max_len)
        # # [batch, max_len, max_len] -> [batch * max_len * max_len]
        # mask = torch.triu(mask.float()).view(-1)
        # loss_2 = (loss_2 * mask).sum() / mask.sum()
        
        outputs = [loss_2 + loss_3 * 0.5 - 0.5 * loss_4, inspect]
        # print(loss_2)
        return outputs

    def forward_3(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks, eval_masks):
        """
        掩码语言损失+插值表达损失+树的损失
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
            partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
                label_size = observed_label_size + latent_label_size

            input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              eval_masks=eval_masks
                              
        Returns:
            outputs: list 
        """
        inspect = {}
        label_size = self.label_size

        outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output_bert = outputs_bert[0]
        batch_size, seq_len = input_ids.size()
        mask_token_id = self.tokenizer.mask_token_id

        # Create masked input_ids
        masked_input_ids = input_ids.unsqueeze(1).repeat(1, seq_len, 1)
        mask = torch.eye(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)
        masked_input_ids = masked_input_ids.masked_fill(mask == 1, mask_token_id).to(input_ids.device)

        # Expand token_type_ids and attention_mask to match masked_input_ids
        token_type_ids = token_type_ids.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        masked_input_ids = masked_input_ids.view(batch_size * seq_len, -1)

        # Get BERT outputs
        # with autocast():
        outputs = self.mlm(masked_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mlm_logits = outputs.logits

        # Get probabilities through softmax
        mlm_probs = torch.softmax(torch.relu(mlm_logits)+1, dim=-1)
        
        # Extract masked token probabilities
        mlm_probs = mlm_probs.view(batch_size, seq_len, seq_len, -1)
        mask_positions = mask.nonzero(as_tuple=True)
        masked_probs = mlm_probs[mask_positions].view(batch_size, seq_len, -1)

        # Get word embeddings and form new sentence representations
        word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        sequence_output_mlm = torch.matmul(masked_probs.to(input_ids.device), word_embeddings)

        # outputs = self.mlm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # mlm_logits = outputs.logits

        # # for name, param in self.bert.named_parameters():
        # #     if not param.requires_grad:
        # #         print(f"Parameter {name} is frozen and will not be updated during training.")
        # #     else:
        # #         print(f"Parameter {name} is trainable and will be updated during training.")

        # # Get probabilities through softmax
        # mlm_probs = torch.softmax(mlm_logits, dim=-1)

        # # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        # sequence_output_mlm = torch.matmul(mlm_probs.to(input_ids.device), word_embeddings)

        sequence_output = sequence_output_bert * 1 + sequence_output_mlm * 0.5

        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape

        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]


        # prepare for tree CRF

        log_potentials = self.parser(gather_output)

        lengths = gather_masks.sum(1)
        max_len = log_potentials.size(1)
        # TODO: use vanilla span classification 
        if (self.use_crf is False):
            # [batch * max_len * max_len]
            targets = partial_mask_to_targets(partial_masks).view(-1)
            # [batch * max_len * max_len, label_size]
            prob = log_potentials.reshape(-1, label_size)
            loss = F.cross_entropy(prob, targets, reduction='none')

            # [batch, max_len, max_len]
            mask = tmu.lengths_to_squared_mask(lengths, max_len)
            # [batch, max_len, max_len] -> [batch * max_len * max_len]
            mask = torch.triu(mask.float()).view(-1)
            loss = (loss * mask).sum() / mask.sum()

        else:
            # log_prob_sum_partial.size = [batch]
            # TODO: check partial_masks boundary, Done

            log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
                self.tree_crf(log_potentials, partial_masks, lengths, eval_masks)

            if (self.structure_smoothing):
                loss_4 = -log_prob_smooth.mean()
            else:
                loss_4 = -log_prob_sum_partial.mean()
            loss_4 -= self.lambda_ent * entropy.mean()

        # tensor_a = eval_masks.clone()
        # indices = (tensor_a == 1).nonzero(as_tuple=True)
        # last_index = indices[3].max().item()
        # last_indices_mask = indices[3] == last_index
        # tensor_a[indices[0][last_indices_mask], indices[1][last_indices_mask], indices[2][last_indices_mask], last_index] = 0
        # torch.set_printoptions(profile="full")
        # print(log_potentials)
        # stop
        # padding mask
        batch_size, max_len, _, label_size = log_potentials.shape
        pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        log_potentials = log_potentials.permute(0, 3, 1, 2)
        log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

        # 排除下三角
        mask = torch.tril(torch.ones_like(log_potentials), -1)
        log_potentials = log_potentials - mask * 1e6
        log_potentials = log_potentials / 64 ** 0.5
        
        # [24, 121, 64, 64]
        logits = log_potentials[:,:-1,:,:]
        tensor_a = partial_masks[:,:,:,:-1].permute(0, 3, 1, 2)
        prob = logits.reshape(batch_size * (label_size - 1), -1)
        targets = tensor_a.reshape(batch_size * (label_size - 1), -1)
        loss_2 = self.multilabel_categorical_crossentropy(prob, targets).mean()

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        # print(masked_probs.shape, input_ids.shape)
        loss_3 = criterion(masked_probs.view(-1, masked_probs.shape[-1]), input_ids.view(-1))

        # torch.set_printoptions(profile="full")
        # print(logits)
        # print(eval_masks)
        # print(tensor_a)
        # print(loss_2)
        # stop

        # criterion = ATLoss()
        # loss_2 = criterion(prob[:,40:], targets[:,40:])
        # loss_2 = (loss_2.view(input_ids.size(0), -1)).sum(dim=-1).mean()

        # # [batch * max_len * max_len]
        # targets = partial_mask_to_targets(eval_masks).view(-1)
        # # [batch * max_len * max_len, label_size]
        # prob = log_potentials.reshape(-1, label_size)
        # loss_2 = F.cross_entropy(prob, targets, reduction='none')
        # # [batch, max_len, max_len]
        # mask = tmu.lengths_to_squared_mask(lengths, max_len)
        # # [batch, max_len, max_len] -> [batch * max_len * max_len]
        # mask = torch.triu(mask.float()).view(-1)
        # loss_2 = (loss_2 * mask).sum() / mask.sum()
        
        outputs = [loss_2 + loss_3 * 0.5 + loss_4 * 0.5, inspect]
        # print(loss_2)
        return outputs

    def forward(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks, eval_masks):
        """
        添加掩码语言损失+掩码插值表达的损失+KL损失+角色子空间
        将self.bert换成self.mlm
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
            partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
                label_size = observed_label_size + latent_label_size

            input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              eval_masks=eval_masks
                              
        Returns:
            outputs: list 
        """
        inspect = {}
        label_size = self.label_size

        outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        outputs_bert_embedding = self.bert.get_input_embeddings()(input_ids)

        sequence_output_bert = outputs_bert[0]
        # outputs_bert = self.mlm(input_ids, position_ids=None, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, output_hidden_states=True)
        # sequence_output_bert = outputs_bert.hidden_states[-1]
        batch_size, seq_len = input_ids.size()
        mask_token_id = self.tokenizer.mask_token_id

        # Create masked input_ids
        masked_input_ids = input_ids.unsqueeze(1).repeat(1, seq_len, 1)
        mask = torch.eye(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)
        masked_input_ids = masked_input_ids.masked_fill(mask == 1, mask_token_id).to(input_ids.device)

        # Expand token_type_ids and attention_mask to match masked_input_ids
        token_type_ids_ = token_type_ids.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        attention_mask_ = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        masked_input_ids = masked_input_ids.view(batch_size * seq_len, -1)

        # Get BERT outputs
        # with autocast():
        outputs = self.mlm(masked_input_ids, attention_mask=attention_mask_, token_type_ids=token_type_ids_)
        mlm_logits = outputs.logits

        # Get probabilities through softmax
        mlm_probs = torch.softmax(torch.relu(mlm_logits) + 1, dim=-1)
        
        # Extract masked token probabilities
        mlm_probs = mlm_probs.view(batch_size, seq_len, seq_len, -1)
        mask_positions = mask.nonzero(as_tuple=True)
        masked_probs = mlm_probs[mask_positions].view(batch_size, seq_len, -1)

        # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        word_embeddings = self.mlm.get_input_embeddings().weight
        sequence_output_mlm = torch.matmul(masked_probs.to(input_ids.device), word_embeddings)

        loss_4 = self.kl_div_loss(outputs_bert_embedding, sequence_output_mlm)

        # print(sequence_output_mlm.shape)
        sequence_output_mlm = self.mlm.bert(inputs_embeds=sequence_output_mlm,attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        # outputs = self.mlm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # mlm_logits = outputs.logits

        # # for name, param in self.bert.named_parameters():
        # #     if not param.requires_grad:
        # #         print(f"Parameter {name} is frozen and will not be updated during training.")
        # #     else:
        # #         print(f"Parameter {name} is trainable and will be updated during training.")

        # # Get probabilities through softmax
        # mlm_probs = torch.softmax(mlm_logits, dim=-1)

        # # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        # sequence_output_mlm = torch.matmul(mlm_probs.to(input_ids.device), word_embeddings)
        # print(sequence_output_mlm.shape)
        batch_size, sequence_length, hidden_size = sequence_output_bert.shape
        # matrix = torch.ones(batch_size, sequence_length, hidden_size)
        
        types_embeddings = [self.bert(type_input.to(sequence_output_bert.device))[0] for type_input in self.types_inputs]
        selected_embeddings = [embedding[:,1,:] for embedding in types_embeddings]
        types_embeddings = torch.cat(selected_embeddings, dim=0)
        matrix = torch.matmul(torch.softmax(torch.matmul(sequence_output_bert, types_embeddings.view(hidden_size, -1)), -1), types_embeddings)
        sequence_output_bert_new = self.fusion_module(sequence_output_bert, matrix.to(sequence_output_bert.device))
        
        sequence_output = sequence_output_bert * 1 + sequence_output_mlm * 0.5 + sequence_output_bert_new * 0.5

        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape

        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]


        # prepare for tree CRF

        log_potentials = self.parser(gather_output)

        lengths = gather_masks.sum(1)
        max_len = log_potentials.size(1)
        # TODO: use vanilla span classification 
        # if (self.use_crf is False):
        #     # [batch * max_len * max_len]
        #     targets = partial_mask_to_targets(partial_masks).view(-1)
        #     # [batch * max_len * max_len, label_size]
        #     prob = log_potentials.reshape(-1, label_size)
        #     loss = F.cross_entropy(prob, targets, reduction='none')

        #     # [batch, max_len, max_len]
        #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
        #     # [batch, max_len, max_len] -> [batch * max_len * max_len]
        #     mask = torch.triu(mask.float()).view(-1)
        #     loss = (loss * mask).sum() / mask.sum()

        # else:
        #     # log_prob_sum_partial.size = [batch]
        #     # TODO: check partial_masks boundary, Done

        #     log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
        #         self.tree_crf(log_potentials, partial_masks, lengths, eval_masks)

        #     if (self.structure_smoothing):
        #         loss = -log_prob_smooth.mean()
        #     else:
        #         loss = -log_prob_sum_partial.mean()
        #     loss -= self.lambda_ent * entropy.mean()

        # tensor_a = eval_masks.clone()
        # indices = (tensor_a == 1).nonzero(as_tuple=True)
        # last_index = indices[3].max().item()
        # last_indices_mask = indices[3] == last_index
        # tensor_a[indices[0][last_indices_mask], indices[1][last_indices_mask], indices[2][last_indices_mask], last_index] = 0
        # torch.set_printoptions(profile="full")
        # print(log_potentials)
        # stop
        # padding mask
        batch_size, max_len, _, label_size = log_potentials.shape
        pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        log_potentials = log_potentials.permute(0, 3, 1, 2)
        log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

        # 排除下三角
        mask = torch.tril(torch.ones_like(log_potentials), -1)
        log_potentials = log_potentials - mask * 1e6
        log_potentials = log_potentials / 64 ** 0.5
        
        # [24, 121, 64, 64]
        logits = log_potentials[:,:-1,:,:]
        tensor_a = eval_masks[:,:,:,:-1].permute(0, 3, 1, 2)
        prob = logits.reshape(batch_size * (label_size - 1), -1)
        targets = tensor_a.reshape(batch_size * (label_size - 1), -1)
        loss_2 = self.multilabel_categorical_crossentropy(prob, targets).mean()

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        # print(masked_probs.shape, input_ids.shape)
        loss_3 = criterion(masked_probs.view(-1, masked_probs.shape[-1]), input_ids.view(-1))

        # torch.set_printoptions(profile="full")
        # print(logits)
        # print(eval_masks)
        # print(tensor_a)
        # print(loss_2)
        # stop

        # criterion = ATLoss()
        # loss_2 = criterion(prob[:,40:], targets[:,40:])
        # loss_2 = (loss_2.view(input_ids.size(0), -1)).sum(dim=-1).mean()

        # # [batch * max_len * max_len]
        # targets = partial_mask_to_targets(eval_masks).view(-1)
        # # [batch * max_len * max_len, label_size]
        # prob = log_potentials.reshape(-1, label_size)
        # loss_2 = F.cross_entropy(prob, targets, reduction='none')
        # # [batch, max_len, max_len]
        # mask = tmu.lengths_to_squared_mask(lengths, max_len)
        # # [batch, max_len, max_len] -> [batch * max_len * max_len]
        # mask = torch.triu(mask.float()).view(-1)
        # loss_2 = (loss_2 * mask).sum() / mask.sum()
        
        outputs = [loss_2 + loss_3 * 0.5 - 0.5 * loss_4, inspect]
        # print(loss_2)
        return outputs

    # def forward_4(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks, eval_masks):
    #     """
    #     掩码语言损失+插值表达损失+基于类别子空间
    #     Args:
    #         input_ids: torch.LongTensor, size=[batch, max_len]
    #         token_type_ids:
    #         attention_mask:
    #         gather_ids:
    #         gather_masks: torch.FloatTensor, size=[batch, max_len]
    #         partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
    #             label_size = observed_label_size + latent_label_size

    #         input_ids=input_ids,
    #                           input_mask=input_mask,
    #                           segment_ids=segment_ids,
    #                           partial_masks=partial_masks,
    #                           gather_ids=gather_ids,
    #                           gather_masks=gather_masks,
    #                           eval_masks=eval_masks
                              
    #     Returns:
    #         outputs: list 
    #     """
    #     inspect = {}
    #     label_size = self.label_size

    #     outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
    #                         attention_mask=attention_mask)

    #     sequence_output_bert = outputs_bert[0]
    #     batch_size, seq_len = input_ids.size()
    #     mask_token_id = self.tokenizer.mask_token_id

    #     # Create masked input_ids
    #     masked_input_ids = input_ids.unsqueeze(1).repeat(1, seq_len, 1)
    #     mask = torch.eye(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)
    #     masked_input_ids = masked_input_ids.masked_fill(mask == 1, mask_token_id).to(input_ids.device)

    #     # Expand token_type_ids and attention_mask to match masked_input_ids
    #     token_type_ids = token_type_ids.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
    #     attention_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
    #     masked_input_ids = masked_input_ids.view(batch_size * seq_len, -1)

    #     # Get BERT outputs
    #     # with autocast():
    #     outputs = self.mlm(masked_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     mlm_logits = outputs.logits

    #     # Get probabilities through softmax
    #     mlm_probs = torch.softmax(torch.relu(mlm_logits)+1, dim=-1)
        
    #     # Extract masked token probabilities
    #     mlm_probs = mlm_probs.view(batch_size, seq_len, seq_len, -1)
    #     mask_positions = mask.nonzero(as_tuple=True)
    #     masked_probs = mlm_probs[mask_positions].view(batch_size, seq_len, -1)

    #     # Get word embeddings and form new sentence representations
    #     word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
    #     sequence_output_mlm = torch.matmul(masked_probs.to(input_ids.device), word_embeddings)

    #     # outputs = self.mlm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     # mlm_logits = outputs.logits

    #     # # for name, param in self.bert.named_parameters():
    #     # #     if not param.requires_grad:
    #     # #         print(f"Parameter {name} is frozen and will not be updated during training.")
    #     # #     else:
    #     # #         print(f"Parameter {name} is trainable and will be updated during training.")

    #     # # Get probabilities through softmax
    #     # mlm_probs = torch.softmax(mlm_logits, dim=-1)

    #     # # Get word embeddings and form new sentence representations
    #     # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
    #     # sequence_output_mlm = torch.matmul(mlm_probs.to(input_ids.device), word_embeddings)
      
    #     batch_size, sequence_length, hidden_size = sequence_output_bert.shape
    #     # matrix = torch.ones(batch_size, sequence_length, hidden_size)
    #     types_tokens = self.tokenizer(self.types)
    #     types_tokens = ["[CLS]"] + types_tokens + ["[SEP]"]
    #     types_inputs = tokenizer.convert_tokens_to_ids(types_tokens)
    #     types_embeddings = self.bert(types_inputs)[1]
    #     matrix = torch.softmax(sequence_output_bert * types_embeddings.view(hidden_size, -1), -1) * types_embeddings
    #     sequence_output_bert_new = self.fusion_module(sequence_output_bert, matrix.to(sequence_output_bert.device))
        
    #     sequence_output = sequence_output_bert * 1 + sequence_output_mlm * 0.5 + sequence_output_bert_new * 0.5
    #     sequence_output = self.dropout(sequence_output)

    #     gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
    #         sequence_output.shape)
    #     gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

    #     # prepare for tree CRF

    #     log_potentials = self.parser(gather_output)

    #     lengths = gather_masks.sum(1)
    #     max_len = log_potentials.size(1)
    #     # TODO: use vanilla span classification 
    #     if (self.use_crf is False):
    #         # [batch * max_len * max_len]
    #         targets = partial_mask_to_targets(partial_masks).view(-1)
    #         # [batch * max_len * max_len, label_size]
    #         prob = log_potentials.reshape(-1, label_size)
    #         loss = F.cross_entropy(prob, targets, reduction='none')

    #         # [batch, max_len, max_len]
    #         mask = tmu.lengths_to_squared_mask(lengths, max_len)
    #         # [batch, max_len, max_len] -> [batch * max_len * max_len]
    #         mask = torch.triu(mask.float()).view(-1)
    #         loss = (loss * mask).sum() / mask.sum()

    #     else:
    #         # log_prob_sum_partial.size = [batch]
    #         # TODO: check partial_masks boundary, Done

    #         log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
    #             self.tree_crf(log_potentials, partial_masks, lengths, eval_masks)

    #         if (self.structure_smoothing):
    #             loss_4 = -log_prob_smooth.mean()
    #         else:
    #             loss_4 = -log_prob_sum_partial.mean()
    #         loss_4 -= self.lambda_ent * entropy.mean()

    #     # tensor_a = eval_masks.clone()
    #     # indices = (tensor_a == 1).nonzero(as_tuple=True)
    #     # last_index = indices[3].max().item()
    #     # last_indices_mask = indices[3] == last_index
    #     # tensor_a[indices[0][last_indices_mask], indices[1][last_indices_mask], indices[2][last_indices_mask], last_index] = 0
    #     # torch.set_printoptions(profile="full")
    #     # print(log_potentials)
    #     # stop
    #     # padding mask
    #     batch_size, max_len, _, label_size = log_potentials.shape
    #     pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
    #     # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
    #     # pad_mask = pad_mask_v&pad_mask_h
    #     log_potentials = log_potentials.permute(0, 3, 1, 2)
    #     log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

    #     # 排除下三角
    #     mask = torch.tril(torch.ones_like(log_potentials), -1)
    #     log_potentials = log_potentials - mask * 1e6
    #     log_potentials = log_potentials / 64 ** 0.5
        
    #     # [24, 121, 64, 64]
    #     logits = log_potentials[:,:-1,:,:]
    #     tensor_a = partial_masks[:,:,:,:-1].permute(0, 3, 1, 2)
    #     prob = logits.reshape(batch_size * (label_size - 1), -1)
    #     targets = tensor_a.reshape(batch_size * (label_size - 1), -1)
    #     loss_2 = self.multilabel_categorical_crossentropy(prob, targets).mean()

    #     criterion = nn.CrossEntropyLoss(ignore_index=0)
    #     # print(masked_probs.shape, input_ids.shape)
    #     loss_3 = criterion(masked_probs.view(-1, masked_probs.shape[-1]), input_ids.view(-1))

    #     # torch.set_printoptions(profile="full")
    #     # print(logits)
    #     # print(eval_masks)
    #     # print(tensor_a)
    #     # print(loss_2)
    #     # stop

    #     # criterion = ATLoss()
    #     # loss_2 = criterion(prob[:,40:], targets[:,40:])
    #     # loss_2 = (loss_2.view(input_ids.size(0), -1)).sum(dim=-1).mean()

    #     # # [batch * max_len * max_len]
    #     # targets = partial_mask_to_targets(eval_masks).view(-1)
    #     # # [batch * max_len * max_len, label_size]
    #     # prob = log_potentials.reshape(-1, label_size)
    #     # loss_2 = F.cross_entropy(prob, targets, reduction='none')
    #     # # [batch, max_len, max_len]
    #     # mask = tmu.lengths_to_squared_mask(lengths, max_len)
    #     # # [batch, max_len, max_len] -> [batch * max_len * max_len]
    #     # mask = torch.triu(mask.float()).view(-1)
    #     # loss_2 = (loss_2 * mask).sum() / mask.sum()
        
    #     outputs = [loss_2 + loss_3 * 0.5 + loss_4 * 0.5, inspect]
    #     # print(loss_2)
    #     return outputs

    def infer_0(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        """
        原始
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
        Returns:
            outputs: list 
        """
        label_size = self.label_size

        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # print(sequence_output.shape)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)

        if (self.use_crf is False):
            # [batch, max_len, max_len]
            trees = log_potentials.argmax(-1)
            max_len = log_potentials.size(1)
            # [batch, max_len, max_len]
            mask = tmu.lengths_to_squared_mask(lengths, max_len)
            mask = torch.triu(mask.float())
            trees = trees * mask - (1. - mask)
        else:
            trees = self.tree_crf.decode(log_potentials, lengths)

        outputs = [trees]
        return outputs

    def infer_1(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        """
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
        Returns:
            outputs: list 
        """
        label_size = self.label_size

        outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs_bert[0]
        sequence_output = self.dropout(sequence_output)
        # print(sequence_output.shape)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)

        # if (self.use_crf is False):
        #     # [batch, max_len, max_len]
        #     trees = log_potentials.argmax(-1)
        #     max_len = log_potentials.size(1)
        #     # [batch, max_len, max_len]
        #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
        #     mask = torch.triu(mask.float())
        #     trees = trees * mask - (1. - mask)
        # else:
        #     trees = self.tree_crf.decode(log_potentials, lengths)

        # padding mask
        batch_size, max_len, _, label_size = log_potentials.shape
        pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        log_potentials = log_potentials.permute(0, 3, 1, 2)
        # torch.set_printoptions(profile="full")
        # print(log_potentials)
        log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

        # 排除下三角
        mask = torch.tril(torch.ones_like(log_potentials), -1)
        log_potentials = log_potentials - mask * 1e6
        log_potentials = log_potentials / 64 ** 0.5

        # outputs = [min(trees, preds)]
        outputs = [log_potentials]
        # positive_mask = log_potentials > 0

        # # 使用布尔掩码提取正的元素
        # positive_elements = log_potentials[positive_mask]

        
        # print(log_potentials)
        # print(pad_mask)
        # print(mask)
        # print(positive_elements)
        
        return outputs
    
    def infer_now(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        """
        掩码语言模型 + 掩码插值表达 + KL损失
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
        Returns:
            outputs: list 
        """
        label_size = self.label_size

        outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        outputs_bert_embedding = self.bert.get_input_embeddings()(input_ids)

        sequence_output_bert = outputs_bert[0]
        # outputs_bert = self.mlm(input_ids, position_ids=None, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, output_hidden_states=True)
        # sequence_output_bert = outputs_bert.hidden_states[-1]

        batch_size, seq_len = input_ids.size()
        mask_token_id = self.tokenizer.mask_token_id

        # Create masked input_ids
        masked_input_ids = input_ids.unsqueeze(1).repeat(1, seq_len, 1)
        mask = torch.eye(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)
        masked_input_ids = masked_input_ids.masked_fill(mask == 1, mask_token_id).to(input_ids.device)

        # Expand token_type_ids and attention_mask to match masked_input_ids
        token_type_ids_ = token_type_ids.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        attention_mask_ = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        masked_input_ids = masked_input_ids.view(batch_size * seq_len, -1)

        # Get BERT outputs
        # mlm_logits = []
        # for i in range(masked_input_ids.shape[0]):
        #     outputs = self.bert(masked_input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), token_type_ids=token_type_ids[i].unsqueeze(0))
        #     mlm_logits.append(outputs.logits)

        # mlm_logits = torch.stack(mlm_logits, dim=0).to(input_ids.device)  # 转换回GPU

        outputs = self.mlm(masked_input_ids, attention_mask=attention_mask_, token_type_ids=token_type_ids_)
        mlm_logits = outputs.logits

        # Get probabilities through softmax
        mlm_probs = torch.softmax(mlm_logits, dim=-1)
        
        # Extract masked token probabilities
        mlm_probs = mlm_probs.view(batch_size, seq_len, seq_len, -1)
        mask_positions = mask.nonzero(as_tuple=True)
        masked_probs = mlm_probs[mask_positions].view(batch_size, seq_len, -1)

        # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        word_embeddings = self.mlm.get_input_embeddings().weight
        sequence_output_mlm = torch.matmul(masked_probs.to(input_ids.device), word_embeddings)
        sequence_output = self.mlm.roberta(inputs_embeds=sequence_output_mlm,attention_mask=attention_mask, token_type_ids=token_type_ids)[0]  * 0.5 + sequence_output_bert

        # outputs = self.mlm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # mlm_logits = outputs.logits

        # # Get probabilities through softmax
        # mlm_probs = torch.softmax(mlm_logits, dim=-1)

        # # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        # sequence_output = torch.matmul(mlm_probs.to(input_ids.device), word_embeddings) * 0.5 + sequence_output_bert

        sequence_output = self.dropout(sequence_output)
        # print(sequence_output.shape)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)

        # if (self.use_crf is False):
        #     # [batch, max_len, max_len]
        #     trees = log_potentials.argmax(-1)
        #     max_len = log_potentials.size(1)
        #     # [batch, max_len, max_len]
        #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
        #     mask = torch.triu(mask.float())
        #     trees = trees * mask - (1. - mask)
        # else:
        #     trees = self.tree_crf.decode(log_potentials, lengths)

        # padding mask
        batch_size, max_len, _, label_size = log_potentials.shape
        pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        log_potentials = log_potentials.permute(0, 3, 1, 2)
        # torch.set_printoptions(profile="full")
        # print(log_potentials)
        log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

        # 排除下三角
        mask = torch.tril(torch.ones_like(log_potentials), -1)
        log_potentials = log_potentials - mask * 1e6
        log_potentials = log_potentials / 64 ** 0.5

        # outputs = [min(trees, preds)]
        outputs = [log_potentials]
        # positive_mask = log_potentials > 0

        # # 使用布尔掩码提取正的元素
        # positive_elements = log_potentials[positive_mask]

        
        # print(log_potentials)
        # print(pad_mask)
        # print(mask)
        # print(positive_elements)
        
        return outputs

    def infer(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        """
        掩码语言模型 + 掩码插值表达 + KL损失 + 角色子空间
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
        Returns:
            outputs: list 
        """
        label_size = self.label_size

        outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        outputs_bert_embedding = self.bert.get_input_embeddings()(input_ids)

        sequence_output_bert = outputs_bert[0]
        # outputs_bert = self.mlm(input_ids, position_ids=None, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, output_hidden_states=True)
        # sequence_output_bert = outputs_bert.hidden_states[-1]

        batch_size, seq_len = input_ids.size()
        mask_token_id = self.tokenizer.mask_token_id

        # Create masked input_ids
        masked_input_ids = input_ids.unsqueeze(1).repeat(1, seq_len, 1)
        mask = torch.eye(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)
        masked_input_ids = masked_input_ids.masked_fill(mask == 1, mask_token_id).to(input_ids.device)

        # Expand token_type_ids and attention_mask to match masked_input_ids
        token_type_ids_ = token_type_ids.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        attention_mask_ = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        masked_input_ids = masked_input_ids.view(batch_size * seq_len, -1)

        # Get BERT outputs
        # mlm_logits = []
        # for i in range(masked_input_ids.shape[0]):
        #     outputs = self.bert(masked_input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), token_type_ids=token_type_ids[i].unsqueeze(0))
        #     mlm_logits.append(outputs.logits)

        # mlm_logits = torch.stack(mlm_logits, dim=0).to(input_ids.device)  # 转换回GPU

        outputs = self.mlm(masked_input_ids, attention_mask=attention_mask_, token_type_ids=token_type_ids_)
        mlm_logits = outputs.logits

        # Get probabilities through softmax
        mlm_probs = torch.softmax(mlm_logits, dim=-1)
        
        # Extract masked token probabilities
        mlm_probs = mlm_probs.view(batch_size, seq_len, seq_len, -1)
        mask_positions = mask.nonzero(as_tuple=True)
        masked_probs = mlm_probs[mask_positions].view(batch_size, seq_len, -1)

        # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        word_embeddings = self.mlm.get_input_embeddings().weight
        sequence_output_mlm = torch.matmul(masked_probs.to(input_ids.device), word_embeddings)
        sequence_output = self.mlm.bert(inputs_embeds=sequence_output_mlm,attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        # outputs = self.mlm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # mlm_logits = outputs.logits

        # # Get probabilities through softmax
        # mlm_probs = torch.softmax(mlm_logits, dim=-1)

        # # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        # sequence_output = torch.matmul(mlm_probs.to(input_ids.device), word_embeddings) * 0.5 + sequence_output_bert

        batch_size, sequence_length, hidden_size = sequence_output_bert.shape
        # matrix = torch.ones(batch_size, sequence_length, hidden_size)
        types_embeddings = [self.bert(type_input.to(sequence_output_bert.device))[0] for type_input in self.types_inputs]
        selected_embeddings = [embedding[:,1,:] for embedding in types_embeddings]
        types_embeddings = torch.cat(selected_embeddings, dim=0)
        matrix = torch.matmul(torch.softmax(torch.matmul(sequence_output_bert, types_embeddings.view(hidden_size, -1)), -1), types_embeddings)
        sequence_output_bert_new = self.fusion_module(sequence_output_bert, matrix.to(sequence_output_bert.device))

        sequence_output = sequence_output * 0.5 + sequence_output_bert + sequence_output_bert_new * 0.5

        sequence_output = self.dropout(sequence_output)
        # print(sequence_output.shape)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)

        # if (self.use_crf is False):
        #     # [batch, max_len, max_len]
        #     trees = log_potentials.argmax(-1)
        #     max_len = log_potentials.size(1)
        #     # [batch, max_len, max_len]
        #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
        #     mask = torch.triu(mask.float())
        #     trees = trees * mask - (1. - mask)
        # else:
        #     trees = self.tree_crf.decode(log_potentials, lengths)

        # padding mask
        batch_size, max_len, _, label_size = log_potentials.shape
        pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        log_potentials = log_potentials.permute(0, 3, 1, 2)
        # torch.set_printoptions(profile="full")
        # print(log_potentials)
        log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

        # 排除下三角
        mask = torch.tril(torch.ones_like(log_potentials), -1)
        log_potentials = log_potentials - mask * 1e6
        log_potentials = log_potentials / 64 ** 0.5

        # outputs = [min(trees, preds)]
        outputs = [log_potentials]
        # positive_mask = log_potentials > 0

        # # 使用布尔掩码提取正的元素
        # positive_elements = log_potentials[positive_mask]

        
        # print(log_potentials)
        # print(pad_mask)
        # print(mask)
        # print(positive_elements)
        
        return outputs

    # def infer_4(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
    #     """
    #     掩码语言模型 + 掩码插值表达 + 类别子空间
    #     Args:
    #         input_ids: torch.LongTensor, size=[batch, max_len]
    #         token_type_ids:
    #         attention_mask:
    #         gather_ids:
    #         gather_masks: torch.FloatTensor, size=[batch, max_len]
    #     Returns:
    #         outputs: list 
    #     """
    #     label_size = self.label_size

    #     outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
    #                         attention_mask=attention_mask)

    #     sequence_output_bert = outputs_bert[0]
    #     # outputs_bert = self.mlm(input_ids, position_ids=None, token_type_ids=token_type_ids,
    #     #                     attention_mask=attention_mask, output_hidden_states=True)
    #     # sequence_output_bert = outputs_bert.hidden_states[-1]

    #     batch_size, seq_len = input_ids.size()
    #     mask_token_id = self.tokenizer.mask_token_id

    #     # Create masked input_ids
    #     masked_input_ids = input_ids.unsqueeze(1).repeat(1, seq_len, 1)
    #     mask = torch.eye(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)
    #     masked_input_ids = masked_input_ids.masked_fill(mask == 1, mask_token_id).to(input_ids.device)

    #     # Expand token_type_ids and attention_mask to match masked_input_ids
    #     token_type_ids = token_type_ids.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
    #     attention_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
    #     masked_input_ids = masked_input_ids.view(batch_size * seq_len, -1)

    #     # Get BERT outputs
    #     # mlm_logits = []
    #     # for i in range(masked_input_ids.shape[0]):
    #     #     outputs = self.bert(masked_input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), token_type_ids=token_type_ids[i].unsqueeze(0))
    #     #     mlm_logits.append(outputs.logits)

    #     # mlm_logits = torch.stack(mlm_logits, dim=0).to(input_ids.device)  # 转换回GPU

    #     outputs = self.mlm(masked_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     mlm_logits = outputs.logits

    #     # Get probabilities through softmax
    #     mlm_probs = torch.softmax(mlm_logits, dim=-1)
        
    #     # Extract masked token probabilities
    #     mlm_probs = mlm_probs.view(batch_size, seq_len, seq_len, -1)
    #     mask_positions = mask.nonzero(as_tuple=True)
    #     masked_probs = mlm_probs[mask_positions].view(batch_size, seq_len, -1)

    #     # Get word embeddings and form new sentence representations
    #     # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
    #     word_embeddings = self.mlm.get_input_embeddings().weight

    #     batch_size, sequence_length, hidden_size = sequence_output_bert.shape
    #     matrix = torch.ones(batch_size, sequence_length, hidden_size)
    #     sequence_output_bert_new = self.fusion_module(sequence_output_bert, matrix.to(sequence_output_bert.device))

    #     sequence_output = torch.matmul(masked_probs.to(input_ids.device), word_embeddings) * 0.5 + sequence_output_bert + sequence_output_bert_new * 0.5

    #     # outputs = self.mlm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     # mlm_logits = outputs.logits

    #     # # Get probabilities through softmax
    #     # mlm_probs = torch.softmax(mlm_logits, dim=-1)

    #     # # Get word embeddings and form new sentence representations
    #     # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
    #     # sequence_output = torch.matmul(mlm_probs.to(input_ids.device), word_embeddings) * 0.5 + sequence_output_bert

        
    #     sequence_output = self.dropout(sequence_output)

    #     gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
    #         sequence_output.shape)
    #     gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

    #     log_potentials = self.parser(gather_output)
    #     lengths = gather_masks.sum(1)

    #     # if (self.use_crf is False):
    #     #     # [batch, max_len, max_len]
    #     #     trees = log_potentials.argmax(-1)
    #     #     max_len = log_potentials.size(1)
    #     #     # [batch, max_len, max_len]
    #     #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
    #     #     mask = torch.triu(mask.float())
    #     #     trees = trees * mask - (1. - mask)
    #     # else:
    #     #     trees = self.tree_crf.decode(log_potentials, lengths)

    #     # padding mask
    #     batch_size, max_len, _, label_size = log_potentials.shape
    #     pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
    #     # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
    #     # pad_mask = pad_mask_v&pad_mask_h
    #     log_potentials = log_potentials.permute(0, 3, 1, 2)
    #     # torch.set_printoptions(profile="full")
    #     # print(log_potentials)
    #     log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

    #     # 排除下三角
    #     mask = torch.tril(torch.ones_like(log_potentials), -1)
    #     log_potentials = log_potentials - mask * 1e6
    #     log_potentials = log_potentials / 64 ** 0.5

    #     # outputs = [min(trees, preds)]
    #     outputs = [log_potentials]
    #     # positive_mask = log_potentials > 0

    #     # # 使用布尔掩码提取正的元素
    #     # positive_elements = log_potentials[positive_mask]

        
    #     # print(log_potentials)
    #     # print(pad_mask)
    #     # print(mask)
    #     # print(positive_elements)
        
    #     return outputs

    def infer_3(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        """
        掩码+ 树
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
        Returns:
            outputs: list 
        """
        label_size = self.label_size

        outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output_bert = outputs_bert[0]
        batch_size, seq_len = input_ids.size()
        mask_token_id = self.tokenizer.mask_token_id

        # Create masked input_ids
        masked_input_ids = input_ids.unsqueeze(1).repeat(1, seq_len, 1)
        mask = torch.eye(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)
        masked_input_ids = masked_input_ids.masked_fill(mask == 1, mask_token_id).to(input_ids.device)

        # Expand token_type_ids and attention_mask to match masked_input_ids
        token_type_ids = token_type_ids.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1).to(input_ids.device)
        masked_input_ids = masked_input_ids.view(batch_size * seq_len, -1)

        # Get BERT outputs
        # mlm_logits = []
        # for i in range(masked_input_ids.shape[0]):
        #     outputs = self.bert(masked_input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), token_type_ids=token_type_ids[i].unsqueeze(0))
        #     mlm_logits.append(outputs.logits)

        # mlm_logits = torch.stack(mlm_logits, dim=0).to(input_ids.device)  # 转换回GPU

        outputs = self.mlm(masked_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mlm_logits = outputs.logits

        # Get probabilities through softmax
        mlm_probs = torch.softmax(mlm_logits, dim=-1)
        
        # Extract masked token probabilities
        mlm_probs = mlm_probs.view(batch_size, seq_len, seq_len, -1)
        mask_positions = mask.nonzero(as_tuple=True)
        masked_probs = mlm_probs[mask_positions].view(batch_size, seq_len, -1)

        # Get word embeddings and form new sentence representations
        word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        sequence_output = torch.matmul(masked_probs.to(input_ids.device), word_embeddings) * 0.5+ sequence_output_bert

        # outputs = self.mlm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # mlm_logits = outputs.logits

        # # Get probabilities through softmax
        # mlm_probs = torch.softmax(mlm_logits, dim=-1)

        # # Get word embeddings and form new sentence representations
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        # sequence_output = torch.matmul(mlm_probs.to(input_ids.device), word_embeddings) * 0.5 + sequence_output_bert

        sequence_output = self.dropout(sequence_output)
        # print(sequence_output.shape)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)

        if (self.use_crf is False):
            # [batch, max_len, max_len]
            trees = log_potentials.argmax(-1)
            max_len = log_potentials.size(1)
            # [batch, max_len, max_len]
            mask = tmu.lengths_to_squared_mask(lengths, max_len)
            mask = torch.triu(mask.float())
            trees = trees * mask - (1. - mask)
        else:
            trees = self.tree_crf.decode(log_potentials, lengths)

        # padding mask
        batch_size, max_len, _, label_size = log_potentials.shape
        pad_mask = gather_masks.unsqueeze(1).unsqueeze(1).expand(batch_size, label_size, max_len, max_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        log_potentials = log_potentials.permute(0, 3, 1, 2)
        # torch.set_printoptions(profile="full")
        # print(log_potentials)
        log_potentials = log_potentials * pad_mask - (1 - pad_mask) * 1e6

        # 排除下三角
        mask = torch.tril(torch.ones_like(log_potentials), -1)
        log_potentials = log_potentials - mask * 1e6
        log_potentials = log_potentials / 64 ** 0.5

        # outputs = [min(trees, preds)]
        outputs = [log_potentials, trees]
        # positive_mask = log_potentials > 0

        # # 使用布尔掩码提取正的元素
        # positive_elements = log_potentials[positive_mask]

        
        # print(log_potentials)
        # print(pad_mask)
        # print(mask)
        # print(positive_elements)
        
        return outputs



    def infer_1(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        """
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
        Returns:
            outputs: list 
        """
        label_size = self.label_size

        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)

        # if (self.use_crf is False):
        #     # [batch, max_len, max_len]
        #     trees = log_potentials.argmax(-1)
        #     max_len = log_potentials.size(1)
        #     # [batch, max_len, max_len]
        #     mask = tmu.lengths_to_squared_mask(lengths, max_len)
        #     mask = torch.triu(mask.float())
        #     trees = trees * mask - (1. - mask)
        # else:
        #     trees = self.tree_crf.decode(log_potentials, lengths)

        # outputs = [trees]
        outputs = [log_potentials.argmax(-1)]
        return outputs

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[..., 0] = 1.0
        labels[..., 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e6
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e6
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label)

        # Sum two parts
        loss = loss1 + loss2
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[..., :1]
        output = torch.zeros_like(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[..., 0] = (output[..., 1:].sum(-1) == 0.)
        return output