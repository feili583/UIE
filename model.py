import torch
import torch.nn.functional as F
import torch_model_utils as tmu

from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel, BertForMaskedLM, BertTokenizer, RobertaForMaskedLM, RobertaTokenizer
from torch import nn

from tree_crf_layer import TreeCRFLayer
from parser import Bilinear, BiAffine, DeepBiaffine
from torch.cuda.amp import autocast, GradScaler


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
        # self.tokenizer=BertTokenizer.from_pretrained('./bert-base-cased')
        # self.mlm = BertForMaskedLM.from_pretrained('./bert-base-cased')
        # self.bert = RobertaModel(config)
        self.tokenizer = RobertaTokenizer.from_pretrained('./roberta-large')
        self.mlm = RobertaForMaskedLM.from_pretrained('./roberta-large')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
    def forward(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks, eval_masks):
        """
        添加掩码语言损失+掩码插值表达的损失
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

        # outputs_bert = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask)

        # sequence_output_bert = outputs_bert[0]
        outputs_bert = self.mlm(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, output_hidden_states=True)
        sequence_output_bert = outputs_bert.hidden_states[-1]
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
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        word_embeddings = self.mlm.get_input_embeddings().weight
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
        
        outputs = [loss_2 + loss_3 * 0.5, inspect]
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
    
    def infer(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        """
        掩码语言模型 + 掩码插值表达
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

        # outputs_bert = self.mlm(input_ids, position_ids=None, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask)

        # sequence_output_bert = outputs_bert[0]
        outputs_bert = self.mlm(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, output_hidden_states=True)
        sequence_output_bert = outputs_bert.hidden_states[-1]

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
        # word_embeddings = self.mlm.bert.embeddings.word_embeddings.weight
        word_embeddings = self.mlm.get_input_embeddings().weight
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