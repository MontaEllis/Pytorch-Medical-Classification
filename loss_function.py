from torch import nn
import torch
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Classification_Loss(nn.Module):
    def __init__(self):
        super(Classification_Loss, self).__init__()
        self.criterionCE = nn.CrossEntropyLoss()


    def forward(self, model_output, targets, model):

        # torch.empty(3, dtype=torch.long)
        # model_output = model_output.long()
        # targets = targets.long()
        # print(model_output)
        # print(F.sigmoid(model_output))
        # print(targets)
        # print('kkk')
        regularization_loss = 0
        for param in model.module.parameters():
            regularization_loss += torch.sum(torch.abs(param)) #+torch.sum(torch.abs(param))**2
        # loss = 0.00001 * regularization_loss
        loss = 0

        # model_output = F.sigmoid(model_output)
        # loss = self.mse_criterion(model_output,targets)
        loss += self.criterionCE(model_output,targets)
        return loss
