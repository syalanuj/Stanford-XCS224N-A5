#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class Highway(nn.Module):
    """ Highway Network for Conv NN to use
    """
    def __init__(self, embed_size, dropout_rate=0.2):
        """ Init Highway model
        """
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(embed_size,embed_size)
        self.w_gated = nn.Linear(embed_size,embed_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """ Model that takes x_conv_out and applies hightway with projection and gates
            returns output in same size
        """
        x_proj = F.relu(self.w_proj(x_conv_out))
        x_gate = torch.sigmoid(self.w_gated(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        x_word_embed = self.dropout(x_highway)
        return x_word_embed


### END YOUR CODE 

