#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
# • The forward() function will need to map from xreshaped to xconv out.
# • Note that although the model description above is not batched, your forward() function
# should operate on batches of words.
# • Make sure that your module uses one nn.Conv1d layer (this is important for the autograder).
# • Use a kernel size of k = 5
class CNN(nn.Module):
    """ CNN Network
    """
    def __init__(self, kernel_size, filter_size, char_embed_dim, max_word_len):
        """ Init CNN model
        """
        # print("====cnnn",{"kernel_size":kernel_size,"filter_size":filter_size,"char_embed_dim":char_embed_dim,"max_word_len":max_word_len})
        super(CNN, self).__init__()

        self.w_conv = nn.Conv1d(char_embed_dim,filter_size,kernel_size)
        self.max_pool = nn.MaxPool1d(max_word_len - kernel_size + 1)
    
    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """ Takes word embedding then applies CNN to generate x_conv_out
        """
        # print("===cnn x_reshaped size",x_reshaped.size())
        x_conv = self.w_conv(x_reshaped)
        # print("===cnn x_conv size",x_conv.size())
        x_conv = F.relu(x_conv)
        # print("===cnn x_conv relu size",x_conv.size())
        x_conv_out = self.max_pool(x_conv)
        # print("===cnn x_conv_out relu size",x_conv_out.size())
        x_conv_out = x_conv_out.squeeze(2)
        return x_conv_out

### END YOUR CODE 

