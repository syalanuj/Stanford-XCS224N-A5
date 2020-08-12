#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.vocab = vocab
        self.embed_size = embed_size
        dropout_probability = 0.3
        self.highway = Highway(embed_size,dropout_probability)

        max_word_length = 21
        kernel_size = 5
        char_embedding_dimension = 50
        # embed_size,5,50,21
        #(kernel_size, embed_size, char_emb_dim, max_sentence_len)
        self.cnn = CNN(kernel_size,embed_size,char_embedding_dimension,max_word_length)
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), char_embedding_dimension, padding_idx=pad_token_idx)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        # print("===== before embeddings",input_tensor.size())

        embeddings = self.embeddings(input_tensor)
        
        # print("======= after applying embeddings",embeddings.size())
        sentence_length, batch_size, max_word_length, char_emb_dim  = embeddings.size()
        
        # print("===== changing shapes using permutation")
        embeddings_permuted = embeddings.permute(0, 1, 3, 2)
        # print("====after permutations",embeddings_permuted.size())
        # print("====reshaping")
        # embed_size,5,50,21
        flat_embeddings = embeddings_permuted.reshape(sentence_length*batch_size,char_emb_dim,max_word_length)
        # print("====after reshaping",flat_embeddings.size())
        x_conv = self.cnn(flat_embeddings)
        # print("====after cnn",x_conv.size())
        gate_out = self.highway(x_conv)
        # print("====after highway",gate_out.size())

        word_embedded = gate_out.reshape(sentence_length, batch_size, -1)
        # print("====after reshape",word_embedded.size())
        return word_embedded
        ### END YOUR CODE
