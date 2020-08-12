#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        pad_token_idx = target_vocab.char2id['<pad>']
        self.target_vocab = target_vocab
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, padding_idx=pad_token_idx)
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id), bias=True)
        
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # print("=====input.size",input.size())
        char_embedded= self.decoderCharEmb(input)
        # print("=====char_embedded.size",char_embedded.size())
        out, dec_hidden = self.charDecoder(char_embedded,dec_hidden)
        # print("=====out.size",out.size()) #dimensions (seq_length, batch, hidden_size)
        
        out_batch_first = out.permute(1, 0, 2) #dimensions (seq_length, batch, hidden_size)
        o_proj = self.char_output_projection(out_batch_first)
        scores = o_proj.permute(1, 0, 2) #dimensions (seq_length, batch, hidden_size)
        return scores,dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.
        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        # the input sequence for the CharDecoderLSTM is [x1, . . . , xn] = [<START>,m,u,s,i,c]
        # the target sequence for the CharDecoderLSTM is [x2, . . . , xn+1] = [m,u,s,i,c,<END>].
        inp_char_seq = char_sequence[: -1, :]
        target_out_seq = char_sequence[1:, :]

        # shape (seq_length, batch, vocab_size), ((1, batch, hidden_size), (1, batch, hidden_size))
        scores, dec_hidden = self.forward(inp_char_seq, dec_hidden)

        # create target mask at padded locations - shape (seq_length, )
        target_masks = (target_out_seq != self.target_vocab.char2id['<pad>']).float()

        # calculate loss
        log_softmax_scores = nn.functional.log_softmax(scores, dim=2)
        loss_per_timestep = -1 * torch.gather(log_softmax_scores, index=target_out_seq.unsqueeze(2), dim=2).squeeze(2)
        loss_per_timestep_masked = loss_per_timestep * target_masks
        loss = loss_per_timestep_masked.sum()
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.

        Note that although Algorithm 1 is described for a single example,
        your implementation must work over a batch. Algorithm 1 also indicates that you should break
        when you reach the <END> token, but in the batched case you might find it more convenient to
        complete all max length steps of the for-loop, then truncate the output words afterwards.
        Run the following for a non-exhaustive sanity check:
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        def dec_step(prev_chars, dec_hidden):
            """Run decoder for one step for all chars in batch
            @param prev_chars (torch.Tensor): characters decoded in prev step. Tensor of shape (1, batch_size)
            @param dec_hidden (torch.Tensor): previous hidden state of decoder LSTM """

            scores, dec_hidden = self.forward(prev_chars, dec_hidden)
            next_chars = scores.argmax(dim=-1)
            return next_chars, dec_hidden

        words_for_batch = []
        batch_size = list(initialStates[0].size())[1]
        prev_chars = self.target_vocab.char2id['{'] * torch.ones((1, batch_size), dtype=torch.long,
                                                                          device=device)
        dec_hidden = initialStates
        
        for t in range(max_length):
            next_chars, dec_hidden = dec_step(prev_chars, dec_hidden)
            words_for_batch.append(next_chars)
            prev_chars = next_chars
        words_for_batch = torch.cat(words_for_batch, dim=0)
        words = []
        for i in range(batch_size):
            curr_word = ''
            for j in range(max_length):
                if words_for_batch[j][i] == self.target_vocab.char2id['}']:
                    break
                else:
                    curr_word += self.target_vocab.id2char[int(words_for_batch[j][i])]
            words.append(curr_word)
        return words
        ### END YOUR CODE

