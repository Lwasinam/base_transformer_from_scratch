import torch
import torch.nn as nn
import math



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        print(x.shape) 
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        
        return self.embedding(x) * math.sqrt(self.d_model) 



class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, batch) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.batch = batch
        self.dropout = nn.Dropout(p=0.1)
    
        ##initialize the positional encoding with zeros
        self.positional_encoding = torch.zeros(self.seq_len, self.d_model)
     
        ##first path of the equation is postion/scaling factor per dimesnsion
        postion  = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
    
        ## this calculates the scaling term per dimension (512)
        # div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / d_model))

        div_term = torch.pow(10,  torch.arange(0,self.d_model, 2)*-4/self.d_model)
      

        ## this calculates the sin values for even indices
        self.positional_encoding[:, 0::2] = torch.sin(postion * div_term) 

        ## this calculates the cos values for odd indices
        self.positional_encoding[:, 1::2] = torch.cos(postion * div_term)
        self.register_buffer('pe', self.positional_encoding.to('cuda'))
    
    def forward(self, x):
         print(self.positional_encoding.get_device())
         print("Using deviccccce:", device)
         x =  x + (self.positional_encoding.unsqueeze(0)).requires_grad_(False).to('cuda')
         return self.dropout(x)





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, batch, heads) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.batch = batch
        self.heads = heads

        assert self.d_model % self.heads == 0, 'cannot divide d_model by heads'

        ## initialize the query, key and value weights 512*512
        self.query_weight = nn.Linear(self.d_model, self.d_model, )
        self.key_weight = nn.Linear(self.d_model, self.d_model,)
        self.value_weight = nn.Linear(self.d_model, self.d_model)
        self.final_weight  = nn.Linear(self.d_model, d_model)
        self.dropout = nn.Dropout(p=0.1)

   

        self.head_dim = d_model // self.heads


    @staticmethod    
    def self_attention(self,query, key, value, mask, dropout):
        #splitting query, key and value into heads
        query = query.view(query.shape[0], query.shape[1],self.heads,self.head_dim).transpose(2,1)
        key = key.view(key.shape[0], key.shape[1],self.heads,self.head_dim).transpose(2,1)
        value = value.view(value.shape[0], value.shape[1],self.heads,self.head_dim).transpose(2,1)
       
        attention = query @ key.transpose(3,2)
        

        attention = attention / math.sqrt(self.d_model)

        

        
        # print(mask.shape)
        # print(attention.shape)
        
        if mask is not None:
            attention = attention.masked_fill_(mask == 0, -1e9)
            
        attention = torch.softmax(attention, dim=3)    
            
        if dropout is not None:
            attention = dropout(attention)


        attention_scores =  attention @ value 
        


        return attention_scores.view(attention_scores.shape[0], attention_scores.shape[2], self.head_dim * self.heads).transpose(2,1).transpose(2,1)
      

        #this gives us a dimension of batch, num_heads, seq_len by 64. basically 1 sentence is converted to have 8 parts (heads)
    def forward(self,x, mask):
        self.mask  = mask
      


        ## initialize the query, key and value matrices to give us seq_len by 512
        self.query = self.query_weight(x)
        self.key = self.key_weight(x)
        self.value = self.value_weight(x)


        
        attention = MultiHeadAttention.self_attention(self, self.query, self.key, self.value, self.mask, self.dropout)
        return self.final_weight(attention)



       
    


class LayerNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added 

    def forward(self,x):
        self.x = x
       

        ##calculates mean of layer ie 512 numbers
        mean = torch.mean(self.x, dim=-1,)
        std = torch.std(self.x, dim=-1,)

        ##normalizes the layer
        norm = (self.x - mean.unsqueeze(-1)) / (std.unsqueeze(-1) + 1e-6)
        return self.alpha * norm + self.bias    



class FeedForward(nn.Module):
    def __init__(self,d_model, d_ff, ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(self.d_model, self.d_ff)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer
        self.fc2 = nn.Linear(self.d_ff, self.d_model)  # Fully connected layer 2

      
        
    
    def forward(self,x ):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.fc = nn.Linear(self.d_model, self.vocab_size)
    def forward(self, x):
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self, seq_len, batch, d_model, head, d_ff) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.batch = batch
        self.heads = head
        self.positional_encoding = PositionalEncoding(self.seq_len, self.d_model, self.batch)
        self.multiheadattention = MultiHeadAttention(self.d_model, self.batch, self.heads)
        self.layer_norm1 = LayerNormalize()
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(self.d_model, self.d_ff)
        self.layer_norm2 = LayerNormalize()
        self.dropout2 = nn.Dropout(p=0.1)
    def forward(self, x, src_mask):
        ## following th encoder structure
        ## position_encoding -> multiheadattention -> add&layer_norm -> dropout -> feedforward -> add&layer_norm -> dropout

        x = self.positional_encoding(x)
        x = self.dropout1(x)

        ##storing residual value
        x_resid = x
        x = self.multiheadattention(x, src_mask)
        x = self.layer_norm1(x + x_resid)

        ## storing the 2nd residual value
        x_resid2 = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.layer_norm2(x + x_resid2)
        return x 
    

class Encoder(nn.Module):
    def __init__(self, number_of_block, seq_len, batch, d_model, head, d_ff) -> None:
        super().__init__()
        self.number_of_block = number_of_block
        self.seq_len = seq_len
        self.batch = batch
        self.d_model = d_model
        self.heads = head
        self.d_ff = d_ff

        self.encoder = EncoderBlock(self.seq_len, self.batch, self.d_model, self.heads, self.d_ff)
    def forward(self,x, src_mask):
        for i in range(self.number_of_block):
            x = self.encoder(x, src_mask)
        return x    





class DecoderBlock(nn.Module):
    def __init__(self, seq_len, batch, d_model, head, d_ff) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.batch = batch
        self.heads = head
        self.head_dim = self.d_model // self.heads
        self.positional_encoding = PositionalEncoding(self.seq_len, self.d_model, self.batch)
        self.multiheadattention = MultiHeadAttention(self.d_model, self.batch, self.heads)
        self.layer_norm1 = LayerNormalize()
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(self.d_model, self.d_ff)
        self.layer_norm2 = LayerNormalize()
        self.dropout2 = nn.Dropout(p=0.1)
    def forward(self, x, src_mask, tgt_mask, encoder_output):
        ## following th encoder structure
        ## position_encoding -> multiheadattention -> add&layer_norm -> dropout -> feedforward -> add&layer_norm -> dropout

        x = self.positional_encoding(x)
        x = self.dropout1(x)
        x_resid = x
        x = self.multiheadattention(x, src_mask)
        x = self.layer_norm1(x + x_resid)
        x_resid2 = x


        ##cross attention
        x = MultiHeadAttention.self_attention(self,x, encoder_output, encoder_output, tgt_mask, self.dropout1)

        print(x.shape)
        x = self.layer_norm2(x + x_resid2)
        x_resid3 = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.layer_norm2(x + x_resid3)
        return x   
    
class Decoder(nn.Module):
    def __init__(self, number_of_block, seq_len, batch, d_model, head, d_ff) -> None:
        super().__init__()
        self.number_of_block = number_of_block
        self.seq_len = seq_len
        self.batch = batch
        self.d_model = d_model
        self.heads = head
        self.d_ff = d_ff

        self.decoder = DecoderBlock(self.seq_len, self.batch, self.d_model, self.heads, self.d_ff)


    def forward(self, x, src_mask, tgt_mask, encoder_output):
      
        for i in range(self.number_of_block):
            x = self.decoder(x, src_mask, tgt_mask, encoder_output )
        return x


class Transformer(nn.Module):
    def __init__(self, seq_len, batch, d_model,target_vocab_size, source_vocab_size, head: int = 8, d_ff: int =  2048, number_of_block: int = 6) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.number_of_blocks = number_of_block
        self.batch = batch
        self.heads = head
        self.target_vocab_size = target_vocab_size
        self.source_vocab_size  = source_vocab_size
   
       
        self.encoder = Encoder(self.number_of_blocks,self.seq_len, self.batch, self.d_model, self.heads, self.d_ff )
        self.decoder = Decoder(self.number_of_blocks,self.seq_len, self.batch, self.d_model, self.heads, self.d_ff )
        self.projection = ProjectionLayer(self.d_model, self.target_vocab_size)
        self.source_embedding = InputEmbedding(self.d_model,self.source_vocab_size )
        self.target_embedding = InputEmbedding(self.d_model,self.target_vocab_size)
       
    # def forward(self, mask, source_idx, target_idx, padding_idx ):
       


        
    #     x = self.encoder(self.target_embedding)
    #     x = self.decoder(self.source_embedding, mask, x)
    #     x = self.projection(x)
    #     return x
    def encode(self,x, src_mask):
        x = self.source_embedding(x)
        print('here')
        print(x.shape)
        x = self.encoder(x, src_mask)
        return x
    def decode(self,x, src_mask, tgt_mask, encoder_output,):
        x = self.target_embedding(x)
        x = self.decoder(x, src_mask,tgt_mask, encoder_output )
        return x
    def project(self, x):
        x = self.projection(x)
        return x


def build_transformer(seq_len, batch, target_vocab_size, source_vocab_size,  d_model):
    

    transformer = Transformer(seq_len, batch,  d_model,  target_vocab_size, source_vocab_size,  )

      #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer 









          







        
       
        