import torch
import torch.nn as nn
import math



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        
        return self.embedding(x) * math.sqrt(self.d_model) 






class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, batch) -> None:
        super(PositionalEncoding, self).__init__()
        # self.seq_len = seq_len
        # self.d_model = d_model
        # self.batch = batch
        self.dropout = nn.Dropout(p=0.1)
    
        ##initialize the positional encoding with zeros
        positional_encoding = torch.zeros(seq_len, d_model)
     
        ##first path of the equation is postion/scaling factor per dimesnsion
        postion  = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
        ## this calculates the scaling term per dimension (512)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # div_term = torch.pow(10,  torch.arange(0,self.d_model, 2).float() *-4/self.d_model)
      

        ## this calculates the sin values for even indices
        positional_encoding[:, 0::2] = torch.sin(postion * div_term) 

      
        ## this calculates the cos values for odd indices
        positional_encoding[:, 1::2] = torch.cos(postion * div_term)

        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)
    
    def forward(self, x):  
         x =  x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False).to(device)
         return self.dropout(x)





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, batch, heads) -> None:
        super(MultiHeadAttention,self).__init__()
        self.head = heads
        self.head_dim = d_model // heads
        


        assert self.d_model % self.heads == 0, 'cannot divide d_model by heads'

        ## initialize the query, key and value weights 512*512
        self.query_weight = nn.Linear(d_model, d_model, bias=False)
        self.key_weight = nn.Linear(d_model, d_model,bias=False)
        self.value_weight = nn.Linear(d_model, d_model,bias=False)
        self.final_weight  = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=0.1)

   

       


    # @staticmethod    
    def self_attention(self,query, key, value, mask,dropout):
        #splitting query, key and value into heads
        query = query.view(query.shape[0], query.shape[1],self.head,self.head_dim).transpose(2,1)
        key = key.view(key.shape[0], key.shape[1],self.head,self.head_dim).transpose(2,1)
        value = value.view(value.shape[0], value.shape[1],self.head,self.head_dim).transpose(2,1)


        
        attention = query @ key.transpose(3,2)

     
        

        attention = attention / math.sqrt(query.shape[-1])

        
        
        if mask is not None:
           attention = attention.masked_fill(mask == 0, -1e9)
            
        attention = torch.softmax(attention, dim=3)    
            
        if dropout is not None:
            attention = dropout(attention)


        attention_scores =  attention @ value 

     
        

        # attention_scores = attention_scores.transpose(2,1)
       
        return attention_scores.transpose(2,1).contiguous().view(attention_scores.shape[0], -1, self.head_dim * self.heads)
      

        #this gives us a dimension of batch, num_heads, seq_len by 64. basically 1 sentence is converted to have 8 parts (heads)
    def forward(self,query, key, value,mask):
        mask
      


        ## initialize the query, key and value matrices to give us seq_len by 512
        query = self.query_weight(query)
        key = self.key_weight(key)
        value = self.value_weight(value)


        
        attention = MultiHeadAttention.self_attention(self, query, key, value, mask, self.dropout)
        return self.final_weight(attention)



       
    


class LayerNormalization(nn.Module):
    def __init__(self) -> None:
        super(LayerNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added 

    def forward(self,x):
         
       

        ##calculates mean of layer ie 512 numbers
        mean = torch.mean(x, dim=-1,)
        std = torch.std(x, dim=-1,)

        ##normalizes the layer
     
        norm = (x - mean.unsqueeze(-1)) / (std.unsqueeze(-1) + 10**-6)
        return self.alpha * norm + self.bias    



class FeedForward(nn.Module):
    def __init__(self,d_model, d_ff, ) -> None:
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer
        self.fc2 = nn.Linear(d_ff, d_model)  # Fully connected layer 2

      
        
    
    def forward(self,x ):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super(ProjectionLayer, self).__init__()
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self, seq_len, batch, d_model, head, d_ff) -> None:
        super(EncoderBlock, self).__init__()    
        self.multiheadattention = MultiHeadAttention(d_model, batch, head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)
    def forward(self, x, src_mask):
        ## following th encoder structure
        ## position_encoding -> multiheadattention -> add&layer_norm -> dropout -> feedforward -> add&layer_norm -> dropout

       
       

        ##storing residual value
        x_resid = x
        x = self.multiheadattention(x,x,x, src_mask)
        x = self.dropout1(x)
        x = self.layer_norm1(x + x_resid)

        ## storing the 2nd residual value
        x_resid2 = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.layer_norm2(x + x_resid2)
        return x
    

class Encoder(nn.Module):
    def __init__(self, number_of_block, seq_len, batch, d_model, head, d_ff) -> None:
        super(Encoder, self).__init__()
 
        # Use nn.ModuleList to store the EncoderBlock instances
        self.encoders = nn.ModuleList([EncoderBlock(seq_len, batch, d_model, head, d_ff) 
                                       for _ in range(number_of_block)])

    def forward(self, x, src_mask):
        for encoder_block in self.encoders:
            x = encoder_block(x, src_mask)
        return x
     
    





class DecoderBlock(nn.Module):
    def __init__(self, seq_len, batch, d_model, head, d_ff) -> None:
        super(DecoderBlock, self).__init__()
        # self.seq_len = seq_len
        # self.d_model = d_model
        # print()
        # self.d_ff = d_ff
        # self.batch = batch
        # self.heads = head
        self.head_dim = d_model // head
        
        self.multiheadattention = MultiHeadAttention(d_model, batch, head)
        self.crossattention = MultiHeadAttention(d_model, batch, head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(d_model,d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
    def forward(self, x, src_mask, tgt_mask, encoder_output):
        ## following th encoder structure
        ## position_encoding -> multiheadattention -> add&layer_norm -> dropout -> feedforward -> add&layer_norm -> dropout

        
        
        x_resid = x
        x = self.multiheadattention(x,x,x, tgt_mask)
        x = self.dropout1(x)
        x = self.layer_norm1(x + x_resid)
        x_resid2 = x


        ##cross attention
        x = self.crossattention(x, encoder_output, encoder_output, src_mask,)
        x = self.dropout2(x)
        

       
        x = self.layer_norm2(x + x_resid2)
        x_resid3 = x
        x = self.feedforward(x)
        x = self.dropout3(x)
        x = self.layer_norm3(x + x_resid3)
        return x

class Decoder(nn.Module):
    def __init__(self, number_of_block, seq_len, batch, d_model, head, d_ff) -> None:
        super(Decoder, self).__init__()
        # self.number_of_block = number_of_block
        # self.seq_len = seq_len
        # self.batch = batch
        # self.d_model = d_model
        # self.heads = head
        # self.d_ff = d_ff

        # Use nn.ModuleList to store the DecoderBlock instances
        self.decoders = nn.ModuleList([DecoderBlock(seq_len, batch, d_model, head, d_ff) 
                                       for _ in range(number_of_block)])

    def forward(self, x, src_mask, tgt_mask, encoder_output):
        for decoder_block in self.decoders:
            x = decoder_block(x, src_mask, tgt_mask, encoder_output)
        return x       
    


class Transformer(nn.Module):
    def __init__(self, seq_len, batch, d_model,target_vocab_size, source_vocab_size, head: int = 8, d_ff: int =  2048, number_of_block: int = 6) -> None:
        super(Transformer, self).__init__()
    
       
        self.encoder = Encoder(number_of_block,seq_len, batch, d_model, head, d_ff )
        self.decoder = Decoder(number_of_block,seq_len, batch, d_model, head, d_ff )
        self.projection = ProjectionLayer(d_model, target_vocab_size)
        self.source_embedding = InputEmbeddings(d_model,source_vocab_size )
        self.target_embedding = InputEmbeddings(d_model,target_vocab_size)
        self.positional_encoding = PositionalEncoding(seq_len, d_model, batch)
   
    def encode(self,x, src_mask):
        x = self.source_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_mask)
        return x
    def decode(self,x, src_mask, tgt_mask, encoder_output):
        x = self.target_embedding(x)
        x = self.positional_encoding(x)
        x = self.decoder(x, src_mask,tgt_mask, encoder_output )
        return x
    def project(self, x):
        x = self.projection(x)
        return x


def build_transformer(seq_len, batch, target_vocab_size, source_vocab_size,  d_model)-> Transformer:
    

    transformer = Transformer(seq_len, batch,  d_model,  target_vocab_size, source_vocab_size )

      #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer 









          







        
       
        