import torch
import torch.nn as nn
import math



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
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
    def __init__(self, d_model, heads) -> None:
        super(MultiHeadAttention,self).__init__()
        self.head = heads
        self.head_dim = d_model // heads
        


        assert d_model % heads == 0, 'cannot divide d_model by heads'

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
       
        return attention_scores.transpose(2,1).contiguous().view(attention_scores.shape[0], -1, self.head_dim * self.head)
      

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
        return self.fc2(x)
     
    

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
        self.multiheadattention = MultiHeadAttention(d_model,head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)
    def forward(self, x, src_mask):
       # Self-attention block
        x_resid = x
        x = self.multiheadattention(x, x, x, src_mask)
        x = self.dropout1(x)
        x = x_resid + x
        x = self.layer_norm1(x)
        

        # Feedforward block
        # x = self.layer_norm1(x)
        x_resid2 = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        return  x_resid2 + x
        # return self.layer_norm3(x)
       
    

class Encoder(nn.Module):
    def __init__(self, number_of_block, seq_len, batch, d_model, head, d_ff) -> None:
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(d_model)
 
        # Use nn.ModuleList to store the EncoderBlock instances
        self.encoders = nn.ModuleList([EncoderBlock(seq_len, batch, d_model, head, d_ff) 
                                       for _ in range(number_of_block)])

    def forward(self, x, src_mask):
        for encoder_block in self.encoders:
            x = encoder_block(x, src_mask)
        return self.norm(x)
     
    
class DecoderBlock(nn.Module):
    def __init__(self, seq_len, batch, d_model, head, d_ff) -> None:
        super(DecoderBlock, self).__init__()
        self.head_dim = d_model // head
        
        self.multiheadattention = MultiHeadAttention(d_model, head)
        self.crossattention = MultiHeadAttention(d_model, head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(d_model,d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
    def forward(self, x, src_mask, tgt_mask, encoder_output):
         # Self-attention block
        x_resid = x
        x = self.multiheadattention(x, x, x, tgt_mask)
        x = self.dropout1(x)
        x = x_resid + x
        x = self.layer_norm1(x)
       

        # Cross-attention block
      
        x_resid2 = x
        x = self.crossattention(x, encoder_output, encoder_output, src_mask)
        x = self.dropout2(x)
        x = x_resid2 + x
        x = self.layer_norm2(x)
        

        # Feedforward block
        # x = self.layer_norm3(x)
        x_resid3 = x
        x = self.feedforward(x)
        x = self.dropout3(x)
        return x_resid3 + x
        # return self.layer_norm4(x)
        

class Decoder(nn.Module):
    def __init__(self, number_of_block, seq_len, batch, d_model, head, d_ff) -> None:
        super(Decoder, self).__init__()
        self.norm = nn.LayerNorm(d_model)
 
        self.decoders = nn.ModuleList([DecoderBlock(seq_len, batch, d_model, head, d_ff) 
                                       for _ in range(number_of_block)])

    def forward(self, x, src_mask, tgt_mask, encoder_output):
        for decoder_block in self.decoders:
            x = decoder_block(x, src_mask, tgt_mask, encoder_output)
        return self.norm(x)      
    


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
        return self.encoder(x, src_mask)
       
    def decode(self,x, src_mask, tgt_mask, encoder_output):
        x = self.target_embedding(x)
        x = self.positional_encoding(x)
        return self.decoder(x, src_mask,tgt_mask, encoder_output )
        
    def project(self, x):
        return self.projection(x)
        


def build_transformer(seq_len, batch, target_vocab_size, source_vocab_size,  d_model)-> Transformer:
    

    transformer = Transformer(seq_len, batch,  d_model,  target_vocab_size, source_vocab_size )

      #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer 


# import torch
# import torch.nn as nn
# import math

# class LayerNormalization(nn.Module):

#     def __init__(self, eps:float=10**-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter
#         self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter

#     def forward(self, x):
#         # x: (batch, seq_len, hidden_size)
#          # Keep the dimension for broadcasting
#         mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
#         # Keep the dimension for broadcasting
#         std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
#         # eps is to prevent dividing by zero or when std is very small
#         return self.alpha * (x - mean) / (std + self.eps) + self.bias

# class FeedForwardBlock(nn.Module):

#     def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
#         super().__init__()
#         self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

#     def forward(self, x):
#         # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
#         return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# class InputEmbeddings(nn.Module):

#     def __init__(self, d_model: int, vocab_size: int) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.vocab_size = vocab_size
#         self.embedding = nn.Embedding(vocab_size, d_model)

#     def forward(self, x):
#         # (batch, seq_len) --> (batch, seq_len, d_model)
#         # Multiply by sqrt(d_model) to scale the embeddings according to the paper
#         return self.embedding(x) * math.sqrt(self.d_model)
    
# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.seq_len = seq_len
#         self.dropout = nn.Dropout(dropout)
#         # Create a matrix of shape (seq_len, d_model)
#         pe = torch.zeros(seq_len, d_model)
#         # Create a vector of shape (seq_len)
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
#         # Create a vector of shape (d_model)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
#         # Apply sine to even indices
#         pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
#         # Apply cosine to odd indices
#         pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
#         # Add a batch dimension to the positional encoding
#         pe = pe.unsqueeze(0) # (1, seq_len, d_model)
#         # Register the positional encoding as a buffer
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
#         return self.dropout(x)

# class ResidualConnection(nn.Module):
    
#         def __init__(self, dropout: float) -> None:
#             super().__init__()
#             self.dropout = nn.Dropout(dropout)
#             self.norm = LayerNormalization()
    
#         def forward(self, x, sublayer):
#             return x + self.dropout(sublayer(self.norm(x)))

# class MultiHeadAttentionBlock(nn.Module):

#     def __init__(self, d_model: int, h: int, dropout: float) -> None:
#         super().__init__()
#         self.d_model = d_model # Embedding vector size
#         self.h = h # Number of heads
#         # Make sure d_model is divisible by h
#         assert d_model % h == 0, "d_model is not divisible by h"

#         self.d_k = d_model // h # Dimension of vector seen by each head
#         self.w_q = nn.Linear(d_model, d_model) # Wq
#         self.w_k = nn.Linear(d_model, d_model) # Wk
#         self.w_v = nn.Linear(d_model, d_model) # Wv
#         self.w_o = nn.Linear(d_model, d_model) # Wo
#         self.dropout = nn.Dropout(dropout)

#     @staticmethod
#     def attention(query, key, value, mask, dropout: nn.Dropout):
#         d_k = query.shape[-1]
#         # Just apply the formula from the paper
#         # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
       
#         attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
       
       
#         if mask is not None:
#             # Write a very low value (indicating -inf) to the positions where mask == 0
#             attention_scores.masked_fill_(mask == 0, -1e9)
#         attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
#         if dropout is not None:
#             attention_scores = dropout(attention_scores)
#         # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
#         # return attention scores which can be used for visualization
#         return (attention_scores @ value), attention_scores

#     def forward(self, q, k, v, mask):
#         query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
#         key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
#         value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

#         # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
#         query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
#         key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
#         value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

#         # Calculate attention
#         x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        
#         # Combine all the heads together
#         # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
#         x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

#         # Multiply by Wo
#         # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
#         return self.w_o(x)

# class EncoderBlock(nn.Module):

#     def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

#     def forward(self, x, src_mask):
#         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
#         x = self.residual_connections[1](x, self.feed_forward_block)
#         return x
    
# class Encoder(nn.Module):

#     def __init__(self, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization()

#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)

# class DecoderBlock(nn.Module):

#     def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         self.cross_attention_block = cross_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
#         x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
#         x = self.residual_connections[2](x, self.feed_forward_block)
#         return x
    
# class Decoder(nn.Module):

#     def __init__(self, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization()

#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         for layer in self.layers:
#             x = layer(x, encoder_output, src_mask, tgt_mask)
#         return self.norm(x)

# class ProjectionLayer(nn.Module):

#     def __init__(self, d_model, vocab_size) -> None:
#         super().__init__()
#         self.proj = nn.Linear(d_model, vocab_size)

#     def forward(self, x) -> None:
#         # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
#         return torch.log_softmax(self.proj(x), dim = -1)
    
# class Transformer(nn.Module):

#     def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.src_pos = src_pos
#         self.tgt_pos = tgt_pos
#         self.projection_layer = projection_layer

#     def encode(self, src, src_mask):
#         # (batch, seq_len, d_model)
#         src = self.src_embed(src)
#         src = self.src_pos(src)
#         return self.encoder(src, src_mask)
    
#     def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
#         # (batch, seq_len, d_model)
#         tgt = self.tgt_embed(tgt)
#         tgt = self.tgt_pos(tgt)
#         return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
#     def project(self, x):
#         # (batch, seq_len, vocab_size)
#         return self.projection_layer(x)
    
# def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
#     # Create the embedding layers
#     src_embed = InputEmbeddings(d_model, src_vocab_size)
#     tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

#     # Create the positional encoding layers
#     src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
#     tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
#     # Create the encoder blocks
#     encoder_blocks = []
#     for _ in range(N):
#         encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
#         feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
#         encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
#         encoder_blocks.append(encoder_block)

#     # Create the decoder blocks
#     decoder_blocks = []
#     for _ in range(N):
#         decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
#         decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
#         feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
#         decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
#         decoder_blocks.append(decoder_block)
    
#     # Create the encoder and decoder
#     encoder = Encoder(nn.ModuleList(encoder_blocks))
#     decoder = Decoder(nn.ModuleList(decoder_blocks))
    
#     # Create the projection layer
#     projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
#     # Create the transformer
#     transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
#     # Initialize the parameters
#     for p in transformer.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
    
#     return transformer






          







        
       
        