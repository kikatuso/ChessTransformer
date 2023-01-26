import torch
from torch import Tensor,nn
import torch.nn.functional as f
import sys 
import numpy as np 
from gensim.models import Word2Vec
from torch.nn.modules.activation import MultiheadAttention
import os

dirname = os.path.dirname(__file__)


def attention_mechanism(query: Tensor, key: Tensor,value: Tensor,mask: Tensor) -> Tensor:
  num=query.bmm(key.transpose(1,2))
  num_masked = num.masked_fill(mask == 0, float("-1e20"))
  scale = query.size(-1) ** 0.5
  softmax = f.softmax(num_masked/scale,dim=-1)
  out = softmax.bmm(value)
  return out 

def position_encoding(seq_len: int, dim_model: int, device: torch.device = torch.device("cpu") )-> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim / dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int= 512, dim_feedforward : int = 2048) -> nn.Module:
  return nn.Sequential(
      nn.Linear(dim_input,dim_feedforward),
       nn.ReLU(),       
       nn.Linear(dim_feedforward, dim_input))


class AttentionHead(nn.Module):
  def __init__(self,dim_in: int,dim_q:int,dim_k :int):
    super().__init__()
    self.q = nn.Linear(dim_in,dim_q)
    self.k = nn.Linear(dim_in,dim_k)
    self.v = nn.Linear(dim_in,dim_k)

  def forward(self,query: Tensor,key: Tensor,value : Tensor, mask: Tensor) -> Tensor:
    return attention_mechanism(self.q(query),self.k(key),self.v(value),mask)


class MultiHeadAttention(nn.Module):
  def __init__(self,num_heads : int, dim_in: int, dim_q : int, dim_k : int):
    super().__init__()
    self.heads = nn.ModuleList(
        [AttentionHead(dim_in,dim_q,dim_k) for _ in range(num_heads)]
    )
    self.linear = nn.Linear(num_heads * dim_k,dim_in)

  def forward(self,query: Tensor,key: Tensor, value: Tensor,mask: Tensor) -> Tensor:
    out_attention = [h(query,key,value,mask) for h in self.heads]
    out_cat = torch.cat(out_attention,dim=-1)
    out_lin = self.linear(out_cat)
    return out_lin


class Residual(nn.Module):
  def __init__(self,sublayer: nn.Module, dimension: int,dropout: float = 0.1):
    super().__init__()
    self.sublayer = sublayer
    self.norm = nn.LayerNorm(dimension)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self,*tensors:Tensor) -> Tensor:
    return self.norm(tensors[0]+self.dropout(self.sublayer(*tensors)))



class Embedding(nn.Module):
  
  def __init__(self,embed_path:str="chess_embedding/chess2vec.model"):
    super().__init__()
    self.embed_layer=Word2Vec.load(os.path.join(dirname,embed_path))
    self.dim_embed  = self.embed_layer.vector_size
    self.corpus_length = len(self.embed_layer.wv.index_to_key)
    self.index_to_word = {key:value for (key,value) in enumerate(self.embed_layer.wv.index_to_key)}
    self.word_to_index = self.embed_layer.wv.key_to_index

    
  def embed(self,src):
    return torch.Tensor(np.array([self.embed_layer.wv[key] for key in src]))

  def translate_itw(self,src):    
    return np.vectorize(self.index_to_word.__getitem__)(src)

  def translate_wti(self,src):    
    return np.vectorize(self.word_to_index.__getitem__)(src)

  

class TransformerDecoderLayer(nn.Module):

  def __init__(self,
               dim_model:int = 512,
               num_heads:int = 6,
               dim_feedforward:int = 2048,
               dropout:float=0.1):
    
    super().__init__()
    dim_q = dim_k = max(dim_model // num_heads, 1)
    self.attention = Residual(
        MultiHeadAttention(num_heads,dim_model,dim_q,dim_k),
        dimension = dim_model,
        dropout = dropout)
    self.feed_forward = Residual(
        feed_forward(dim_model,dim_feedforward),
        dimension = dim_model,
        dropout = dropout)
    
  def forward(self,src: Tensor,mask:Tensor) -> Tensor:
    src = self.attention(src, src, src,mask)
    return self.feed_forward(src)



class Decoder(nn.Module):
  def __init__(self,
              embed_path:str="chess_embedding/chess2vec.model",
              num_layers:int=6,
              num_heads:int=6,
              dim_feedforward:int=2048,
              dropout:float=0.1):
    super().__init__()
    self.embed_layer=Embedding(embed_path=embed_path)
    self.layers = nn.ModuleList([TransformerDecoderLayer(self.embed_layer.dim_embed,num_heads,dim_feedforward,dropout) for _ in range(num_layers)])
    self.linear = nn.Linear(self.embed_layer.dim_embed,self.embed_layer.corpus_length)


  def masking(self,batch_size,seq_len):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        # returns the lower triangular part of matrix filled with ones
        mask = torch.tril(torch.ones((seq_len,seq_len))).expand(
            batch_size,seq_len,seq_len)
        
        return mask
    

  def forward(self, src: Tensor) -> Tensor:
    src = self.embed_layer.embed(src)
    batch_size,seq_len, dimension = src.size(0),src.size(1), src.size(2)
    src += position_encoding(seq_len, dimension)
    mask = self.masking(batch_size,seq_len)
    for layer in self.layers:
        src = layer(src,mask)
    out = self.linear(src)
    return torch.softmax(out,dim=-1)




class ChessTransformer(nn.Module):
  def __init__(self,
              embed_path:str="chess_embedding/chess2vec.model",
               num_layers:int=6,
               num_heads:int=6,
               dim_feedforward:int=2048,
               dropout:float=0.1):
    super().__init__()
    self.decoder = Decoder(embed_path=embed_path,
                           num_layers=num_layers,
                           num_heads=num_heads,
                           dim_feedforward=dim_feedforward,
                           dropout=dropout)
    
    self.embedding = Embedding(embed_path=embed_path)


  def forward(self, src: np.array) -> Tensor:
    out = self.decoder(src)
    return out


  def decode(self,src,num_moves):
      """
      for inference
      Args:
          src: input to decoder
      out:
          out_labels : returns final prediction of sequence
      """


      src = np.array(src)
      if len(src.shape)==1:
          src=np.expand_dims(src,0)
      out_seq = src
      for i in range(num_moves):
        out = self.decoder(out_seq) #bs x seq_len x vocab_d
        out = out[:,-1,:] # taking the last token
        out = torch.unsqueeze(out,axis=1)
        out = torch.argmax(out,-1)
        out = self.embedding.translate_itw(out)
        out_seq = np.append(out_seq,out,axis=1)

      return out_seq




