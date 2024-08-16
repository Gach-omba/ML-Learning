# will encode a transformer from scratch using pytorch
import torch # help in creating tensors to store the data
import torch.nn as nn # for module, linear and embedding classes and other helper functions
import torch.nn.functional as F # to help access the softmax activation function when calculating attention

from torch.optim import Adam # optimizer to help in backpropagation
from torch.utils.data import TensorDataset, DataLoader # for creating datasets and dataloaders at largescale and also large scale transformer networks

import lightning as L  #  it makes it easier to write our code and for automatic code optimization and scaling in the cloud


## prepare the dataset to be used by the model
## encode the tokens into numbers as nn.Embedding only accepts numbers as inputs

token_to_id={
    'what' :0,
    'is' :1,
    'statquest' :2,
    'awesome' :3,
    '<EOS>' :4
}
id_to_token=dict(map(reversed,token_to_id.items()))
# token to id and id to token help in formating the input and alos interpreting the output from the transformer
# in decoder model , the tokens used as input in training come from processing the prompt as well as the generated output
# the outputs used too come from both the input and output
# we therefore can code our inputs as follows:

## Our prompts are: what is statquest and statquest is what where the response is awesome
inputs= torch.tensor([[token_to_id["what"],
                       token_to_id["is"],
                       token_to_id["statquest"],
                       token_to_id["<EOS>"],
                       token_to_id["awesome"]],
                      [token_to_id["statquest"],
                       token_to_id["is"],
                       token_to_id["what"],
                       token_to_id["<EOS>"],
                       token_to_id["awesome"]]])
# the output of the decoder units become the labels
labels=torch.tensor([[token_to_id["is"],
                      token_to_id["statquest"],
                      token_to_id["<EOS>"],
                      token_to_id["awesome"],
                      token_to_id["<EOS>"]],
                     [token_to_id["is"],
                      token_to_id["what"],
                      token_to_id["<EOS>"],
                      token_to_id["awesome"],
                      token_to_id["<EOS>"]]])
dataset=TensorDataset(inputs,labels)   # this creates a dataset
dataloader=DataLoader(dataset)
# next we create word embedding - this will be done automatically by nn.Embedding     
# next we will create positional encoding - it commonly uses a sequence of alternating sine and cosine squiggles to calculate values for each token and embedding value
# next we will create positional embedding - it commonly uses a sequence of alternating sine and cosine squiggles to calculate values for each token and embedding value
 ## code to precompute and add position encoding values to tokens

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=2,max_len=6): # d_model = number of word embeddings words per token , max_len is the maximum number of tokens our transformer can process( input and output combined). In practice you should set them to higher values
        super().__init__()
        pe=torch.zeros(max_len,d_model) # matrix of positional encoding values
        
        position=torch.arange(start=0,end=max_len,step=1).float().unsqueeze(1) # column matrix to represent the positions pos for each token. This is based on the formula PE(pos,2i) = sin(pos/10000^(2i/d_model)) and PE(pos,2i+1) = cos(pos/10000^(2i/d_model)
        # torch.arange() is used to create a sequence of numbers between 0 and max_len.. .unsqueeze(1) turns the sequence of numbers into a column matrix
        
        embedding_index= torch.arange(start=0,end=d_model,step=2).float() # this is a row matrix embedding_position that represents the index i times 2 for each word embedding
        # setting step to 2 resuts in the same sequence of numbers that we would get if we multiplied i by 2
        div_term= 1/torch.tensor(10000.0)**(embedding_index/d_model)
        
        pe[:,0::2]=torch.sin(position*div_term) # 0,2,4
        pe[:,1::2]=torch.cos(position*div_term) #1,3,4...
        
         
        self.register_buffer('pe',pe)
    def forward(self,word_embeddings):
        return word_embeddings+self.pe[:word_embeddings.size(0),:]
## masked self attention
# the resultant matrix after multiplying query has one row per encoded token
# we multiply the word embedding + positional encoding matrix with the query weights matrix to get the queries
# We also mutltiply the word embedding + positional encoding matrix with the key weights matrix to get the keys
# Finally we multiply the word embedding + positional encoding matrix with the value weights matrix to get the values
class Attention(nn.Module):
    def __init__(self, d_model=2): # d_model is the dimensions of the model (the number of word embedding values per token)
        # the dimensions/ number of word embeddings need to be known so that we can define how large the weight matrices are that we use to create the Queries, Keys and values
        
        super().__init__()
        # create the query matrix to compute Q
        self.W_q = nn.Linear(in_features=d_model,out_features=d_model,bias=False) # in_features defines the number of rows in the weight matrix while out_features defines the number of columns in the weight matrix
        # since W_q is a linear object, it will do the math for us when the time comes
        self.W_k= nn.Linear(in_features=d_model,out_features=d_model,bias=False)
        self.W_v= nn.Linear(in_features=d_model,out_features=d_model,bias=False)
        
        # define the column and row values
        self.row_dim=0
        self.col_dim=1
        
        
    def forward(self,query,key,value,mask=None):
        ''' # for flexibility, we allow the query, key and values to be calculated from different token encodings
            encoder- decoder models have encoder-decoder attention where the keys and values are calculated from the encoded tokens in the encoder
            while the queries are calculated from the encoded tokens in the decoder
            
            Hence allowing the encodings to come from different sources gives us the flexibility to do Encoder-Decoder Attention if we want to
            We also want to be able to do Masked self-attention, we can pass in a mask
            
            query has one row per encoded token
            key has one row per encoded token
            value has one row per encoded token
            mask is a binary matrix which has 1 in the places where we want to apply the mask
            '''
           
        q=self.W_q(query) # calculate the query, key, values for each token by passing the encodings to each Linear() object
        k=self.W_k(key)
        v=self.W_v(value)
        
        # calculate the attention
        sims= torch.matmul(q,k.transpose(dim0=self.row_dim,dim1=self.col_dim)) # this calculates the similarities between the queries and keys we have in sims
        # scale the sims by the square root of the number of values used in each key
        
        scaled_sims = sims/ torch.tensor(k.size(self.col_dim) **0.5) # only helps out when the model is relatively large
        # add a mask if we are using one to the scaled similarities
        # masking hides the values we want to replace by allocating a large negative number
        if mask is not None:
            scaled_sims=scaled_sims.masked_fill(mask=mask, value=-1e9) # -1e9 is a large negative number that prevents early tokens from looking at later tokens
            # Trues are replaced with -1e9 while Falses are replaced with 0
        # next, run the softmax function on the scaled similarities
        attention_percentages= F.softmax(scaled_sims,dim=self.col_dim) # this is the attention percentages. It determines how much influence a token should have on the others
        # finally, we multiply the torch.matmul() of the values and the attention percentages
        attention_scores=torch.matmul(attention_percentages,v)
        
        return attention_scores
## Next, we create a class that will combine the first three steps and then add the residual connection to create a fully connected layer
## we then run the outputs through softmax to get the actual output
class DecoderOnlyTransformer(L.LightningModule):
# having to inherit lightning module only once avoids the overhead of having to import the module every time
    def __init__(self,num_tokens=4,d_model=2,max_len=6):
        # num_tokens - this is the number of tokens in the vocabulary, d_model is the number of word embeddings per token, max_len is the maximum number of tokens our transformer can process( input and output combined). In practice you should set them to higher values
        super().__init__()
        # create word embedding object
        self.we = nn.Embedding(num_embeddings=num_tokens,embedding_dim=d_model) # the embedding needs to know the number of tokens in each vocabulary and the number of representations per token (d_model)
        # create positional encoding object
        self.pe=PositionalEncoding(d_model=d_model,max_len=max_len)
        # create an attention object
        self.self_attention= Attention(d_model=d_model)
        
        # next, we create a fully connected layer
        self.fc_layer=nn.Linear(in_features=d_model,out_features=num_tokens) # how many inputs and outputs there are
        # create the loss function. If the model has multiple outputs, we can use nn.CrossEntropyLoss(). It will also automatically apply softmax to the outputs
        self.loss=nn.CrossEntropyLoss()
        
    # we put all pieces together in the forward method
    def forward(self, token_ids): # it takes as input an array of token id numbers that will be used as input to the transformers
        # embed the input ids
        word_embeddings=self.we(token_ids)
        position_encoding=self.pe(word_embeddings)
        # create the mask that will prevengt early tokens to access the late tokens when we calculate attention
        mask=torch.tril(torch.ones((token_ids.size(dim=0),token_ids.size(dim=0))))
        # torch.tril leaves the lower triangle as 1s and convert the rest into 0s
        mask=mask==0 # converts the 0s into Trues and the ones into Falses
        
        # we can now calculate attention
        self_attention_values=self.self_attention(
            position_encoding,
            position_encoding,
            position_encoding,
            mask=mask
        )
        # since the query, key and values will be calculated from the same token encodings, we pass one 3 times
        residual_connection=position_encoding+self_attention_values
        # next, we run everything through a fully connected layer
        fc_layer_output=self.fc_layer(residual_connection)
        
        return fc_layer_output
    
    ## next, let's write the code required to train it
    def configure_optimizers(self):
        # define the optimizer you will be using. In this case, we will be using Adam which is like SGD but is a little less stochastic
        return Adam(self.parameters(),lr=0.1) # all model parameters are passed to Adam where we then set up the learning rate. This will make the model learn very fast. However the default value is 0.001

    def training_step(self,batch,batch_idx): # we then train the model by taking the training batch and the index for that batch
        input_tokens,labels=batch # split the batch into input tokens and their labels
        output=self.forward(input_tokens[0])
        loss=self.loss(output,labels[0]) # compare the output from the transformers to the one of the known labels
        
        return loss
            
# running the model before training it
model = DecoderOnlyTransformer(num_tokens=len(token_to_id),d_model=2,max_len=6)

## NB , always test that everything works before training
# # this is a sample input prompt   
# model_input= torch.tensor([token_to_id["what"],
#                            token_to_id["is"],
#                            token_to_id["statquest"],
#                            token_to_id["<EOS>"]])   
# # get the input length as our model can only process a maximum of 6 tokens
# input_length= model_input.size(dim=0)
# # keeping track of the tokens in the input tells us how many we can track as output
# predictions=model(model_input)
# # we are only interested in the prediction that comes after the <EOS> token so we use -1 to index the output
# predicted_id= torch.tensor([torch.argmax(predictions[-1,:])]) # we use argmax to identify the token with the largest value
# # NB we don't have to take the token with the largest value if we don't want and that is often configured in more complicated models
# predicted_ids= predicted_id # we save the token so that we can output it later

# max_length =6
# for i in range(input_length,max_length): # keep generating tokens until we reach the maximum length
#     if (predicted_id == token_to_id["<EOS>"]): # or if we reach EOS token
#         break
#     model_input=torch.cat((model_input,predicted_id)) # each time we generate a new output token, we add it to the input, so that each prediction is made with the full context
    
#     predictions=model(model_input)
#     predicted_id= torch.tensor([torch.argmax(predictions[-1,:])]) # model then predicts the next token
#     predicted_ids=torch.cat((predicted_ids,predicted_id))
# # finally, we print out the generated tokens after converting them from id numbers to text

# print("Predicted Tokens:\n")
# for id in predicted_ids:
#     print("\t",id_to_token[id.item()])
        

# after confirming everything works, we can now train the model
trainer = L.Trainer(max_epochs=30)
trainer.fit(model,train_dataloaders=dataloader)
# this is a sample input prompt   
model_input= torch.tensor([token_to_id["what"],
                           token_to_id["is"],
                           token_to_id["statquest"],
                           token_to_id["<EOS>"]])   
# get the input length as our model can only process a maximum of 6 tokens
input_length= model_input.size(dim=0)
# keeping track of the tokens in the input tells us how many we can track as output
predictions=model(model_input)
# we are only interested in the prediction that comes after the <EOS> token so we use -1 to index the output
predicted_id= torch.tensor([torch.argmax(predictions[-1,:])]) # we use argmax to identify the token with the largest value
# NB we don't have to take the token with the largest value if we don't want and that is often configured in more complicated models
predicted_ids= predicted_id # we save the token so that we can output it later

max_length =6
for i in range(input_length,max_length): # keep generating tokens until we reach the maximum length
    if (predicted_id == token_to_id["<EOS>"]): # or if we reach EOS token
        break
    model_input=torch.cat((model_input,predicted_id)) # each time we generate a new output token, we add it to the input, so that each prediction is made with the full context
    
    predictions=model(model_input)
    predicted_id= torch.tensor([torch.argmax(predictions[-1,:])]) # model then predicts the next token
    predicted_ids=torch.cat((predicted_ids,predicted_id))
# finally, we print out the generated tokens after converting them from id numbers to text

print("Predicted Tokens:\n")
for id in predicted_ids:
    print("\t",id_to_token[id.item()])
        
        
        
            
        
        
