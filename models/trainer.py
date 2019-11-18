import torch
import torch.nn as nn
from models.model import PTransformer,TransformerLM,LSTMLM
from torch.utils.data import DataLoader
from tqdm import tqdm,trange
from data.process import TextConverter,get_dataset,get_dataset_lm

def get_data(convert,opt):
    if opt.model in ['lstm','transformerlm']:
        dataset= get_dataset_lm(opt.data_path, convert.text_to_arr, opt.tgt_max_len)
    elif opt.model in ['transformer']:
        dataset= get_dataset(opt.data_path, convert.text_to_arr, opt.src_max_len, opt.tgt_max_len)
    return DataLoader(dataset, opt.batch_size, shuffle=True)

class Trainer_LSTM(object):
    def __init__(self, convert, opt):
        self.config=opt
        self.convert=convert
    
    def train(self):
        model=LSTMLM(d_model=self.config.hidden_dims,num_layers=self.config.num_layers,dropout=self.config.dropout,vocab_size=self.convert.vocab_size)
        if torch.cuda.is_available() and self.config.use_gpu:
            print("using gpu to accelerate")
            model=model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion=nn.CrossEntropyLoss(ignore_index=0,size_average=True)
        training_data=get_data(self.convert,self.config)
        for epoch in range(self.config.num_epochs):
            print("epoch:",epoch)
            running_loss=0
            updates=0
            for data in tqdm(training_data):
                tgt,label=data
                if torch.cuda.is_available() and self.config.use_gpu:
                    tgt=tgt.cuda()
                    label=label.cuda()
                optimizer.zero_grad()
                out=model(tgt,use_gpu=self.config.use_gpu and torch.cuda.is_available())
                #print(out,tgt)
                out=out.transpose(0,1).contiguous().view(-1,self.convert.vocab_size)
                label=label.view(-1)
                loss=criterion(out,label)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                updates+=1
                #print(loss)
            print("training loss:",running_loss/updates)
        torch.save(model.cpu(),"output/model-lstm.pkl")

    def test(self):
        model=torch.load('output/model-lstm.pkl')
        model.eval()
        model=model.cuda()
        tgt_list=[1]+self.convert.text_to_arr(self.config.tgt_text)
        for i in range(self.config.tgt_max_len):
            tgt=torch.tensor(tgt_list)
            tgt=tgt.cuda().unsqueeze(0)
            out=model(tgt,use_gpu=self.config.use_gpu and torch.cuda.is_available())
            #print(self.convert.int_to_word(int(out.argmax(-1)[0,-1])))
            if int(out.argmax(-1)[-1,0])==1:
                break
            tgt_list.append(int(out.argmax(-1)[-1,0]))
        print(self.convert.arr_to_text(tgt_list)[5:])

    def predict(self):
        pass

class Trainer_TransformerLM(object):
    def __init__(self, convert, opt):
        self.config=opt
        self.convert=convert
    
    def train(self):
        tgt_mask=torch.triu(torch.ones(self.config.tgt_max_len,self.config.tgt_max_len),1)
        tgt_mask=tgt_mask.masked_fill(tgt_mask.byte(),value=torch.tensor(float('-inf')))
        model=TransformerLM(d_model=self.config.hidden_dims, nhead=self.config.num_heads, 
                            num_layers=self.config.num_layers,  dim_feedforward=4*self.config.hidden_dims,
                            dropout=self.config.dropout,vocab_size=self.convert.vocab_size)
        if torch.cuda.is_available() and self.config.use_gpu:
            print("using gpu to accelerate")
            model=model.cuda()
            tgt_mask=tgt_mask.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion=nn.CrossEntropyLoss(ignore_index=0,size_average=True)
        training_data=get_data(self.convert,self.config)
        for epoch in range(self.config.num_epochs):
            print("epoch:",epoch)
            running_loss=0
            updates=0
            for data in tqdm(training_data):
                tgt,label=data
                if torch.cuda.is_available() and self.config.use_gpu:
                    tgt=tgt.cuda()
                    label=label.cuda()
                tgt_padding=torch.eq(tgt,0)
                tgt_nonpadding=torch.ne(tgt,0)
                optimizer.zero_grad()
                out=model(tgt, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_padding)
                #out=model(tgt,use_gpu=self.config.use_gpu and torch.cuda.is_available())
                #print(out,tgt)
                out=out.transpose(0,1).contiguous().view(-1,self.convert.vocab_size)
                label=label.view(-1)
                loss=criterion(out,label)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                updates+=1
                #print(loss)
            print("training loss:",running_loss/updates)
        torch.save(model.cpu(),"output/model-transformerlm.pkl")

    def test(self):
        model=torch.load('output/model-transformerlm.pkl')
        model.eval()
        model=model.cuda()
        tgt_list=[1]+self.convert.text_to_arr(self.config.tgt_text)
        for i in range(self.config.tgt_max_len):
            tgt=torch.tensor(tgt_list)
            tgt=tgt.cuda().unsqueeze(0)
            out=model(tgt)
            #print(self.convert.int_to_word(int(out.argmax(-1)[0,-1])))
            if int(out.argmax(-1)[-1,0])==1:
                break
            tgt_list.append(int(out.argmax(-1)[-1,0]))
        print(self.convert.arr_to_text(tgt_list)[5:])

    def predict(self):
        pass


class Trainer_Transformer(object):
    def __init__(self, convert, opt):
        self.config=opt
        self.convert=convert
    
    def train(self):
        tgt_mask=torch.triu(torch.ones(self.config.tgt_max_len,self.config.tgt_max_len),1)
        tgt_mask=tgt_mask.masked_fill(tgt_mask.byte(),value=torch.tensor(float('-inf')))
        model=PTransformer(d_model=self.config.hidden_dims, nhead=self.config.num_heads, 
                            num_encoder_layers=self.config.num_encoder_layers, num_decoder_layers=self.config.num_decoder_layers, 
                            dim_feedforward=4*self.config.hidden_dims, dropout=self.config.dropout,vocab_size=self.convert.vocab_size)
        if torch.cuda.is_available() and self.config.use_gpu:
            print("using gpu to accelerate")
            model=model.cuda()
            tgt_mask=tgt_mask.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion=nn.CrossEntropyLoss(ignore_index=0,size_average=True)
        training_data=get_data(self.convert,self.config)
        for epoch in range(self.config.num_epochs):
            print("epoch:",epoch)
            running_loss=0
            updates=0
            for data in tqdm(training_data):
                src,tgt,label=data
                if torch.cuda.is_available() and self.config.use_gpu:
                    src=src.cuda()
                    tgt=tgt.cuda()
                    label=label.cuda()
                src_padding=torch.eq(src,0)
                tgt_padding=torch.eq(tgt,0)
                #src_nonpadding=torch.ne(src,0)
                tgt_nonpadding=torch.ne(tgt,0)
                optimizer.zero_grad()
                out=model(src, tgt, tgt_mask=tgt_mask,
                        src_key_padding_mask=src_padding, tgt_key_padding_mask=tgt_padding, memory_key_padding_mask=src_padding)
                #out=model(tgt,use_gpu=self.config.use_gpu and torch.cuda.is_available())
                #print(out,tgt)
                out=out.transpose(0,1).contiguous().view(-1,self.convert.vocab_size)
                label=label.view(-1)
                loss=criterion(out,label)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                updates+=1
                #print(loss)
            print("training loss:",running_loss/updates)
        torch.save(model.cpu(),"output/model-transformer.pkl")

    def test(self):
        model=torch.load('output/model-transformer.pkl')
        model.eval()
        model=model.cuda()
        src=torch.tensor(self.convert.text_to_arr(self.config.src_text))
        src=src.cuda().unsqueeze(0)
        tgt_list=[1]+self.convert.text_to_arr(self.config.tgt_text)
        for i in range(self.config.tgt_max_len):
            tgt=torch.tensor(tgt_list)
            tgt=tgt.cuda().unsqueeze(0)
            out=model(src,tgt)
            if int(out.argmax(-1)[-1,0])==1:
                break
            tgt_list.append(int(out.argmax(-1)[-1,0]))
        print(self.convert.arr_to_text(tgt_list)[5:])

    def predict(self):
        pass