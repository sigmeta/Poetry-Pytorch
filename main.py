import torch
import torch.nn as nn
from models.transformer import PTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm,trange
from data.process import TextConverter,get_dataset

class Opt(object):
    #path
    data_path="dataset/poetry.txt"

    #model
    hidden_dims=256
    num_encoder_layers=1
    num_decoder_layers=2
    num_heads=2
    dropout=0.1

    #training
    use_gpu=True
    num_epochs=100
    batch_size=64
    lr=1e-3
    src_max_len=10
    tgt_max_len=64
    num_workers=1




def get_data(convert,opt):
    dataset= get_dataset(opt.data_path, convert.text_to_arr,convert.src_max_len,convert.tgt_max_len)
    return DataLoader(dataset, opt.batch_size, shuffle=True)


class Trainer(object):
    def __init__(self, convert):
        self.config=Opt()
        self.convert=convert
    
    def train(self):
        tgt_mask=torch.triu(torch.ones(self.config.tgt_max_len,self.config.tgt_max_len),1)
        tgt_mask=tgt_mask.masked_fill(tgt_mask.byte(),value=torch.tensor(float('-inf')))
        print(tgt_mask)
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
        torch.save(model.cpu(),"output/model.pkl")

    def eval(self):
        pass

    def test(self):
        model=torch.load('output/model.pkl')
        model.eval()
        model=model.cuda()
        src=torch.tensor(self.convert.text_to_arr('沁'))
        src=src.cuda().unsqueeze(0)
        tgt_list=[1]
        for i in range(self.config.tgt_max_len):
            tgt=torch.tensor(tgt_list)
            
            tgt=tgt.cuda().unsqueeze(0)
            print(src.size(),tgt.size())
            out=model(src,tgt)
            #print(self.convert.int_to_word(int(out.argmax(-1)[0,-1])))
            if int(out.argmax(-1)[0,-1])==1:
                break
            tgt_list.append(int(out.argmax(-1)[0,-1]))
        print(self.convert.arr_to_text(tgt_list)[1:])


    def predict(self):
        pass


def main():
    opt=Opt()
    convert = TextConverter(opt.data_path,src_max_len=opt.src_max_len, tgt_max_len=opt.tgt_max_len)
    trainer=Trainer(convert)
    trainer.train()
    trainer.test()


if __name__=="__main__":
    main()
