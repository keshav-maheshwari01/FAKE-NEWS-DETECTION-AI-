from transformers import AutoTokenizer


#here we will change our news into tokenids as computer only can understand number then we embedd it 
tokenizer = AutoTokenizer.from_pretrained("microsoft/DeBERTa-v3-base")

# a = "hey this is human"
# token = tokenizer(a)
# print(token)
# for id in token["input_ids"]:
#     print(tokenizer.decode(id))



