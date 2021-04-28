import random
from util import get_sents_from_file, save_sents

file_path = "modal_sentences"
file_names = ["can.txt", "could.txt" , "may.txt" , "might.txt"  ,"must.txt",  "shall.txt"  ,"should.txt" , "will.txt"]

for name in file_names:
    sents = get_sents_from_file("{}/{}".format(file_path,name))
    test = random.sample(sents, 512)
    train = list(set(sents).difference(set(test)))
    save_sents(test, "{}/test/{}".format(file_path,name))
    save_sents(train,"{}/train/{}".format(file_path,name))


