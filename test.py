from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('oppo.model')

f = open("test_trans.tsv", "r+", encoding="utf-8")
fout = open("result.txt", "w+", encoding="utf-8")
data = f.readlines()
for line in data:
    line = line.strip("\n")
    line = line.split("\t")
    # Sentences are encoded by calling model.encode()
    emb1 = model.encode(line[0])
    emb2 = model.encode(line[1])
    cos_sim = util.pytorch_cos_sim(emb1, emb2)
    fout.writelines("{}\n".format(cos_sim[0][0]))