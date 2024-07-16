"""
Part of this code was drawn from the awesome-align demo from https://github.com/neulab/awesome-align.
"""
import transformers
import torch
import itertools
import argparse
from tqdm import tqdm
from google.colab import drive

# pre-processing
def preprocess(src, tgt):
  sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
  token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
  wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
  ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
  sub2word_map_src = []
  for i, word_list in enumerate(token_src):
    sub2word_map_src += [i for x in word_list]
  sub2word_map_tgt = []
  for i, word_list in enumerate(token_tgt):
    sub2word_map_tgt += [i for x in word_list]

  # alignment
  align_layer = 8
  threshold = 1e-3
  model.eval()
  with torch.no_grad():
    out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
    out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

    dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

    softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
    softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

    softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

  align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
  align_words = set()
  for i, j in align_subwords:
    align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

  # align_words: set of (src, tgt) tuples, where src and tgt are source and target indices
  align_words = sorted(align_words)
  return align_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate alignments with awesome-align")
    parser.add_argument("--subsetfolder", required=True, type=str, help="Root directory that stores English filenames and each translation text.")
    
    args = parser.parse_args()

    model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Process English filenames
    with open(args.subset_folder + "en.txt", "r") as f:
        english_sentences = f.readlines()
        english_sentences = list(map(lambda x: x.rstrip(), english_sentences))

    # List of target languages contained in CoVoST 2
    lang_codes = ['lv', 'ar', 'fa', 'tr', 'cy', 'sv-SE', 'ca', 'mn', 'id', 'sl', 'et', 'de', 'ta']
    for lang_code in lang_codes:
        print(f"Processing {lang_code}...")

        target_sentences = []
        align_dict = []
        
        with open(args.subsetfolder + f"{lang_code}.txt", "r") as f:
            target_sentences = f.readlines()
            target_sentences = list(map(lambda x: x.rstrip(), target_sentences))

        # English sentences should equal target sentences length
        assert(len(english_sentences) == len(target_sentences))

        for src, tgt in zip(tqdm(english_sentences), target_sentences):
            align_dict.append(preprocess(src, tgt))

        with open(args.subsetfolder +f'align_{lang_code}.txt', 'w') as f:
            f.write('\n'.join(map(str, align_dict)))
