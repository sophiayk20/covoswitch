import argparse
import re
import random
import os
from tqdm import tqdm
from itertools import combinations

def process_string_to_tuple(inp_str):
  """
    Function takes in string-represented tuples and parses them using regular expression into tuples for further use in this code.

    Args: 
        inp_str (str): String-represented tuples, such as '(1, 2), (3, 4), (5, 6)'
    
    Returns:
        Actual list of tuples (List(tuple)): such as [(1, 2), (3, 4), (5, 6)]
  """
  tuple_strings = re.findall(r'\((.*?)\)', inp_str)
  list_of_tuples = [tuple(eval(tuple_str)) for tuple_str in tuple_strings]
  return list_of_tuples

# global variables of relevance
alignments = {} # lang dep
detected_boundaries = [] # lang indp
source_transcriptions = [] # lang indp: english transcriptions
target_transcriptions = {} # lang dep: target transcriptions
idx_dict = {} # lang indp: global IU detection store (dict of dicts, where 0th idx is idx of boundary, 1st idx indicates jth token in that transcription)

separator='<|IU_Boundary|>'
lang_codes = ['lv', 'ar', 'fa', 'tr', 'cy', 'sv-SE', 'ca', 'mn', 'id', 'sl', 'et', 'de', 'ta']

def open_global(transcriptions_folder):
  """
    Reads in data and stores in global variables of relevance.

    Args:
        transcriptions_folder (str): path to transcriptions folder

    Returns: 
        void
  """
  global alignments
  global detected_boundaries
  global source_transcriptions
  global target_transcriptions
  global idx_dict

  # store detected IU boundaries - LANG INDP (only to English)
  with open(transcriptions_folder+'boundary-detection.txt', 'r') as f:
    detected_boundaries = f.readlines()
    detected_boundaries = list(map(lambda x: x.rstrip(), detected_boundaries))

  # store source transcriptions (in English) - LANG INDP (only to English)
  with open(transcriptions_folder+'en.txt', 'r') as f:
    source_transcriptions = f.readlines()
    source_transcriptions = list(map(lambda x: x.rstrip(), source_transcriptions))

  print(f"length of detected boundaries: {len(detected_boundaries)}")

  # store idx dict
  cur_idx = 0
  for idx, sent in enumerate(tqdm(detected_boundaries)):
    # for each sentence, get IU boundary
    # ['The original statuette', 'is currently found', 'inside Churubusco Studios', 'in Mexico City']
    tokens = list(map(lambda x: x.rstrip(), sent.split(separator)))

    tok_dict = {}

    cur_idx = 0
    for tok_idx, token in enumerate(tokens):
      toklen = len(token.strip().split())
      # 'The original statuette' {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]}
      tok_dict[tok_idx] = [i for i in range(cur_idx, cur_idx +toklen)]
      cur_idx += toklen

    idx_dict[idx] = tok_dict

  for lang_code in lang_codes:
    print(f"Processing {lang_code}...")
    # Read target transcriptions - LANG DEP
    with open(transcriptions_folder+f"{lang_code}.txt", "r") as f:
      target_transcriptions[lang_code] = f.readlines()
      target_transcriptions[lang_code] = list(map(lambda x: x.rstrip(), target_transcriptions[lang_code]))

    # Read alignments
    with open(transcriptions_folder+f"align_{lang_code}.txt", "r") as f:
      alignments[lang_code] = f.readlines()
      alignments[lang_code] = list(map(lambda x: process_string_to_tuple(x.rstrip()), alignments[lang_code]))

# Code-Switching Text Creation
def find_target_idx(src_indices, alignment, print_bool=False):
    """
        # for each IU to replace, call this function, and will return tgt_indices in this order

        Args:
            src_indices (List(int)): list of within IU indices in source text that we must match
            alignment (List(tuple)): List of word-level alignments between source and target text
        
        Returns:
            Target language indices mapped to src_indices fed in as argument (List(int))
    """
    # list of [(src, tgt), (src, tgt)]
    # exception: <|IU_Boundary|> at the end of the sentence -> src_indices empty (idx:98)
    if not src_indices:
      return []
    tgt_indices = []
    for src_idx in src_indices:
        # can be 1 to many mapping from src to target
        # Others, however, disagreed
        # Altres, tanmateix, no estaven d'acord.
        # store all target indices mappings for this source index.
        this_idx = []
        for src_t, tgt_t in alignment:
        # if matches our search, return first such matched item, and if last element appended is not this tgt index
        # prevent repetition matches
        # the inner ear -> l'oïda l'oïda l'oïda
            if src_t == src_idx and tgt_indices and tgt_indices[-1] != tgt_t: # matches our search - return first such matched item
                this_idx.append(tgt_t)
            elif src_t == src_idx and not tgt_indices:
              this_idx.append(tgt_t)
        if not this_idx:
            tgt_indices.append(-1) # no alignment found, will replace with empty string in IU replacement
        else:
            tgt_indices.extend(this_idx)
    if not tgt_indices:
      print("tgt_indices empty")
      print(f"src indices: {src_indices}")
    # only keep set of tgt indices, in case multiple words are mapped to same word
    unique_tgt_indices = list(set(tgt_indices))
    # return tgt indices in tgt order
    unique_tgt_indices.sort()
    # and if we missed any word, add to the indices
    min_tgt_idx = min(unique_tgt_indices)
    max_tgt_idx = max(unique_tgt_indices)
    new_tgt_indices = [i for i in range(min_tgt_idx, max_tgt_idx+1)]
    if len(new_tgt_indices) == len(unique_tgt_indices) + 1:
      return new_tgt_indices
    return unique_tgt_indices

# To simulate a more active code-switcher, prevent choice of consecutive indices
def nonconsecutive_combinations(iterable, r):
    """
        Given a list of intonation units, gives a combination of nonconsecutive indices to replace.
        
        Args:
            iterable (List(int)): list of intonation units
            r (int): number of intonation units to replace (r == 1 means we are replacing 1 intonation unit)
        
        Returns:
            list of combinations to replace (List(int))
    """
    if not iterable:
        return

    if r == 1:
        for item in iterable:
            yield (item,)
        return

    for combination in combinations(iterable, r):
        if all(combination[i+1] - combination[i] != 1 for i in range(len(combination)-1)):
            yield combination

def create_replacements(lang_code, csdata_folder):
    if not os.path.exists(csdata_folder):
       os.makedirs(csdata_folder)
    
    random.seed(10)
    with open(csdata_folder+f"cs_lang_{lang_code}.txt", "w") as f:
        with open(csdata_folder+f"cs_lang_{lang_code}_stat.txt", "w") as fp:
          with open(csdata_folder+f"cs_lang_{lang_code}_choseidx.txt", "w") as fpp:
            with open(csdata_folder+f"cs_lang_{lang_code}_enfiles.txt", "w") as fp4:
              with open(csdata_folder+f"cs_lang_{lang_code}_datasetinfo.txt", "w") as fp5:
                with open(csdata_folder+f"cs_lang_{lang_code}_targetfiles.txt", "w") as fp6:
                  for idx in tqdm(idx_dict.keys()):
                  #for idx in tqdm(range(5)):
                      # dict for this token - based on source indices
                      token_dict = idx_dict[idx]
                      # num tokens
                      num_tokens = len(token_dict.keys())
                      # source tokens
                      src_tokens = source_transcriptions[idx].strip().split()
                      # target tokens
                      tgt_tokens = target_transcriptions[lang_code][idx].split()

                      for r in range(1, num_tokens): # r = 1, 2, ..., n - 1
                          # for each r count, we will choose 1 random combination
                          # ex. combs = [(0,), (1,), (2,), (3,)] -> chosen_comb = random.choice(combs)
                          combs = []
                          combs = list(nonconsecutive_combinations(range(0, num_tokens), r))
                          if not combs: # no nonconsecutive choices possible
                              combs = list(combinations(range(0, num_tokens), r))
                          chosen_comb = random.choice(combs)

                          # indices to replace: key = src token_idx (int), value = tgt index (list of int)
                          indices_to_replace = {}
                          # choose which IU we are replacing -> if chosen_comb: (0, 1): token_idx => 0 or 1
                          for token_idx in chosen_comb:
                              # get indices of within IU in source (==English) [0, 1, 2] (source indices to replace)
                              src_indices = token_dict[token_idx]
                              # get indices of target indices
                              tgt_indices = find_target_idx(src_indices, alignments[lang_code][idx])
                              # indices_to_replace
                              indices_to_replace[token_idx] = tgt_indices

                          # now that we fetched all indices we have to replace, construct the code switched text
                          code_switched_text = []

                          # Creation of code-switched text
                          # MUST MEASURE CMI, SPF here!
                          src_token_count = 0
                          tgt_token_count = 0

                          spf_lst = []

                          # tok_idx: [0, 1, 2, 3]
                          for tok_idx in range(num_tokens):
                              # if tok_idx (0 or 1) was in the list of indices to replace in comb (we have to append target tokens)
                              if tok_idx in indices_to_replace.keys():
                                  fetched_tgt_indices = indices_to_replace[tok_idx]
                                  for item in fetched_tgt_indices:
                                      # no alignment matched, append empty string (continue)
                                      if item == -1:
                                          continue
                                      else:
                                        code_switched_text.append(tgt_tokens[item])
                                        spf_lst.append('tgt')
                                        tgt_token_count += 1
                              # if tok_idx is not in list of indices to replace in comb (we have to append source tokens)
                              else:
                                  # gets list of indices for that token
                                  for item in token_dict[tok_idx]:
                                      code_switched_text.append(src_tokens[item])
                                      spf_lst.append('src')
                                      src_token_count += 1

                          # now we finished creating code_switched_text, so join and write
                          # if for some reason, code_switched_text is equal to original text, skip
                          if " ".join(code_switched_text) == source_transcriptions[idx]:
                              continue
                          # if target not in spf_lst: -> exception handling 'fa', idx: 44960
                          if "tgt" not in spf_lst:
                            continue
                          if "src" not in spf_lst:
                            continue

                          f.write(" ".join(code_switched_text) + '\n')

                          fp4.write(source_transcriptions[idx] + '\n')
                          fpp.write(str(chosen_comb)+'\n')
                          fp6.write(target_transcriptions[lang_code][idx] + '\n')

                          # calculate code-switching relevant statistics
                          # for each sentence, we create the following
                          # 1. CMI
                          cmi = 1 - ((max(src_token_count, tgt_token_count) / (src_token_count+tgt_token_count)))

                          # 2. SPF
                          spf_sum = 0
                          for i in range(0, len(spf_lst)-1):
                              # if tokens[i] and tokens[i+1] belong to different languages
                              if spf_lst[i] != spf_lst[i+1]:
                                  spf_sum += 1
                          spf = spf_sum / (len(spf_lst)-1)
                          fp.write(f'cmi: {cmi:.2f} spf: {spf:.2f}' + '\n')

                          # we are also going to keep track of how many source and target tokens there are
                          fp5.write(f'src: {src_token_count} tgt: {tgt_token_count} tot: {src_token_count+tgt_token_count}' + '\n')

                      # finished generating code-switched text for this index

# Parse the code-switching fraction file -> for descriptive table!
def get_statistics(lang_code, csdata_folder):
    """
        Reads in statistics saved by create_replacements function and prints descriptive statistics.

        Args: 
            lang_code (str): language to parse, i.e. 'ca'
            csdata_folder: root directory to where code-switched data and other relevant data was saved.
        
        Returns:
            void
    """
    with open(csdata_folder+f"cs_lang_{lang_code}_stat.txt", "r") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.rstrip(), lines))
        lines = [line for line in lines if "=" not in line]

    with open(csdata_folder+f"cs_lang_{lang_code}_datasetinfo.txt", "r") as f:
        dsinfo = f.readlines()
        dsinfo = list(map(lambda x: x.rstrip(), dsinfo))
        dsinfo = [info for info in dsinfo]

    cmi_sum = 0
    spf_sum = 0

    total_generated = len(lines)

    for line in lines:
        # ['cmi:', '0.25', 'spf:', '0.18']
        data = line.split()
        cmi_sum += float(data[1])
        spf_sum += float(data[3])

    src_percentage_sum = 0
    tgt_percentage_sum = 0
    for info in dsinfo:
        # [0, 1, 2, 3, 4, 5]
        # ['src:' , '15' 'tgt:', '20', 'total:' , '35']
        data = info.split()
        src_percentage_sum += float("{:.2f}".format(100*(float(data[1])/float(data[5]))))
        tgt_percentage_sum += float("{:.2f}".format(100*(float(data[3])/float(data[5]))))

    print(f"Total generated: {total_generated:,}") # make number print with commas for every 3 digits
    print(f"Average src percentage: {src_percentage_sum / total_generated:.2f}")
    print(f"Average tgt percentage: {tgt_percentage_sum / total_generated:.2f}")
    print(f"Average CMI: {100 * (cmi_sum / total_generated):.2f}")
    print(f"Average SPF: {(spf_sum / total_generated):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replace intonation units with code-switched text")
    parser.add_argument("--transcriptionsfolder", required=True, type=str, help="Root directory to your transcriptions")
    parser.add_argument("--csfolder", required=True, type=str, help="Root directory to write code-switched data")
    args = parser.parse_args()

    open_global(args.transcriptionsfolder)

    '''
    examples:
        print(source_transcriptions[0]) - English transcriptions
        >> The original statuette is currently found inside Churubusco Studios in Mexico City.

        print(target_transcriptions['ca'][0]) - Catalan transcriptions
        >> L’estàtua original es troba actualment als Estudis Churubusco a la ciutat de Mèxic.

        print(alignments['ca'][0]) - Alignments between English and Catalan
        >> [(0, 0), (1, 1), (2, 0), (3, 2), (4, 4), (5, 3), (6, 5), (7, 7), (8, 6), (9, 8), (10, 12), (11, 10), (11, 12)]

        print(len(idx_dict.keys()) - Number of utterances used
        >> 195153
    '''

    for lang_code in lang_codes:
       # Create code-switched text and save to args.csfolder
       create_replacements(lang_code, args.csfolder)
       
        # Parse descriptive statistics
       get_statistics(lang_code, args.csfolder)

