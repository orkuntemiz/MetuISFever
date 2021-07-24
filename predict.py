import tensorflow as tf
import tensorflow_hub as hub
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

torch_device = 'cuda'

QA_TOKENIZER = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")
QA_MODEL = AutoModelForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")

QA_MODEL.to(torch_device)
QA_MODEL.eval()

import sentencepiece
from transformers import BartTokenizer, BartForConditionalGeneration
SUMMARY_TOKENIZER = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
SUMMARY_MODEL = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

#from transformers import PegasusTokenizer, PegasusForConditionalGeneration
#SUMMARY_TOKENIZER = PegasusTokenizer.from_pretrained('mayu0007/pegasus_large_covid')
#SUMMARY_MODEL = PegasusForConditionalGeneration.from_pretrained('mayu0007/pegasus_large_covid')

SUMMARY_MODEL.to(torch_device)
SUMMARY_MODEL.eval()
def Summarizer(string_all):    
    if 1==1:
        ## try generating an exacutive summary with bart abstractive summarizer
        allAnswersTxt = string_all.replace('\n','')

        answers_input_ids = SUMMARY_TOKENIZER.batch_encode_plus([allAnswersTxt], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
        #answers_input_ids = SUMMARY_TOKENIZER.prepare_seq2seq_batch([allAnswersTxt], return_tensors='pt', truncation=True, padding='longest')['input_ids'].to(torch_device)
        summary_ids = SUMMARY_MODEL.generate(answers_input_ids,
                                               num_beams=4,
                                               length_penalty=0.8,
                                               repetition_penalty=2.0,
                                               min_length=5,
                                               no_repeat_ngram_size=0,
                                                do_sample=False )

        exec_sum = SUMMARY_TOKENIZER.decode(summary_ids.squeeze(), skip_special_tokens=True)
        return exec_sum

# Program to find most frequent  
# element in a list 
  
from collections import Counter 
  
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0]

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForSequenceClassification
#hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

tokenizer_nli = AutoTokenizer.from_pretrained(hg_model_hub_name, use_fast=False)
model_nli = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
model_nli.to(torch_device)
model_nli.eval()

from pyserini.search import pysearch

import re
import sys
from anglicize import anglicize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
nlp.add_pipe("merge_noun_chunks")
tokenizer = Tokenizer(nlp.vocab)
from timeit import default_timer as timer
import time
import json
import numpy as np
import statistics
import pandas as pd

dataset_path = "dev.jsonl"
output_path = dataset_path.replace(".jsonl", "_predictions.jsonl")
evaluation_path = dataset_path.replace(".jsonl", "_evaluation.jsonl")

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

import sqlalchemy as sqla

db_fullpath = "feverous_wikiv1.db"
db = sqla.create_engine("sqlite:///{}".format(db_fullpath))

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=db)
sql_session = Session()

lucene_dir = 'anserini/indexes/fever/lucene-index-fever-paragraph'
searcher = pysearch.SimpleSearcher(lucene_dir)

def get_relevant_pages(claim, FULL_CLAIM=True, SPACY_ENTITIES=True, CASE_ENTITIES=True, LINKED_PAGES=False, LINKING_PAGES=False, N_RESULTS=70):
    
    if not SPACY_ENTITIES and not CASE_ENTITIES and not LINKED_PAGES:
        FULL_CLAIM = True
    
    start_timer = timer()

    claim = nlp(claim.replace(" ", " ").replace("­", " ")) # Replaces the obnoxious space character with normal space

    keywords = set()
    entities = set()

    if SPACY_ENTITIES:
        spacy_entities = [entity.text for entity in claim.ents]
        entities.update(spacy_entities)

    if CASE_ENTITIES:
        case_entities = set()
        chunks = claim.noun_chunks
        for chunk in chunks:
            for token in tokenizer(chunk.text):
                if token.text[0].isupper():
                    case_entities.add(chunk.text)
                    break

        entities.update(case_entities)
        #print(case_entities)
        #print(entities)
        #sys.exit(0)

    keywords.update(entities)

    if not FULL_CLAIM:
        search_query = (", ".join(keywords) + '"' + '", "'.join(entities) + '"')
#            search_query = (", ".join(keywords) + ", ".join(entities))
    else:
        search_query = (claim.text + ' "' + '", "'.join(keywords) + '" "' + '", "'.join(entities) + '"')
#            search_query = (claim.text + " " + ", ".join(keywords) + ", ".join(entities))

    if FULL_CLAIM or keywords:
        try:
            lucene_hits = searcher.search(search_query.encode("utf-8"), k=N_RESULTS)
        except:
            lucene_hits = None
    else:
        lucene_hits = None

    hit_dictionary = {}

    if lucene_hits:

        for hit in lucene_hits:
            hit_dict = {"abstract": None, "real_abstract": None, "title": None}
            hit_dict['abstract']=hit.lucene_document.get("raw")
            hit_dict['real_abstract']=hit.lucene_document.get("raw")
            hit_dict['title'] = str(hit.docid)
            hit_dictionary[str(hit.docid)] = hit_dict
    
        linked_pages = set()
        if LINKED_PAGES:
            for hit in lucene_hits:
                links = re.findall(r"(?:\[\[)(.*?)(?:\|)", hit.raw)
                for link in links:
                    linked_pages.update([link.replace("_", " ").encode("latin-1", "ignore").decode("utf-8", "ignore")])
                    
        linking_pages = set()
        if LINKING_PAGES and sql_session:
            linking_results = sql_session.execute('select target, sources from inlinks where target IN ("{}")'.format('", "'.join([hit.docid.replace('"', '""') for hit in lucene_hits])))
            for row in linking_results:
                linking_pages.update(row["sources"].split(";"))
                    
        found_pages = [hit.docid for hit in lucene_hits]
        for i in range(0, len(found_pages)):

            try:
                found_pages[i] = found_pages[i].encode("latin-1", "ignore").decode("utf-8", "ignore")
                #print("Done:",found_pages[i])
            except:
                import regex
                print(found_pages[i], anglicize(found_pages[i]), regex.sub(r'[^\p{Latin}]', '', found_pages[i]).encode("latin-1").decode("utf-8"))
                
        found_pages = set(found_pages)

        hit_scores = [hit.score for hit in lucene_hits]
        hit_score_min = min(hit_scores)
        hit_score_25 = np.percentile(hit_scores, 25)
        hit_score_mean = np.mean(hit_scores)
        hit_score_median = statistics.median(hit_scores)
        hit_score_75 = np.percentile(hit_scores, 75)
        hit_score_max = max(hit_scores)
    
    else:
        
        is_found = False
        found_pages = set()
        entities = set()
        keywords = set()
        linked_pages = set()
        linking_pages = set()
        lucene_hits = []
        hit_scores = []
        hit_score_min = None
        hit_score_25 = None
        hit_score_mean = None
        hit_score_median = None
        hit_score_75 = None
        hit_score_max = None

    end_timer = timer()
    elapsed = int(end_timer - start_timer)
    elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        
    result = [claim.text, elapsed_formatted, 
                found_pages, 
                keywords, 
                linked_pages, 
                linking_pages,
                len(lucene_hits), 
                hit_score_min, hit_score_25, hit_score_mean, hit_score_median, 
                hit_score_75, hit_score_max, hit_scores]

    #print(result)
    
    result = pd.DataFrame([result], columns=["CLAIM", "ELAPSED", 
                                             "PAGES_FOUND",
                                             "KEYWORDS", 
                                             "LINKED_PAGES",
                                             "LINKING_PAGES",
                                             "N_LUCENE_HITS", 
                                             "HIT_SCORE_MIN", "HIT_SCORE_25", "HIT_SCORE_MEAN", "HIT_SCORE_MEDIAN", 
                                             "HIT_SCORE_75", "HIT_SCORE_MAX", "HIT_SCORES"])
    
    return result, hit_dictionary

def CheckEntailmentNeturalorContradict(ranked_aswers, string_all, pandasData):
    if 1 == 1:
        premise_list = []
        hypothesis_list = []
        entailment_prob = []
        neutral_prob = []
        contradict_prob = []
        final_result_list = []
        if len(ranked_aswers) > 8:
            v_range = 8
        else:
            v_range = len(ranked_aswers)
        for i in range(v_range):
            if pandasData[i][2] >= 0.4:
                max_length = 512
                candidate_answer = Summarizer(ranked_aswers[i])
                # premise = "symptoms of covid-19 range from none (asymptomatic) to severe pneumonia and it can be fatal. fever and cough are the most common symptoms in patients with 2019-ncov infection. Several people have muscle soreness or fatigue as well as ards. diarrhea, hemoptysis, headache, sore throat, shock, and other symptoms only occur in a small number of patients"
                if len(candidate_answer) > len(ranked_aswers[i]):
                    premise = ranked_aswers[i]
                else:
                    premise = candidate_answer
                # premise = "covid-19 symptom is highly various in each patient, with fever, fatigue, shortness of breath, and cough as the main presenting symptoms. patient with covid-19 may shows severe symptom with severe pneumonia and ards, mild symptom resembling simple upper respiration tract infection, or even completely asymptomatic. approximately 80 % of cases is mild "
                hypothesis = string_all
                tokenized_input_seq_pair = tokenizer_nli.encode_plus(
                    premise,
                    hypothesis,
                    max_length=max_length,
                    return_token_type_ids=True,
                    truncation=True,
                )

                input_ids = (
                    torch.Tensor(tokenized_input_seq_pair["input_ids"])
                    .long()
                    .unsqueeze(0)
                    .to(torch_device)
                )
                # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
                token_type_ids = (
                    torch.Tensor(tokenized_input_seq_pair["token_type_ids"])
                    .long()
                    .unsqueeze(0)
                    .to(torch_device)
                )
                attention_mask = (
                    torch.Tensor(tokenized_input_seq_pair["attention_mask"])
                    .long()
                    .unsqueeze(0)
                    .to(torch_device)
                )

                outputs = model_nli(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None,
                )
                # Note:
                # "id2label": {
                #     "0": "entailment",
                #     "1": "neutral",
                #     "2": "contradiction"
                # },

                predicted_probability = torch.softmax(outputs[0], dim=1)[
                    0
                ].tolist()  # batch_size only one

                # print("Premise:", premise)
                premise_list.append(premise)
                # print("Hypothesis:", hypothesis)
                hypothesis_list.append(hypothesis)
                # print("input Ans:",ranked_aswers[i])
                # print("Entailment:", predicted_probability[0])
                entailment_prob.append(predicted_probability[0])
                # print("Neutral:", predicted_probability[1])
                neutral_prob.append(predicted_probability[1])
                # print("Contradiction:", predicted_probability[2])
                contradict_prob.append(predicted_probability[2])
        if len(premise_list) > 0:
            avg_entailment = np.average(entailment_prob)
            avg_neutral = np.average(neutral_prob)
            avg_contradiction = np.average(contradict_prob)
            zipped_list = zip(entailment_prob, neutral_prob, contradict_prob)
            for key, val in enumerate(hypothesis_list):
                if neutral_prob[key] > 0.8:
                    final_result = "Neutral"
                elif entailment_prob[key] > contradict_prob[key]:
                    final_result = "True"
                    final_result_list.append(final_result)
                else:
                    final_result = "False"
                    final_result_list.append(final_result)
            if len(final_result_list) > 0:
                verdict = most_frequent(final_result_list)
            else:
                verdict = "Neutral"
        else:
            #       zipped_list = zip(0,1,0)
            zipped_list = zip([0], [1], [0])
            verdict = "Neutral"
    return zipped_list, verdict

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
embed_fn = embed_useT('/home/titanx/kaggle/working/sentence_wise_email/module/module_useT')

def get_answers(query, hit_dictionary, display_table=False):
    for idx,v in hit_dictionary.items():
        abs_dirty = v['abstract']
        real_abs_dirty = v['real_abstract']
        #abs_dirty = v['paragraph']
        # looks like the abstract value can be an empty list
        v['abstract_paragraphs'] = []
        v['abstract_full'] = ''
        v['real_abstract_full'] = ''
        v['real_abstract_paragraphs']=[]

        if abs_dirty:
            # looks like if it is a list, then the only entry is a dictionary wher text is in 'text' key
            # looks like it is broken up by paragraph if it is in that form.  lets make lists for every paragraph
            # and a new entry that is full abstract text as both could be valuable for BERT derrived QA

            if isinstance(abs_dirty, list):
                for p in abs_dirty:
                    v['abstract_paragraphs'].append(p['text'])
                    v['abstract_full'] += p['text'] + ' \n\n'

            # looks like in some cases the abstract can be straight up text so we can actually leave that alone
            if isinstance(abs_dirty, str):
                v['abstract_paragraphs'].append(abs_dirty)
                v['abstract_full'] += abs_dirty + ' \n\n'
        if real_abs_dirty:
            # looks like if it is a list, then the only entry is a dictionary wher text is in 'text' key
            # looks like it is broken up by paragraph if it is in that form.  lets make lists for every paragraph
            # and a new entry that is full abstract text as both could be valuable for BERT derrived QA


            if isinstance(real_abs_dirty, list):
                for p in real_abs_dirty:
                    v['real_abstract_paragraphs'].append(p['text'])
                    v['real_abstract_full'] += p['text'] + ' \n\n'

            # looks like in some cases the abstract can be straight up text so we can actually leave that alone
            if isinstance(abs_dirty, str):
                v['real_abstract_paragraphs'].append(real_abs_dirty)
                v['real_abstract_full'] += real_abs_dirty + ' \n\n'
        v["indexed_para"]=v["abstract_full"][len(v["real_abstract_full"]):]

#     def embed_useT(module):
#         with tf.Graph().as_default():
#             sentences = tf.compat.v1.placeholder(tf.string)
#             embed = hub.Module(module)
#             embeddings = embed(sentences)
#             session = tf.compat.v1.train.MonitoredSession()
#         return lambda x: session.run(embeddings, {sentences: x})
#     embed_fn = embed_useT('/home/titanx/kaggle/working/sentence_wise_email/module/module_useT')

    import numpy as np
    #Reconstruct text for generating the answer text from the BERT result, do the necessary text preprocessing and construct the answer.
    def reconstructText(tokens, start=0, stop=-1):
        tokens = tokens[start: stop]
        if '[SEP]' in tokens:
            sepind = tokens.index('[SEP]')
            tokens = tokens[sepind+1:]
        txt = ' '.join(tokens)
        txt = txt.replace(' ##', '')
        txt = txt.replace('##', '')
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(' .', '.')
        txt = txt.replace('( ', '(')
        txt = txt.replace(' )', ')')
        txt = txt.replace(' - ', '-')
        txt_list = txt.split(' , ')
        txt = ''
        nTxtL = len(txt_list)
        if nTxtL == 1:
            return txt_list[0]
        newList =[]
        for i,t in enumerate(txt_list):
            if i < nTxtL -1:
                if t[-1].isdigit() and txt_list[i+1][0].isdigit():
                    newList += [t,',']
                else:
                    newList += [t, ', ']
            else:
                newList += [t]
        return ''.join(newList)


    def makeBERTSQuADPrediction(document, question):
        ## we need to rewrite this function so that it chuncks the document into 250-300 word segments with
        ## 50 word overlaps on either end so that it can understand and check longer abstracts
        ## Get chuck of documents in order to overcome the BERT's max 512 token problem. First count the number of tokens. For example if it is
        ##   larger than 2048 token determine the split size and overlapping token count. Split the document into pieces to fed into BERT encoding. Overlapping
        ##      is necessary to maintain the coherence between the splits. If the token count is larger than some number 2560 token approximately this will fail.
        nWords = len(document.split())
        input_ids_all = QA_TOKENIZER.encode(question, document)
        tokens_all = QA_TOKENIZER.convert_ids_to_tokens(input_ids_all)
        overlapFac = 1.1
        if len(input_ids_all)*overlapFac > 2560:
            nSearchWords = int(np.ceil(nWords/6))
            fifth = int(np.ceil(nWords/5))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[fifth-int(nSearchWords*overlapFac/2):fifth+int(fifth*overlapFac/2)]),
                        ' '.join(docSplit[fifth*2-int(nSearchWords*overlapFac/2):fifth*2+int(fifth*overlapFac/2)]),
                        ' '.join(docSplit[fifth*3-int(nSearchWords*overlapFac/2):fifth*3+int(fifth*overlapFac/2)]),
                        ' '.join(docSplit[fifth*4-int(nSearchWords*overlapFac/2):fifth*4+int(fifth*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]

        elif len(input_ids_all)*overlapFac > 2048:
            nSearchWords = int(np.ceil(nWords/5))
            quarter = int(np.ceil(nWords/4))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[quarter-int(nSearchWords*overlapFac/2):quarter+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[quarter*2-int(nSearchWords*overlapFac/2):quarter*2+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[quarter*3-int(nSearchWords*overlapFac/2):quarter*3+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]        
            
        elif len(input_ids_all)*overlapFac > 1536:
            nSearchWords = int(np.ceil(nWords/4))
            third = int(np.ceil(nWords/3))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[third-int(nSearchWords*overlapFac/2):third+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[third*2-int(nSearchWords*overlapFac/2):third*2+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]        
            
        elif len(input_ids_all)*overlapFac > 1024:
            nSearchWords = int(np.ceil(nWords/3))
            middle = int(np.ceil(nWords/2))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[middle-int(nSearchWords*overlapFac/2):middle+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]
        elif len(input_ids_all)*overlapFac > 512:
            nSearchWords = int(np.ceil(nWords/2))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]
        else:
            input_ids = [input_ids_all]
        absTooLong = False    
        
        answers = []
        cons = []
        #print(input_ids)
        for iptIds in input_ids:
            tokens = QA_TOKENIZER.convert_ids_to_tokens(iptIds)
            #print(tokens)
            sep_index = iptIds.index(QA_TOKENIZER.sep_token_id)
            num_seg_a = sep_index + 1
            num_seg_b = len(iptIds) - num_seg_a
            segment_ids = [0]*num_seg_a + [1]*num_seg_b
            assert len(segment_ids) == len(iptIds)
            n_ids = len(segment_ids)
            #print(n_ids)
            if n_ids < 512:
                outputs = QA_MODEL(torch.tensor([iptIds]).to(torch_device), 
                                        token_type_ids=torch.tensor([segment_ids]).to(torch_device))
                #print(QA_MODEL(torch.tensor([iptIds]).to(torch_device), 
                #                        token_type_ids=torch.tensor([segment_ids]).to(torch_device)))
            else:
                #this cuts off the text if its more than 512 words so it fits in model space
                #need run multiple inferences for longer text. add to the todo. I will try to expand this to longer sequences if it is necessary. 
#                 print('****** warning only considering first 512 tokens, document is '+str(nWords)+' words long.  There are '+str(n_ids)+ ' tokens')
                absTooLong = True
                outputs= QA_MODEL(torch.tensor([iptIds[:512]]).to(torch_device), 
                                        token_type_ids=torch.tensor([segment_ids[:512]]).to(torch_device))
            start_scores=outputs.start_logits
            end_scores=outputs.end_logits
            start_scores = start_scores[:,1:-1]
            end_scores = end_scores[:,1:-1]
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            #print(answer_start, answer_end)
            answer = reconstructText(tokens, answer_start, answer_end+2)
        
            if answer.startswith('. ') or answer.startswith(', '):
                answer = answer[2:]
                
            c = start_scores[0,answer_start].item()+end_scores[0,answer_end].item()
            answers.append(answer)
            cons.append(c)
        
        maxC = max(cons)
        iMaxC = [i for i, j in enumerate(cons) if j == maxC][0]
        confidence = cons[iMaxC]
        answer = answers[iMaxC]
        
        sep_index = tokens_all.index('[SEP]')
        full_txt_tokens = tokens_all[sep_index+1:]
        
        abs_returned = reconstructText(full_txt_tokens)

        ans={}
        ans['answer'] = answer
        #print(answer)
        if answer.startswith('[CLS]') or answer_end.item() < sep_index or answer.endswith('[SEP]'):
            ans['confidence'] = -1000000
        else:
            #confidence = torch.max(start_scores) + torch.max(end_scores)
            #confidence = np.log(confidence.item())
            ans['confidence'] = confidence
        #ans['start'] = answer_start.item()
        #ans['end'] = answer_end.item()
        ans['abstract_bert'] = abs_returned
        ans['abs_too_long'] = absTooLong
        return ans

    from tqdm import tqdm
    def searchAbstracts(hit_dictionary, question):
        abstractResults = {}
#         for k,v in tqdm(hit_dictionary.items()):
        for k,v in hit_dictionary.items():
            abstract = v['abstract_full']
            indexed_para=v['indexed_para']
            if abstract:
                ans = makeBERTSQuADPrediction(abstract, question)
                if ans['answer']:
                    confidence = ans['confidence']
                    abstractResults[confidence]={}
                    abstractResults[confidence]['answer'] = ans['answer']
                    #abstractResults[confidence]['start'] = ans['start']
                    #abstractResults[confidence]['end'] = ans['end']
                    abstractResults[confidence]['abstract_bert'] = ans['abstract_bert']
                    abstractResults[confidence]['idx'] = k
                    abstractResults[confidence]['abs_too_long'] = ans['abs_too_long']

                    
        cList = list(abstractResults.keys())

        if cList:
            maxScore = max(cList)
            total = 0.0
            exp_scores = []
            for c in cList:
                s = np.exp(c-maxScore)
                exp_scores.append(s)
            total = sum(exp_scores)
            for i,c in enumerate(cList):
                abstractResults[exp_scores[i]/total] = abstractResults.pop(c)
        return abstractResults

    answers = searchAbstracts(hit_dictionary, query)

    workingPath = '/root/kaggle/working'
    import pandas as pd
    import re
    from IPython.core.display import display, HTML

    #from summarizer import Summarizer
    #summarizerModel = Summarizer()
    def displayResults(hit_dictionary, answers, question, display_table=False):
        
        question_HTML = '<div style="font-family: Times New Roman; font-size: 28px; padding-bottom:28px"><b>Query</b>: '+question+'</div>'
        #all_HTML_txt = question_HTML
        confidence = list(answers.keys())
        confidence.sort(reverse=True)
        
        confidence = list(answers.keys())
        confidence.sort(reverse=True)
        
        page_answers = dict()

        for c in confidence:
            if c>0 and c <= 1 and len(answers[c]['answer']) != 0:
                if 'idx' not in  answers[c]:
                    continue
                rowData = []
                idx = answers[c]['idx']
                title = hit_dictionary[idx]['title']

                
                full_abs = answers[c]['abstract_bert']
                bert_ans = answers[c]['answer']
                
                
                #split_abs = full_abs.split(bert_ans)
                #x=(split_abs[0][:split_abs[0].rfind('. ')+1])
                #sentance_beginning = x[x.rfind('. ')+1:] + split_abs[0][split_abs[0].rfind('. ')+1:]
                #sentance_beginning = split_abs[0][split_abs[0].rfind('.')+1:]
                x=''
                y=''
                z=''
                t=''

                # print(bert_ans)
                split_abs = full_abs.split(bert_ans)
                sentance_beginning = split_abs[0].split(" ~ ~ ")[-1]
                if len(split_abs) > 1:
                    sentance_end = split_abs[1].split(" ~ ~ ")[0]
                else:
                    sentance_end = ""
                    
                sentance_full = sentance_beginning + bert_ans+ sentance_end
#                 print(sentance_full)

                answer_ids = re.findall(r"((sentence|table|cell|section|list|item)( _ [0-9]+)+ )", sentance_full)
                answer_ids_formatted = []

                for ids in answer_ids:
                    if isinstance(ids, str):
                        candidate = ids.replace(" ", "")
                        if "_" in candidate and candidate[0] != "_" and candidate not in ["sentence", "table", "cell", "section", "list", "item"]:  # very hacky
                            answer_ids_formatted.append(candidate)
                    else:
                        for id in ids:
                            candidate = id.replace(" ", "")
                            if "_" in candidate and candidate[0] != "_" and candidate not in ["sentence", "table", "cell", "section", "list", "item"]:  # very hacky
                                answer_ids_formatted.append(candidate)
#                 print(answer_ids_formatted)
                
                page_answers[title] = answer_ids_formatted

                answers[c]['full_answer'] = sentance_beginning+bert_ans+sentance_end
                answers[c]['partial_answer'] = bert_ans+sentance_end
                answers[c]['sentence_beginning'] = sentance_beginning
                answers[c]['sentence_end'] = sentance_end
                answers[c]['title'] = title
            else:
                answers.pop(c)
        
        
        ## now rerank based on semantic similarity of the answers to the question
        ## Universal sentence encoder
        cList = list(answers.keys())
        allAnswers = [answers[c]['full_answer'] for c in cList]
        
        messages = [question]+allAnswers
        
        encoding_matrix = embed_fn(messages)
        similarity_matrix = np.inner(encoding_matrix, encoding_matrix)
        rankings = similarity_matrix[1:,0]
        
        for i,c in enumerate(cList):
            answers[rankings[i]] = answers.pop(c)

        ## now form pandas dv
        confidence = list(answers.keys())
        confidence.sort(reverse=True)
        pandasData = []
        ranked_aswers = []
        full_abs_list=[]
        for c in confidence:
            rowData=[]
            title = answers[c]['title']
            idx = answers[c]['idx']
            rowData += [idx]            
            sentance_html = '<div>' +answers[c]['sentence_beginning'] + " <font color='red'>"+answers[c]['answer']+"</font> "+answers[c]['sentence_end']+'</div>'
            answer_key = answers[c]['answer'].split("\t")[-1].strip() if "\t" in answers[c]['answer'] else answers[c]['answer'].strip()

            # rowData += [sentance_html, answer_key, c,title]
            rowData += [sentance_html, c,title]
            pandasData.append(rowData)
            ranked_aswers.append(' '.join([answers[c]['full_answer']]))
            full_abs_list.append(' '.join([answers[c]['abstract_bert']]))
        else:
            pdata2 = pandasData
            
        if display_table:
            display(HTML(question_HTML))

        #    display(HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:18px"><b>Body of Evidence:</b></div>'))

            # df = pd.DataFrame(pdata2, columns = ['Lucene ID', 'BERT-SQuAD Answer with Highlights', 'Answer Key', 'Confidence', 'Title/Link'])
            df = pd.DataFrame(pdata2, columns = ['Lucene ID', 'BERT-SQuAD Answer with Highlights', 'Confidence', 'Title/Link'])

            display(HTML(df.to_html(render_links=True, escape=False)))
        return page_answers,full_abs_list,ranked_aswers,pandasData

    return displayResults(hit_dictionary, answers, query, display_table)


# depth or breadth
# EVIDENCE_RETRIEVAL = "breadth"
EVIDENCE_RETRIEVAL = "depth"

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('{"id": "", "predicted_label": "", "predicted_evidence": ""}\n')

if EVAL and os.path.isfile(evaluation_path):
    os.remove(evaluation_path)

from tqdm import tqdm
for data in tqdm(dataset[:100]):
    
    claim = data["claim"]
    
    if not claim:
        continue
    
    hit_stats, hit_dictionary = get_relevant_pages(claim)
    
    page_answers, full_abs_list, ranked_aswers, pandasData = get_answers(claim, hit_dictionary, display_table=False)
#     print(page_answers)
#     print(pandasData)
#     continue

    evidence_probs, verdict = CheckEntailmentNeturalorContradict(ranked_aswers,claim,pandasData)

    if verdict == "False":
        predicted_label = "REFUTES"
    elif verdict == "True":
        predicted_label = "SUPPORTS"
    else:
        predicted_label = "NOT ENOUGH INFO"
        
    pages_by_confidence = [row[0] for row in pandasData]
        
    evidences = []
    
    if EVIDENCE_RETRIEVAL == "depth":
        for page in pages_by_confidence:
            idx = page_answers[page]
            for element in idx:
                if len(evidences) < 5:
                    element_type, element_id = element.split("_", 1)
                    evidence = [page, element_type, element_id]
                    if evidence not in evidences:
                        evidences.append(evidence)
                else:
                    break
    else:
        while len(evidences) < 5:
            for page in pages_by_confidence:
                idx = page_answers[page]
                for element in idx:
                    if len(evidences) < 5:
                        element_type, element_id = element.split("_", 1)
                        evidence = [page, element_type, element_id]
                        if evidence not in evidences:
                            evidences.append(evidence)
                            break
                        else:
                            continue
                    else:
                        break
        
    
    output = {"id": data["id"], "predicted_label": predicted_label, "predicted_evidence": evidences}
    
    with open(output_path, 'a', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=None)
        f.write("\n")

    if EVAL and data["label"] and data["evidence"]:

        evaluation = {"id": data["id"], "claim": data["claim"], "label": data["label"],
                      "predicted_label": predicted_label,
                      "evidence": data["evidence"], "predicted_evidence": []}

        for evidence in evidences:
            evidence_str = evidence[0] + "_" + evidence[1] + "_" + evidence[2]
            evaluation["predicted_evidence"].append(evidence_str)

        with open(evaluation_path, 'a', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=None)
            f.write("\n")

#     print(claim)
#     print(output)
    
