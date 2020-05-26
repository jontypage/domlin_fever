import json
import re
import unicodedata
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import argparse


def load_evid(path_to_evid, path_to_pred_evid, path_to_evid_2, path_to_pred_evid_2):
	weighted_evidence = defaultdict(lambda: defaultdict(float))

	#path_to_pred_evid = "new_FEVER_models/IR_with_hingeloss/test_sentences_hinge_loss.tsv"
	with open(path_to_pred_evid) as preds:
		with open(path_to_evid) as infile:
			for pred, line in zip(preds, infile):
				pred = float(pred.strip())
				line = line.strip().split("\t")
				claim, doc, sent, sent_id, claim_id, rest = line[0], line[1], line[2], line[3], line[4], line[5:]
				weighted_evidence[claim_id][(doc, int(sent_id))] = pred

	with open(path_to_pred_evid_2) as preds:
		with open(path_to_evid_2) as infile:
			for pred, line in zip(preds, infile):
				line = line.strip().split("\t")
				claim, evid, doc, sent_id, claim_id,orig_doc, orig_id, rest = line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7:]
				pred = float(pred.strip())
				if pred > 0:
					if (doc, int(sent_id)) not in weighted_evidence[claim_id]:
						weighted_evidence[claim_id][(doc, orig_doc, int(sent_id), int(orig_id))] = pred 

	return weighted_evidence


def generate_RTE_file(path_to_file, path_to_outfile, path_to_evid, path_to_pred_evid, path_to_evid_2, path_to_pred_evid_2, path_to_wiki):
	weighted_evidence = load_evid(path_to_evid, path_to_pred_evid, path_to_evid_2, path_to_pred_evid_2)

	docs = set()
	for i, j in weighted_evidence.items():
		docs.update([x[0] for x in j])

	files = os.listdir(path_to_wiki)
	wiki_docs = {} 
	wiki_titles = set()
	for f in files:
		with open(os.path.join(path_to_wiki, f)) as infile:
			for line in infile:
				data = json.loads(line)
				title = data["id"]
				title = unicodedata.normalize("NFC", title)
				if title in docs:
					wiki_docs[title] = data["lines"].split("\n")
				wiki_titles.add(title)
	
	with open(path_to_outfile, "w") as outfile:
		with open(path_to_file) as infile:
			for line in tqdm(infile):
				data = json.loads(line)
				results = {}
				claim_id = str(data["id"])
				results['id'] = claim_id
				claim = data["claim"]
				evidence = [item for sublist in data['evidence'] for item in sublist]
				evidence = [item[2:] for item in evidence]
				if "label" in data:
					label = data["label"]
					verifiable = 'VERIFIABLE'
				else:
					label = "NOT ENOUGH INFO"
					verifiable = 'NOT VERIFIABLE'
				results['verifiable'] = verifiable
				results['label'] = label
				results['claim'] = claim
				results['evidence'] = data['evidence']
				predicted_pages = set()

				pred = weighted_evidence[claim_id]
				pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
				pred = [list(p[0]) for p in pred if p[1] > 0]

				results['predicted_sentences'] = []
				if not pred:
					continue
				for i in pred:
					if len(i) == 2:
						doc, sent_id = i
						predicted_pages.add(doc)
						sent = wiki_docs[doc][sent_id].split("\t")[1]
						if [doc, sent_id] in evidence:
							sent_label = label
						else:
							sent_label = 'NOT ENOUGH INFO'
						
					elif len(i) == 4:
						doc, orig_doc, sent_id, orig_id = i
						predicted_pages.add(doc)
						sent = wiki_docs[doc][sent_id].split("\t")[1]
						orig_sent = wiki_docs[orig_doc][orig_id].split("\t")[1]
						if [doc, sent_id] in evidence and [orig_doc, orig_id] in evidence:
							sent_label = label
						else:
							sent_label = 'NOT ENOUGH INFO'
						#sent = sent + ' [' + orig_doc + '] ' + orig_sent
					results['predicted_sentences'].append([doc, sent_id, sent_label, sent])
					results['predicted_pages'] = list(predicted_pages)
				outfile.write(json.dumps(results) + '\n')
					



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--infile')
	parser.add_argument('--outfile')
	parser.add_argument('--path_wiki_titles')
	parser.add_argument("--path_evid_1")
	parser.add_argument("--path_evid_1_predicted")
	parser.add_argument("--path_evid_2")
	parser.add_argument("--path_evid_2_predicted")
	args = parser.parse_args()
	generate_RTE_file(args.infile, args.outfile, args.path_evid_1, args.path_evid_1_predicted, args.path_evid_2, args.path_evid_2_predicted, args.path_wiki_titles)
