import json
import re
import unicodedata
import os
from collections import defaultdict
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
						weighted_evidence[claim_id][(doc, int(sent_id))] = pred 

	return weighted_evidence


def generate_RTE_file(path_to_file, path_to_outfile, path_to_evid, path_to_pred_evid, path_to_evid_2, path_to_pred_evid_2, path_to_wiki):
	weighted_evidence = load_evid(path_to_evid, path_to_pred_evid, path_to_evid_2, path_to_pred_evid_2)

	docs = set()
	for i,j in weighted_evidence.items():
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
			for line in infile:
				data = json.loads(line)
				claim_id = str(data["id"])
				claim = data["claim"]
				if "label" in data:
					label = data["label"]
				else:
					label = "NOT ENOUGH INFO"
				#evid = data["evidence"]
				predicted_pages = data["predicted_pages"]

				if claim_id in weighted_evidence:
					pred = weighted_evidence[claim_id]
					pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:5]
					pred = [list(p[0]) for p in pred if p[1] > 0]

					evid_string = ""
					last_wiki_url = ""
					if not pred:
						continue
					for i in pred:
						doc, sent_id = i
						try:
							sent = wiki_docs[doc][sent_id].split("\t")[1]
							if doc == last_wiki_url:
								evid_string = evid_string + " " + sent + " "
							else:
								evid_string = doc + " : " + evid_string + " " + sent + " "
							last_wiki_url = doc
						except Exception as e:
							print (e)
					if evid_string:
						outfile.write(claim + "\t" + evid_string + "\t" + label + "\t" + claim_id + "\n")



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
