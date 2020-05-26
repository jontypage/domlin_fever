import json
import re
import unicodedata
import os
from collections import defaultdict
import sys
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



def generate_test_submission(path_to_labels, path_to_file, path_to_outfile, path_to_evid, path_to_pred_evid, path_to_evid_2, path_to_pred_evid_2):
	retrieved_evidence = load_evid(path_to_evid, path_to_pred_evid, path_to_evid_2, path_to_pred_evid_2)

	labels = {0:"SUPPORTS", 2:"REFUTES", 1:"NOT ENOUGH INFO"}
	claim_labels = {}

	with open(path_to_labels) as preds:
		for line in preds:
			data = json.loads(line)
			pred = str(data['predicted_label'])
			claim_id = str(data['id'])
			claim_labels[claim_id] = pred

	predictions = []
	with open(path_to_file) as infile:
		for line in infile:
			data = json.loads(line)
			claim_id = str(data["id"])
			claim = data["claim"]
			if claim_id not in retrieved_evidence:
				predictions.append({"id":int(claim_id), "predicted_label":"NOT ENOUGH INFO", "predicted_evidence":[["Page", 0]]})
			elif claim_id not in claim_labels:
				predictions.append({"id":int(claim_id), "predicted_label":"NOT ENOUGH INFO", "predicted_evidence":[["Page", 0]]})

			else:
				pred = retrieved_evidence[claim_id]
				label = claim_labels[claim_id]
				pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:5]
				pred = [list(p[0]) for p in pred]
				predictions.append({"id": int(claim_id), "predicted_label":label, "predicted_evidence":list(pred)})

	with open(path_to_outfile, "w") as outfile:
		for line in predictions:
			json.dump(line, outfile)
			outfile.write("\n")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--infile')
	parser.add_argument('--outfile')
	parser.add_argument('--path_wiki_titles')
	parser.add_argument("--path_evid_1")
	parser.add_argument("--path_evid_1_predicted")
	parser.add_argument("--path_evid_2")
	parser.add_argument("--path_evid_2_predicted")
	parser.add_argument("--path_papelo_predictions")
	args = parser.parse_args()

#python src/domlin/generate_submission.py fever/rte.$(basename $1) rte_models/rte.$(basename $1) fever/ir.$(basename $1) $2 fever/sentences_1.$(basename $1) sentence_retrieval_1/sentences_1.$(basename $1) fever/sentences_2.$(basename $1) sentence_retrieval_2/sentences_2.$(basename $1)

# def generate_test_submission(path_to_rte_file, path_to_labels, path_to_file, path_to_outfile, path_to_evid, path_to_pred_evid, path_to_evid_2, path_to_pred_evid_2):
generate_test_submission(args.path_papelo_predictions, args.infile, args.outfile, args.path_evid_1, args.path_evid_1_predicted, args.path_evid_2, args.path_evid_2_predicted)

