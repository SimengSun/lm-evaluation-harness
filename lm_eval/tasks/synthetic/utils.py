import datasets
import pdb
def process_docs_unique_token_copy(dataset: datasets.Dataset):
    def _helper(doc):
      # modifies the contents of a single
      # document in our dataset.
      text = doc["text"].split(">>")
      doc["input"] = text[0] + ">>"
      doc["target"] = text[1]
      return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object

def process_results_gen(doc, results):
    completion = results[0]
    ref = doc["target"]

    return {
       'match_acc': 1 * (completion == ref)
    }    