from flair.datasets import UniversalDependenciesCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import torch
from utils import extract_loss
from analysis import plot_models


class EngineAllModels:
  def __init__(self,model_names,path_list,data_folder,tag_type="upos"):
    self.engine = Engine(path_list=path_list,data_folder=data_folder,tag_type=tag_type)
    self.model_names = model_names
    
  def train(self):
    for model_name in self.model_names:
      self.engine.get_ready(model_name)
      self.engine.train(f"{model_name}-output")
      steps,losses,accuracy = extract_loss(f"{model_name}-output/training.log")
      plot_models(model_name,steps,losses,accuracy)
      




class Engine:
  def __init__(self,path_list,data_folder,tag_type = "upos"):
    self.corpus = UniversalDependenciesCorpus(data_folder=f"./{data_folder}",
                                                train_file=path_list[0],
                                                dev_file=path_list[1],
                                                test_file=path_list[2])
    self.tag_type = tag_type
    
  def get_ready(self,model_name):
    tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)
    embeddings = TransformerWordEmbeddings(
            model=model_name,
            layers="-1",
            subtoken_pooling="first",
            fine_tune=True,
            use_context=False,
            respect_document_boundaries=False,
            pad_token="[PAD]" #optional
        )
    self.tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=self.tag_type,
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,
        )
    print("Labels:")
    tag_dictionary.get_items()
    
  def train(self,output_folder:str,EPOCHS=1):
    trainer = ModelTrainer(self.tagger, self.corpus)
    trainer.train(
            output_folder,
            learning_rate=5.0e-5,
            mini_batch_size=16,
            mini_batch_chunk_size=1,
            max_epochs=EPOCHS,
            # scheduler=OneCycleLR,
            optimizer=torch.optim.AdamW,
            embeddings_storage_mode='none',
            weight_decay=0.,
            train_with_dev=False,
            use_final_model_for_eval=True
        )
    
  
    
  
  