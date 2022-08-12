from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter

def load_interpreter(model_path):
  model = get_validated_path(model_path, "model")
  model_path = get_model(model)
  _, nlu_model = get_model_subdirectories(model_path)
  return Interpreter.load(nlu_model)

nlu_interpreter = load_interpreter("20220801-094054.tar.gz")


import rasa.shared.nlu.training_data.loading as nlu_loading
train_data = nlu_loading.load_data("nlu.yml")

# [m.as_dict() for m in train_data.entity_examples][:3]
# [{'text':'this is my [house](place)','intent':'living'},
# {'text':'[india](place) is where i live','intent':'living'},
# {'text':'i live in [madurai](place)','intent':'living'}]

from pprint import pprint

message = train_data.entity_examples[4]
# pprint(type(message))
for component in nlu_interpreter.pipeline:
    component.process(message)
pprint(message.as_dict_nlu())


# for component in nlu_interpreter.pipeline:
#     pprint(message)