from utils import run_macaw, load_model

model_dict = load_model("allenai/macaw-3b")
res = run_macaw("Q: James went camping in the woods, but forgot to bring a hammer to bang the tent pegs in. What else might he use?\nA", model_dict)