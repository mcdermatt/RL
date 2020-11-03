Dependencies: 
pytorch 10.2+ 
cuda enabled graphics card
python 3
pymunk
pygame
pickle
collections


If computer does not have cuda available:
add <device = torch.device("cpu")> to the top of file in main.py, agent.py, model.py and replaybuffer.py

Run interactive QWOP game with:
<python simpleHuman.py>

Train model with:
<python main.py>

View results with:
<python plotResults.py>


References:

https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

https://spinningup.openai.com/en/latest/algorithms/ddpg.html#the-q-learning-side-of-ddpg
