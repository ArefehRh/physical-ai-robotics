
# 1. Data: explored a LeRobot dataset (PushT)
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

df = load_dataset("lerobot/pusht", split="train").to_pandas()

print("Episodes:", df["episode_index"].nunique())
print("State :", df["observation.state"][0])
print("Action:", df["action"][0])


# 2. Dataset checks: found bad episodes
lengths = df.groupby("episode_index").size()

print("\nShortest:", lengths.min())
print("Longest :", lengths.max())
print("Very short episodes:", list(lengths[lengths < 50].index))

plt.hist(lengths, bins=30)
plt.title("Episode lengths")
plt.savefig("episode_lengths.png")
plt.show()


# 3. Policies: trained a Behavior Cloning model
from sklearn.neural_network import MLPRegressor

X = np.stack(df["observation.state"])
Y = np.stack(df["action"])

model = MLPRegressor(max_iter=200)
model.fit(X, Y)

print("\nAction   :", Y[0])
print("Predicted:", model.predict([X[0]])[0])


# 4. Evaluation: policy success rate
error = np.linalg.norm(model.predict(X) - Y, axis=1)
print("\nSuccess rate:", round((error < 10).mean() * 100), "%")