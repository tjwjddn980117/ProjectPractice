import numpy as np
# ex) np.log(part) = [[-2.302, -1.609, -0.357], [-1.204, -0.916, -1.204], [-1.609, -0.693, -1.204]]
# ex) np.mean(part, 0) = [0.2, 0.367, 0.433] (mean of each class)
# ex) np.expand_dims = [[0.2, -.367, 0.433]]
part = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
scores = []

print(f'np.log(part): {np.log(part)}')
print(f'np.expand_dims(np.mean(part, 0), 0): {np.expand_dims(np.mean(part, 0), 0)}')
print(f'np.log(np.expand_dims(np.mean(part, 0), 0)): {np.log(np.expand_dims(np.mean(part, 0), 0))}')

print()

kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
print(f'kl: {kl}')
print(f'np.sum(kl, 1): {np.sum(kl, 1)}')
print(f'np.mean(np.sum(kl, 1)): {np.mean(np.sum(kl, 1))}')
kl = np.mean(np.sum(kl, 1))
scores.append(np.exp(kl))
print(scores)
print(np.mean(scores), np.std(scores))