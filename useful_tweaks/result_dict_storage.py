import json


class ResultStorage:
    def __init__(self):
        self.storage = dict()

    def _add_node(self, inner_dict, nodes, value):
        if len(nodes) > 1:
            if nodes[0] not in inner_dict:
                inner_dict[nodes[0]] = dict()
            return self._add_node(inner_dict[nodes[0]], nodes[1:], value)
        else:
            if nodes[0] not in inner_dict:
                inner_dict[nodes[0]] = []
            inner_dict[nodes[0]].append(value)

    def _mean_node(self, inner_dict):
        inner_dict["mean"] = 0.0
        for node in inner_dict:
            if type(inner_dict[node]) == list:
                inner_dict["mean"] = sum(inner_dict[node]) / len(inner_dict[node])
                return inner_dict["mean"]
            elif node != "mean":
                inner_dict["mean"] += self._mean_node(inner_dict[node])
        inner_dict["mean"] /= len(inner_dict) - 1
        return inner_dict["mean"]

    def add(self, value, nodes):
        return self._add_node(self.storage, nodes, value)

    def mean(self):
        return self._mean_node(self.storage)


result = ResultStorage()

result.add(0, ["metric1", "window1", "model1", "user1", "folds"])
result.add(1, ["metric1", "window1", "model1", "user1", "folds"])
result.add(2, ["metric1", "window1", "model1", "user1", "folds"])

result.add(0, ["metric1", "window1", "model1", "user2", "folds"])
result.add(1, ["metric1", "window1", "model1", "user2", "folds"])
result.add(2, ["metric1", "window1", "model1", "user2", "folds"])

result.add(0, ["metric1", "window1", "model2", "user1", "folds"])
result.add(1, ["metric1", "window1", "model2", "user1", "folds"])
result.add(2, ["metric1", "window1", "model2", "user1", "folds"])

result.add(0, ["metric1", "window1", "model2", "user2", "folds"])
result.add(1, ["metric1", "window1", "model2", "user2", "folds"])
result.add(2, ["metric1", "window1", "model2", "user2", "folds"])

result.add(0, ["metric1", "window1", "model3", "user1", "folds"])
result.add(1, ["metric1", "window1", "model3", "user1", "folds"])
result.add(2, ["metric1", "window1", "model3", "user1", "folds"])

result.add(0, ["metric1", "window1", "model3", "user2", "folds"])
result.add(1, ["metric1", "window1", "model3", "user2", "folds"])
result.add(2, ["metric1", "window1", "model3", "user2", "folds"])

# print(json.dumps(result.storage, indent=4))

result.mean()

print(json.dumps(result.storage, indent=4))
