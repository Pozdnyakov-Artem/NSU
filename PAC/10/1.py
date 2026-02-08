import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.manifold import TSNE

plt.rcParams['figure.figsize'] = 10, 10

transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5), (0.5))
                                            ])

# Downloading the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False)

test_dataset = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)


# Printing 25 random images from the training dataset
random_samples = np.random.randint(1, len(train_dataset), 25)

for idx in range(random_samples.shape[0]):
    plt.subplot(5, 5, idx + 1)
    plt.imshow(train_dataset[idx][0][0].numpy(), cmap='gray')
    plt.title(train_dataset[idx][1])
    plt.axis('off')


# plt.tight_layout()
# plt.show()

def encode_label(j):
    # 5 -> [[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784,1)) for x in data]
    labels = [encode_label(y[1]) for y in data]
    return zip(features, labels)

def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

def binary_step(x):
    return 1.0 if x>=0 else 0

def find_bias(train_data, target_digit, W):

    target = []
    other = []

    for img, labels in train_data:
        digit = np.argmax(labels)
        dot_product = np.dot(W,img)[0][0]/ np.linalg.norm(W)

        target.append(dot_product) if digit == target_digit else other.append(dot_product)

    norm_W = np.linalg.norm(W)
    if norm_W < 1e-10:
        return -np.random.normal(0, 1)

    return -((max(other) + min(target)) / 2) * norm_W

def predict_one_numb(test, W, b, target_digit):
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for img, labels in test:
        digit = np.argmax(labels)
        dot_product = (np.dot(W,img)[0][0] + b) / np.linalg.norm(W)

        if digit == target_digit:
            if bool(binary_step(dot_product)):
                true_positive += 1
            else:
                false_negative += 1
        else:
            if bool(binary_step(dot_product)):
                false_positive += 1
            else:
                true_negative += 1

    # precision = true_positive / (true_positive + false_positive)
    # recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    # F1 = 2 * (precision * recall) / (precision + recall)
    print(f"для {target_digit} точность: {accuracy}")
    # print(precision, recall, accuracy, F1)

    # return precision, recall

class Mnist:
    def __init__(self):
        self.avg_numbers=[]
        self.bias=[]

    def fit(self,train):

        for numb in range(10):
            self.avg_numbers.append(np.transpose(average_digit(train, numb)))
            self.bias.append(find_bias(train, numb, self.avg_numbers[numb]))

        self.avg_numbers = np.array(self.avg_numbers)
        self.bias = np.array(self.bias)

    def show_accuracy_for_numbers(self, test):
        for i in range(10):
            predict_one_numb(test, self.avg_numbers[i], self.bias[i], i)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict(self, test):
        img, label = test
        logs=[]
        for i in range(10):
            log = (np.dot(self.avg_numbers[i], img)[0][0] + self.bias[i]) / np.linalg.norm(self.avg_numbers[i])
            logs.append(log)
        logs = np.array(logs)
        prob = self._softmax(logs)
        ans = np.zeros(10)
        ans[np.argmax(prob)] = 1
        return ans

    def metrics(self, test_data):
        confusion_matrix = np.zeros((10, 10), dtype=int)

        for test_sample in test_data:
            features, labels = test_sample
            true_digit = np.argmax(labels)
            predicted_digit = np.argmax(self.predict(test_sample))

            confusion_matrix[true_digit][predicted_digit] += 1

        metrics = {}

        all_TP = 0
        all_TN = 0

        for digit in range(10):
            true_positive = confusion_matrix[digit][digit]
            false_positive = np.sum(confusion_matrix[:, digit]) - true_positive
            false_negative = np.sum(confusion_matrix[digit, :]) - true_positive

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

            metrics[digit] = {
                'precision': precision,
                'recall': recall,
            }
            all_TP += true_positive
            all_TN += len(test_data) - (true_positive + false_positive + false_negative)

        return (all_TP + all_TN) / (len(test_data)*10), metrics

    def visual(self, train):
        raw_data = []
        logit_data = []
        true_labels = []

        for digit in range(10):
            count = 0
            for features, label_vec in train:
                if np.argmax(label_vec) == digit and count < 30:

                    raw_data.append(features.flatten())

                    logs = []
                    for i in range(10):
                        W = model.avg_numbers[i]
                        b = model.bias[i]
                        norm_W = np.linalg.norm(W)
                        dot_product = np.dot(W, features)
                        log_val = (dot_product[0][0] + b) / norm_W
                        logs.append(log_val)
                    logit_data.append(logs)
                    true_labels.append(digit)
                    count += 1
                if count >= 30:
                    break

        raw_data = np.array(raw_data)
        logit_data = np.array(logit_data)
        true_labels = np.array(true_labels)
        print(raw_data.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        tsne_raw = TSNE(n_components=2,perplexity=30, random_state=42).fit_transform(raw_data)
        for digit in range(10):
            mask = (true_labels == digit)
            ax1.scatter(tsne_raw[mask, 0], tsne_raw[mask, 1], label=str(digit), alpha=0.7)
        ax1.set_title('Исходные изображения (784D)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        tsne_logits = TSNE(n_components=2,perplexity=5, random_state=42).fit_transform(logit_data)
        for digit in range(10):
            mask = (true_labels == digit)
            ax2.scatter(tsne_logits[mask, 0], tsne_logits[mask, 1], label=str(digit), alpha=0.7)
        ax2.set_title('Логиты модели (10D)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()



train = shape_data(train_dataset)
test = shape_data(test_dataset)

train = list(train)
test = list(test)

# avg_eight = average_digit(train, 8)
# avg_eight = np.transpose(avg_eight)
# print(np.transpose(avg_eight))
# b = find_bias(train, 8, avg_eight)
# print(b)
# predict_one_numb(test, avg_eight, -45, 8)


model = Mnist()
model.fit(train)
model.show_accuracy_for_numbers(test)
# print(np.argmax(test[0][1]),model.predict(test[0]))
# print(len(train))
model.visual(train)
# print(model.metrics(test))
val,dct = model.metrics(test)
print(val)
print(dct)