import math
import re
import collections
import random


class NaiveBayes:
    def __init__(self):
        self.P_spam=0
        self.P_ham=0
        self.P_word_given_spam = {}
        self.P_word_given_ham = {}
        self.spam_count = 0
        self.ham_count = 0
        self.vocabulary = set()

    def delete_sym(self,email):
        text = email.lower()

        text = re.sub(r'[^\w\s]',' ',text)

        words = [word.strip() for word in text.split() if word.strip()]

        return set(words)

    def train(self,emails,labels):
        self.spam_count=0
        self.ham_count=0

        spam_word_counts = collections.defaultdict(int)
        ham_word_counts = collections.defaultdict(int)

        for email,label in zip(emails,labels):
            words = self.delete_sym(email)
            self.vocabulary.update(words)

            if label=='spam':
                self.spam_count +=1
                for word in words:
                    spam_word_counts[word] +=1
            else:
                self.ham_count +=1
                for word in words:
                    ham_word_counts[word] +=1

        total = self.spam_count + self.ham_count
        self.P_spam = self.spam_count / total
        self.P_ham = self.ham_count / total

        for word in self.vocabulary:
            in_spam = spam_word_counts.get(word, 0)
            self.P_word_given_spam[word] = (in_spam + 1) / (self.spam_count + 2)

            in_ham = ham_word_counts.get(word, 0)
            self.P_word_given_ham[word] = (in_ham + 1) / (self.ham_count + 2)

    def classy(self,email):
        words = self.delete_sym(email)

        log_score_spam = math.log(self.P_spam) if self.P_spam > 0 else 0
        log_score_ham = math.log(self.P_ham) if self.P_ham > 0 else 0

        for word in words:
            if word in self.P_word_given_spam:
                log_score_spam += math.log(self.P_word_given_spam[word])
            if word in self.P_word_given_ham:
                log_score_ham += math.log(self.P_word_given_ham[word])

        if log_score_spam > log_score_ham:
            return 'spam'
        return 'ham'

    def metr(self,test_emails,test_labels):

        tp = fp = tn = fn = 0

        total_spam = 0
        total_ham = 0

        for email,label in zip(test_emails,test_labels):
            predict = self.classy(email)

            if label == 'spam':
                total_spam += 1
                if predict == 'spam':
                    tp += 1
                else:
                    fn += 1
            else:
                total_ham += 1
                if predict == 'ham':
                    tn += 1
                else:
                    fp += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / total_spam if total_spam > 0 else 0
        specificity = tn / total_ham if total_ham > 0 else 0

        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
        }

def load_example_data():

    spam_emails = [
        "Купите наш товар! Скидка 50% только сегодня!",
        "Выиграйте миллион! Отправьте СМС сейчас!",
        "Срочно! Акция! Только сегодня скидки на все товары!",
        "Заработайте деньги быстро! Работа на дому!",
        "Бесплатный iPhone! Перейдите по ссылке!",
        "Кредит под 0%! Одобрение гарантировано!",
        "Инвестируйте в криптовалюту! Высокий доход!",
        "Похудение за неделю! Уникальная методика!",
        "Работа за границей! Высокая зарплата!",
        "Товары со скидкой 90%! Успейте купить!"
    ]

    ham_emails = [
        "Привет, как дела? Давай встретимся завтра.",
        "Напоминаю о встрече в понедельник в 10:00.",
        "Отправляю тебе документы по проекту.",
        "Спасибо за помощь вчера, очень выручил!",
        "Мама просила передать, чтобы ты позвонил.",
        "Завтра будет дождь, не забудь зонт.",
        "Отчет по работе готов, проверь пожалуйста.",
        "У нас собрание в 15:00 в конференц-зале.",
        "Привет! Что планируешь на выходные?",
        "Отправляю фотографии с отпуска."
    ]

    emails = spam_emails + ham_emails
    labels = ["spam"] * len(spam_emails) + ["ham"] * len(ham_emails)

    combined = list(zip(emails, labels))
    random.shuffle(combined)
    emails, labels = zip(*combined)

    return list(emails), list(labels)

def split_data(emails, labels, test_size=0.2):

    split_index = int(len(emails) * (1 - test_size))

    train_emails = emails[:split_index]
    train_labels = labels[:split_index]
    test_emails = emails[split_index:]
    test_labels = labels[split_index:]

    return train_emails, test_emails, train_labels, test_labels


def main():

    emails, labels = load_example_data()

    train_emails, test_emails, train_labels, test_labels = split_data(emails, labels, test_size=0.2)

    classifier = NaiveBayes()
    classifier.train(train_emails, train_labels)

    print(f"P(спам) = {classifier.P_spam:.3f}")
    print(f"P(не спам) = {classifier.P_ham:.3f}")
    print(f"Размер словаря: {len(classifier.vocabulary)} слов")

    print("\nОценка модели на тестовых данных")
    results = classifier.metr(test_emails, test_labels)

    print(f"Точность (Accuracy): {results['accuracy']:.3f}")
    print(f"Чувствительность (Sensitivity): {results['sensitivity']:.3f}")
    print(f"Специфичность (Specificity): {results['specificity']:.3f}")

    # Демонстрация классификации новых писем
    print("\n" + "=" * 50)
    print("Демонстрация классификации новых писем:")
    print("=" * 50)

    test_cases = [
        "Купите новый товар со скидкой 50%!",
        "Привет, как твои дела? Что нового?",
        "Выиграй автомобиль! Отправь СМС сейчас!",
        "Напоминаю о завтрашней встрече в 11:00",
        "Заработай миллион за день! Начни сейчас!",
        "Привет:) пошли продавать магазин",
        "Выиграй автомобиль",
        ""
    ]

    for i, email in enumerate(test_cases, 1):
        prediction = classifier.classy(email)
        print(f"{i}. '{email}' -> {prediction}")


if __name__ == "__main__":
    main()