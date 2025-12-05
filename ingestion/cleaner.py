import re
from bs4 import BeautifulSoup, SoupStrainer

pattern_main = r"Главная Mail\s*"
pattern_date = r"Обновлено\s+\d+\s+[а-я].+\s+\d{4}\s+г"
pattern_sign = r"Служба поддержки Mail."
pattern_surwey = (
    r"Была ли эта информация полезной\?\s*Да\s*Нет\s+"
    r"Что именно у вас не получилось\?\s*Попробовал сделать, но не получилось\s*"
    r"Не содержит ответ на мой вопрос\s*"
    r"Недостаточно полная\s*"
    r"Тяжелая для понимания\s*"
    r"Возможность, которую я искал, не существует на проекте\s*"
    r"Отправить Спасибо!"
)
pattern_articl_navi = r"Предыдущая статья\s*[a-zA-Z0-9а-яА-Я ]+\s*Следующая статья\s*[a-zA-Z0-9а-яА-Я ]+$"
pattern_will_help = r"Служба поддержки Поможем решить проблему"
pattern_answered = (
    r"На ваш вопрос уже есть ответ Переходите в Ответы Mail.ru\xa0—\xa0"
    r"здесь пользователи уже нашли ответы Переходите в сообщество "
    r"Ответы\nMail.ru\xa0—\xa0здесь\nпользователи уже нашли ответы\s*Найти ответ"
)
pattern_cloud_help = r"—\s*Облако\s*Mail\.?ru?\s*—\s*Помощь"
pattern_subscription = r"Подписка Mail Space.*?Попробовать"

_GLOBAL_PATTERNS = [
    pattern_main,
    pattern_date,
    pattern_sign,
    pattern_will_help,
    pattern_surwey,
    pattern_articl_navi,
    pattern_answered,
    pattern_cloud_help,
    pattern_subscription,
]

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer("article"))
    text = soup.get_text(strip=True, separator=" ")
    for patt in _GLOBAL_PATTERNS:
        text = re.sub(patt, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

    
