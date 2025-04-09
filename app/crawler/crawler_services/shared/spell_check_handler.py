# Local Imports
import re
import string
from crawler.constants.constant import SPELL_CHECK_CONSTANTS
from crawler.constants.strings import STRINGS
from crawler.crawler_services.shared.helper_method import helper_method

class spell_checker_handler:
  __spell_check = None

  def __init__(self):
      self.__spell_check = set(open(SPELL_CHECK_CONSTANTS.S_DICTIONARY_PATH).read().split())

  def init_dict(self):
    self.__spell_check = set(open(SPELL_CHECK_CONSTANTS.S_DICTIONARY_MINI_PATH).read().split())

  def validate_word(self, p_word):
    if p_word in self.__spell_check:
      return True
    else:
      return False

  @staticmethod
  def sent_tokenize(text, strip_punctuation=False, lowercase=False, min_words=1):
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)

    abbrev_patterns = [
      r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Sen|Rev|Col|Gen|Rep|Lt|Maj|Capt|Sgt|Adm|Ave|etc|e\.g|i\.e|vs|U\.S|U\.K|No|Fig)\.',
      r'\b[A-Z]\.',
      r'\b(?:[A-Z]\.){2,}',
      r'\d+\.\d+',
      r'\w+\.\w{2,3}',
      r'https?:\/\/[^ ]+',
      r'\b\w+@\w+\.\w+'
    ]
    for pattern in abbrev_patterns:
      text = re.sub(pattern, lambda m: m.group().replace('.', '<prd>'), text)

    text = re.sub(r'([😊😂👍💥❤️])', r' \1 ', text)
    text = re.sub(r'(?<=[.!?]["\')\]])\s+(?=[A-Z])', '.<stop>', text)
    text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '.<stop>', text)

    sentences = text.split('<stop>')
    sentences = [s.replace('<prd>', '.').strip() for s in sentences if s.strip()]
    if strip_punctuation:
      sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    if lowercase:
      sentences = [s.lower() for s in sentences]
    sentences = [s for s in sentences if len(s.split()) >= min_words]
    return sentences

  def clean_paragraph(self, p_text):
    sentences = self.sent_tokenize(p_text)
    cleaned_sentences = STRINGS.S_EMPTY
    for sentence in sentences:
      p_sentence = sentence.lower()
      m_valid_count = 0
      m_invalid_count = 0
      m_tokenized_sentence = p_sentence.split()
      for m_token in m_tokenized_sentence:
        if helper_method.is_stop_word(m_token) is True or self.validate_word(m_token):
          m_valid_count += 1
        else:
          m_invalid_count += 1

      if m_valid_count > 0 and m_valid_count / (m_valid_count + m_invalid_count) >= 0.60:
        if len(cleaned_sentences) > 0:
          cleaned_sentences += " - " + sentence
        else:
          cleaned_sentences = sentence

    return cleaned_sentences
