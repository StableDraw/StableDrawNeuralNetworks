import translators
from deep_translator import single_detection

class Translator:
    def __init__(self):
        with open("C:\\StableDraw\\StableDrawNeuralNetworks\\Neural_network_processing\\detect_language_key.txt", "r") as f:
            self.api_key = f.read()

    def detect_language(self, text):
        r = single_detection(text, api_key = self.api_key)
        return r

    def translate(self, text, source_lang = "", dest_lang = "en"):
        if source_lang == "":
            source_lang = self.detect_language(text)
        if source_lang == dest_lang:
            return (source_lang, text)
        txt = text.encode().decode("utf-8")
        f = source_lang.encode().decode("utf-8")
        t = dest_lang.encode().decode("utf-8")
        while True:
            try:
                r = translators.translate_text(txt, translator = "google", from_language = f, to_language = t)
                return (source_lang, r)
            except:
                print("Error again...")
        