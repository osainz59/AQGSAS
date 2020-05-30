import numpy as np
import mysql.connector


class VocabularyExtractor:
    WORDNET_DEFAULT_CONFIG = {
        'user': 'guest',
        'password': 'guest',
        'host': 'adimen.si.ehu.es',
        'database': 'mcr10'
    }

    def __init__(self, wordnet_config=None):

        if wordnet_config is None:
            wordnet_config = self.WORDNET_DEFAULT_CONFIG

        if 'user' not in wordnet_config:
            raise ValueError('user parameters is needed.')
        if 'password' not in wordnet_config:
            raise ValueError('password parameters is needed.')
        if 'host' not in wordnet_config:
            raise ValueError('host parameters is needed.')
        if 'database' not in wordnet_config:
            raise ValueError('database parameters is needed.')

        self._conf = wordnet_config

    @staticmethod
    def _get_word_list(konexioa, domeinua):
        c = konexioa.cursor()
        query = ("""SELECT word
            FROM `wei_domains`, `wei_eng-30_variant`, `wei_eng-30_to_ili`, `wei_ili_to_domains`
            WHERE
            (`wei_domains`.`source` LIKE %s OR `wei_domains`.`target` LIKE %s) AND `wei_domains`.`target` LIKE `wei_ili_to_domains`.`domain` AND
            `wei_eng-30_variant`.`offset` LIKE `wei_eng-30_to_ili`.`offset` AND
            `wei_eng-30_to_ili`.`iliOffset` LIKE `wei_ili_to_domains`.`iliOffset`""")
        c.execute(query, (domeinua, domeinua))
        result = [row[0].decode('utf-8') for row in c]
        c.close()
        return result

    def extract_terms(self, domain="biology", *args, **kwargs):
        # Open connection
        con = mysql.connector.connect(**self._conf)

        # Ask for the words
        words = self._get_word_list(con, domain)
        con.close()

        words = np.array([w.replace('_', ' ') for w in words])

        return words