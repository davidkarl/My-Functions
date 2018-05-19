class my_dict(dict):
    def get_sub_dictionary(self, keywords, fragile=False):
        d = {}
        for k in keywords:
            try:
                d[k] = self[k]
            except KeyError:
                if fragile:
                    raise
        return d
